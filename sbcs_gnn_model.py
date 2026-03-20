"""
SBCS — Soulbound Credit Score: GNN Credit Scoring Engine
=========================================================
Hackathon Implementation

Architecture:
  - GraphSAGE-based Graph Neural Network on wallet interaction graphs
  - Features: on-chain behavioral signals per wallet node
  - Output: normalized credit score 300–850
  - ZK-ready: score + proof metadata prepared for on-chain submission

Dependencies:
  pip install torch torch-geometric web3 numpy pandas scikit-learn

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GraphNorm, global_mean_pool
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import add_self_loops, degree
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import numpy as np
import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# 1. WALLET FEATURES
#    Each wallet node in the graph gets these 20 features.
#    All are normalized to [0, 1] before model input.
# ─────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "repayment_ratio",          # (repaid loans) / (total loans)
    "liquidation_count_norm",   # liquidations in lifetime (log-scaled)
    "governance_votes_norm",    # governance votes in last 90 days
    "lp_days_norm",             # total days providing liquidity
    "tx_count_norm",            # total transactions (log-scaled)
    "avg_holding_period_norm",  # mean token holding time (log-scaled)
    "fund_velocity_norm",       # funds in/out velocity (low = stable)
    "network_centrality",       # PageRank-style centrality in wallet graph
    "verified_counterparties",  # fraction of counterparties with scores
    "protocol_diversity",       # unique DeFi protocols interacted with
    "age_days_norm",            # wallet age in days (log-scaled)
    "max_balance_norm",         # max historical balance (log-scaled)
    "current_balance_norm",     # current balance (log-scaled)
    "flashloan_count_norm",     # flash loan usage (risk signal)
    "defi_yield_earned_norm",   # total yield earned (positive signal)
    "nft_activity_norm",        # NFT trading activity
    "staking_ratio",            # fraction of assets staked long-term
    "cross_chain_activity",     # cross-chain bridge usage
    "sybil_cluster_score",      # GNN-detected Sybil cluster membership
    "governance_proposal_count",# proposals created (high-trust signal)
]

NUM_FEATURES  = len(FEATURE_NAMES)   # 20
SCORE_MIN     = 300
SCORE_MAX     = 850


# ─────────────────────────────────────────────────────────────
# 2. GRAPH NEURAL NETWORK MODEL
#    GraphSAGE with 3 message-passing layers + a regression head.
#    GraphSAGE is ideal here: inductively handles new wallets
#    without retraining, naturally scales to millions of nodes.
# ─────────────────────────────────────────────────────────────

class SBCSGraphSAGE(nn.Module):
    """
    GraphSAGE credit score predictor.

    For wallet W, the model aggregates information from:
      - W's own on-chain features (direct signals)
      - W's 1-hop neighbors (who W transacts with)
      - W's 2-hop neighbors (who W's neighbors transact with)
      - W's 3-hop neighbors (broader network context)

    This multi-hop aggregation makes Sybil attacks expensive:
    a fake wallet needs a plausible *neighborhood* of transactions,
    not just its own history.
    """

    def __init__(
        self,
        in_channels:   int = NUM_FEATURES,
        hidden_channels: int = 128,
        num_layers:    int = 3,
        dropout:       float = 0.3,
    ):
        super().__init__()
        self.dropout = dropout
        self.convs   = nn.ModuleList()
        self.norms   = nn.ModuleList()

        # Layer 0: features → hidden
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.norms.append(GraphNorm(hidden_channels))

        # Layers 1..n-1: hidden → hidden
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.norms.append(GraphNorm(hidden_channels))

        # Regression head: hidden → score
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),       # raw logit → score after sigmoid + rescale
            nn.Sigmoid(),           # → (0, 1)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x:          Node features  [N, NUM_FEATURES]
            edge_index: Graph edges    [2, E]
            batch:      Batch vector   [N]  (None for single graph)
        Returns:
            score: credit score in [0,1] for each node (N,) or for
                   the root node only (used at inference)
        """
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.gelu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return self.head(x).squeeze(-1)   # (N,)

    def predict_score(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        target_node: int = 0,
    ) -> float:
        """
        Predict the SBCS score (300–850) for a single target wallet node.
        The subgraph around target_node should be provided (k-hop sampling).
        """
        self.eval()
        with torch.no_grad():
            raw = self.forward(x, edge_index)    # (N,)
            normalized = raw[target_node].item() # float in (0,1)

        # Rescale to FICO-like range
        score = SCORE_MIN + normalized * (SCORE_MAX - SCORE_MIN)
        return round(score)


# ─────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING
#    Converts raw on-chain data (from web3 / subgraph query) into
#    the normalized feature vector the GNN consumes.
# ─────────────────────────────────────────────────────────────

@dataclass
class WalletRawData:
    """Raw on-chain data for a single wallet, pulled via web3/TheGraph."""
    address:                str
    loan_count:             int   = 0
    loans_repaid:           int   = 0
    liquidation_count:      int   = 0
    governance_votes_90d:   int   = 0
    lp_days:                int   = 0
    tx_count:               int   = 0
    avg_holding_days:       float = 0.0
    inflow_7d_eth:          float = 0.0
    outflow_7d_eth:         float = 0.0
    network_centrality:     float = 0.0  # precomputed PageRank
    verified_counterparty_frac: float = 0.0
    unique_protocols:       int   = 0
    wallet_age_days:        int   = 0
    max_balance_eth:        float = 0.0
    current_balance_eth:    float = 0.0
    flashloan_count:        int   = 0
    defi_yield_eth:         float = 0.0
    nft_trades:             int   = 0
    staked_fraction:        float = 0.0
    cross_chain_txns:       int   = 0
    is_sybil_cluster:       float = 0.0  # 0=clean, 1=cluster member
    governance_proposals:   int   = 0


def _log_norm(x: float, scale: float = 1.0) -> float:
    """Log-normalize a value to [0,1] range with a soft scale."""
    return float(np.log1p(max(x, 0) / scale) / np.log1p(100))


def engineer_features(raw: WalletRawData) -> np.ndarray:
    """
    Convert raw wallet data into the GNN feature vector.
    All values are clipped to [0, 1].

    Returns:
        np.ndarray of shape (NUM_FEATURES,)
    """
    repayment_ratio = (raw.loans_repaid / max(raw.loan_count, 1))
    fund_velocity = (raw.inflow_7d_eth + raw.outflow_7d_eth) / max(raw.current_balance_eth + 1e-6, 1e-6)

    features = np.array([
        np.clip(repayment_ratio,                              0, 1),
        np.clip(_log_norm(raw.liquidation_count, scale=5),   0, 1),
        np.clip(_log_norm(raw.governance_votes_90d, scale=20),0, 1),
        np.clip(_log_norm(raw.lp_days, scale=365),            0, 1),
        np.clip(_log_norm(raw.tx_count, scale=1000),          0, 1),
        np.clip(_log_norm(raw.avg_holding_days, scale=90),    0, 1),
        np.clip(1.0 - min(fund_velocity / 10.0, 1.0),        0, 1), # inverted: low vel = good
        np.clip(raw.network_centrality,                       0, 1),
        np.clip(raw.verified_counterparty_frac,               0, 1),
        np.clip(_log_norm(raw.unique_protocols, scale=20),    0, 1),
        np.clip(_log_norm(raw.wallet_age_days, scale=730),    0, 1),
        np.clip(_log_norm(raw.max_balance_eth, scale=100),    0, 1),
        np.clip(_log_norm(raw.current_balance_eth, scale=10), 0, 1),
        np.clip(1.0 - _log_norm(raw.flashloan_count, scale=10), 0, 1), # inverted: fewer=better
        np.clip(_log_norm(raw.defi_yield_eth, scale=10),      0, 1),
        np.clip(_log_norm(raw.nft_trades, scale=50),          0, 1),
        np.clip(raw.staked_fraction,                          0, 1),
        np.clip(_log_norm(raw.cross_chain_txns, scale=20),    0, 1),
        np.clip(1.0 - raw.is_sybil_cluster,                   0, 1), # inverted
        np.clip(_log_norm(raw.governance_proposals, scale=5), 0, 1),
    ], dtype=np.float32)

    assert features.shape == (NUM_FEATURES,), f"Expected {NUM_FEATURES} features, got {features.shape}"
    return features


# ─────────────────────────────────────────────────────────────
# 4. GRAPH BUILDER
#    Constructs a PyG Data object from a set of wallets
#    and their transaction relationships.
# ─────────────────────────────────────────────────────────────

@dataclass
class TransactionEdge:
    source_addr:  str
    target_addr:  str
    tx_count:     int   = 1
    total_volume: float = 0.0  # ETH


def build_wallet_graph(
    wallets: list[WalletRawData],
    edges:   list[TransactionEdge],
    target_wallet: str,
) -> tuple[Data, int]:
    """
    Build a PyG heterogeneous graph from wallet data.

    Args:
        wallets:       List of WalletRawData for all nodes.
        edges:         Transaction edges between wallets.
        target_wallet: The wallet we want to score (becomes node 0).

    Returns:
        data:        PyG Data object ready for the GNN.
        target_idx:  Index of the target wallet in the node list.
    """
    # Build address → index mapping, target wallet is always index 0
    addr_to_idx: dict[str, int] = {}
    sorted_wallets = sorted(wallets, key=lambda w: (w.address != target_wallet, w.address))

    for i, w in enumerate(sorted_wallets):
        addr_to_idx[w.address] = i

    target_idx = addr_to_idx[target_wallet]
    assert target_idx == 0, "Target wallet must be first node"

    # Node features matrix [N, NUM_FEATURES]
    x_list = [engineer_features(w) for w in sorted_wallets]
    x = torch.tensor(np.stack(x_list), dtype=torch.float)

    # Edge index [2, E]
    src_list, dst_list = [], []
    for e in edges:
        if e.source_addr in addr_to_idx and e.target_addr in addr_to_idx:
            s = addr_to_idx[e.source_addr]
            d = addr_to_idx[e.target_addr]
            src_list.extend([s, d])  # Undirected: add both directions
            dst_list.extend([d, s])

    if src_list:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, num_nodes=len(sorted_wallets))
    return data, target_idx


# ─────────────────────────────────────────────────────────────
# 5. TRAINING
#    Supervised training on historical loan outcome data.
#    Label: did this wallet repay their loan? (binary → score range)
# ─────────────────────────────────────────────────────────────

class SBCSTrainer:
    """
    Train the GNN on historical on-chain credit data.

    The training signal comes from known repayment outcomes:
      - Wallets that repaid on time → high score labels (700–850)
      - Wallets that defaulted     → low score labels  (300–500)
      - Wallets with mixed history → middle labels      (500–700)

    In production: label generation uses a historical simulation
    over past DeFi loan events on Aave, Compound, Euler, etc.
    """

    def __init__(
        self,
        model:     SBCSGraphSAGE,
        lr:        float = 3e-4,
        weight_decay: float = 1e-5,
        device:    str   = "auto",
    ):
        self.model = model
        self.device = torch.device(
            "cuda" if (device == "auto" and torch.cuda.is_available()) else
            "mps"  if (device == "auto" and torch.backends.mps.is_available()) else
            "cpu"
        )
        self.model.to(self.device)
        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = None

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0

        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            preds = self.model(batch.x, batch.edge_index, batch.batch)

            # Normalize labels to [0, 1] for MSE loss
            labels_norm = (batch.y - SCORE_MIN) / (SCORE_MAX - SCORE_MIN)
            loss = F.mse_loss(preds, labels_norm)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item() * batch.num_graphs

        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict:
        self.model.eval()
        maes, preds_all, labels_all = [], [], []

        for batch in loader:
            batch = batch.to(self.device)
            preds_norm = self.model(batch.x, batch.edge_index, batch.batch)
            preds_score = preds_norm * (SCORE_MAX - SCORE_MIN) + SCORE_MIN
            labels_all.extend(batch.y.cpu().tolist())
            preds_all.extend(preds_score.cpu().tolist())
            maes.append(F.l1_loss(preds_score, batch.y.float()).item())

        mae = float(np.mean(maes))
        # Compute Spearman correlation (ranking accuracy)
        from scipy.stats import spearmanr
        rho, _ = spearmanr(labels_all, preds_all)
        return {"mae": mae, "spearman_rho": rho}

    def fit(self, train_loader, val_loader, epochs: int = 50):
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)
        best_val_mae = float("inf")
        best_state = None

        print(f"Training on {self.device}")
        print(f"{'Epoch':>5} {'Train Loss':>12} {'Val MAE':>10} {'Spearman ρ':>12}")
        print("─" * 45)

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            self.scheduler.step()

            if val_metrics["mae"] < best_val_mae:
                best_val_mae = val_metrics["mae"]
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

            if epoch % 5 == 0 or epoch == 1:
                print(f"{epoch:>5}  {train_loss:>12.4f}  {val_metrics['mae']:>10.2f}  {val_metrics['spearman_rho']:>12.4f}")

        # Restore best checkpoint
        if best_state:
            self.model.load_state_dict(best_state)
        print(f"\n✓ Training complete. Best val MAE: {best_val_mae:.2f} score points")
        return self.model


# ─────────────────────────────────────────────────────────────
# 6. ZK-PROOF PREPARATION
#    After scoring, prepare the proof payload for on-chain submission.
#    In production: the proof is generated by a circom circuit
#    compiled with snarkjs and the model weights.
# ─────────────────────────────────────────────────────────────

@dataclass
class ZKProofPayload:
    """
    What gets submitted to the ZKScoreVerifier contract on-chain.
    The wallet only reveals: their score and the proof.
    Raw behavioral data never leaves the client.
    """
    wallet_address:   str
    score:            int
    proof_a:          list[str]       # G1 point (2 field elements as hex)
    proof_b:          list[list[str]] # G2 point (2x2 field elements)
    proof_c:          list[str]       # G1 point
    public_inputs:    list[str]       # [score_field, wallet_hash_mod_field, model_version]
    model_commitment: str             # keccak256 of circuit R1CS
    proof_hash:       str             # keccak256 of full proof (for Solidity)
    timestamp:        int

    def to_solidity_calldata(self) -> dict:
        """Format for direct use in ethers.js / web3.py contract call."""
        return {
            "proof": {
                "a": [int(x, 16) for x in self.proof_a],
                "b": [[int(x, 16) for x in row] for row in self.proof_b],
                "c": [int(x, 16) for x in self.proof_c],
            },
            "publicInputs": [int(x, 16) for x in self.public_inputs],
        }


def prepare_zk_proof(
    wallet_address: str,
    score: int,
    model_version: int = 210,  # v2.1.0
) -> ZKProofPayload:
    """
    Prepare a ZK proof payload for on-chain submission.

    In production: this calls the snarkjs CLI or a WASM prover
    that runs the circom circuit with the model weights and user data
    as the private witness.

    For the hackathon: generates a structurally correct stub proof
    that the ZKScoreVerifier contract will accept.
    """
    import time

    # Compute public inputs
    # Field modulus for BN128 curve
    FIELD_MODULUS = 21888242871839275222246405745257275088548364400416034343698204186575808495617

    score_field      = hex(score % FIELD_MODULUS)
    wallet_hash      = int(hashlib.sha256(wallet_address.encode()).hexdigest(), 16)
    wallet_hash_mod  = hex(wallet_hash % FIELD_MODULUS)
    model_version_field = hex(model_version % FIELD_MODULUS)

    # Stub proof points (in production: output of snarkjs groth16.prove)
    proof_a = [
        hex(int(hashlib.sha256(f"a0:{wallet_address}:{score}".encode()).hexdigest(), 16) % FIELD_MODULUS),
        hex(int(hashlib.sha256(f"a1:{wallet_address}:{score}".encode()).hexdigest(), 16) % FIELD_MODULUS),
    ]
    proof_b = [
        [
            hex(int(hashlib.sha256(f"b00:{wallet_address}:{score}".encode()).hexdigest(), 16) % FIELD_MODULUS),
            hex(int(hashlib.sha256(f"b01:{wallet_address}:{score}".encode()).hexdigest(), 16) % FIELD_MODULUS),
        ],
        [
            hex(int(hashlib.sha256(f"b10:{wallet_address}:{score}".encode()).hexdigest(), 16) % FIELD_MODULUS),
            hex(int(hashlib.sha256(f"b11:{wallet_address}:{score}".encode()).hexdigest(), 16) % FIELD_MODULUS),
        ],
    ]
    proof_c = [
        hex(int(hashlib.sha256(f"c0:{wallet_address}:{score}".encode()).hexdigest(), 16) % FIELD_MODULUS),
        hex(int(hashlib.sha256(f"c1:{wallet_address}:{score}".encode()).hexdigest(), 16) % FIELD_MODULUS),
    ]

    public_inputs = [score_field, wallet_hash_mod, model_version_field]

    # Compute proof hash (what goes in the Solidity contract)
    proof_bytes = json.dumps({
        "a": proof_a, "b": proof_b, "c": proof_c,
        "pub": public_inputs
    }, sort_keys=True).encode()
    proof_hash = "0x" + hashlib.sha256(proof_bytes).hexdigest()

    model_commitment = "0x7f9fade1c0d57a7af66ab4ead79fade1c0d57a7af66ab4ead7c2c2eb7b11a91e"

    return ZKProofPayload(
        wallet_address   = wallet_address,
        score            = score,
        proof_a          = proof_a,
        proof_b          = proof_b,
        proof_c          = proof_c,
        public_inputs    = public_inputs,
        model_commitment = model_commitment,
        proof_hash       = proof_hash,
        timestamp        = int(time.time()),
    )


# ─────────────────────────────────────────────────────────────
# 7. INFERENCE PIPELINE
#    End-to-end: wallet address → SBCS score → ZK proof payload
# ─────────────────────────────────────────────────────────────

class SBCSInferencePipeline:
    """
    Production inference pipeline:
      1. Query on-chain data for the target wallet and its neighbors
      2. Engineer features
      3. Build the wallet graph
      4. Run GNN inference
      5. Prepare ZK proof payload
      6. Return score + proof for on-chain submission
    """

    def __init__(self, model_path: Optional[str] = None, device: str = "auto"):
        self.model = SBCSGraphSAGE(
            in_channels=NUM_FEATURES,
            hidden_channels=128,
            num_layers=3,
            dropout=0.0,  # No dropout at inference
        )
        if model_path and Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
            print(f"✓ Loaded model weights from {model_path}")
        else:
            print("⚠  No model weights found — using random initialization (for demo only)")

        self.device = torch.device(
            "cuda" if (device == "auto" and torch.cuda.is_available()) else "cpu"
        )
        self.model.to(self.device)
        self.model.eval()

    def score_wallet(
        self,
        target_wallet: WalletRawData,
        neighbor_wallets: list[WalletRawData],
        edges: list[TransactionEdge],
    ) -> tuple[int, ZKProofPayload]:
        """
        Score a wallet and return its SBCS score with ZK proof payload.

        Args:
            target_wallet:    The wallet to score.
            neighbor_wallets: k-hop neighbors fetched from on-chain/subgraph.
            edges:            Transaction edges in the subgraph.

        Returns:
            (score, zk_payload)
        """
        all_wallets = [target_wallet] + neighbor_wallets
        graph_data, target_idx = build_wallet_graph(
            all_wallets, edges, target_wallet.address
        )
        graph_data = graph_data.to(self.device)

        with torch.no_grad():
            raw_outputs = self.model(graph_data.x, graph_data.edge_index)
            raw_score   = raw_outputs[target_idx].item()

        score = int(SCORE_MIN + raw_score * (SCORE_MAX - SCORE_MIN))
        score = max(SCORE_MIN, min(SCORE_MAX, score))

        zk_payload = prepare_zk_proof(target_wallet.address, score)

        return score, zk_payload

    def score_and_report(
        self,
        target_wallet: WalletRawData,
        neighbor_wallets: list[WalletRawData],
        edges: list[TransactionEdge],
    ) -> dict:
        """Full pipeline with human-readable output."""
        score, zk_payload = self.score_wallet(target_wallet, neighbor_wallets, edges)

        tier_labels = ["D", "C", "B", "A", "A+"]
        tier = (
            4 if score >= 750 else
            3 if score >= 700 else
            2 if score >= 620 else
            1 if score >= 550 else 0
        )

        result = {
            "wallet": target_wallet.address,
            "score": score,
            "tier": tier_labels[tier],
            "score_range": f"{SCORE_MIN}–{SCORE_MAX}",
            "collateral_ratio_pct": [150, 130, 110, 80, 60][tier],
            "max_apy_pct": [12.0, 9.0, 7.0, 5.0, 3.8][tier],
            "proof_hash": zk_payload.proof_hash,
            "model_commitment": zk_payload.model_commitment,
            "solidity_calldata": zk_payload.to_solidity_calldata(),
        }

        print("\n" + "═" * 52)
        print("  SOULBOUND CREDIT SCORE — INFERENCE RESULT")
        print("═" * 52)
        print(f"  Wallet:      {result['wallet'][:10]}...{result['wallet'][-6:]}")
        print(f"  Score:       {result['score']} / 850")
        print(f"  Tier:        {result['tier']}")
        print(f"  Collateral:  {result['collateral_ratio_pct']}%  (vs 150% baseline)")
        print(f"  Max Borrow:  ${[1_000, 5_000, 20_000, 100_000, 500_000][tier]:,}")
        print(f"  Best APY:    {result['max_apy_pct']}%")
        print(f"  ZK Hash:     {result['proof_hash'][:20]}...")
        print("═" * 52)

        return result


# ─────────────────────────────────────────────────────────────
# 8. DEMO RUN
#    Simulates a full pipeline run with synthetic data
# ─────────────────────────────────────────────────────────────

def run_demo():
    print("\n🔷 SBCS GNN Credit Scoring Engine — Demo Run")
    print("─" * 52)

    # Synthetic target wallet (high-quality borrower)
    target = WalletRawData(
        address                   = "0x3f4a9b2c1d7e8f0a5b6c3d4e7f8a9b0c2d3e4d91c",
        loan_count                = 14,
        loans_repaid              = 14,    # perfect repayment
        liquidation_count         = 0,
        governance_votes_90d      = 7,
        lp_days                   = 280,
        tx_count                  = 4200,
        avg_holding_days          = 62.0,
        inflow_7d_eth             = 1.2,
        outflow_7d_eth            = 0.8,
        network_centrality        = 0.73,
        verified_counterparty_frac= 0.81,
        unique_protocols          = 18,
        wallet_age_days           = 980,
        max_balance_eth           = 48.5,
        current_balance_eth       = 12.3,
        flashloan_count           = 0,
        defi_yield_eth            = 3.2,
        nft_trades                = 12,
        staked_fraction           = 0.45,
        cross_chain_txns          = 8,
        is_sybil_cluster          = 0.0,
        governance_proposals      = 2,
    )

    # Synthetic neighbors (who the target transacts with)
    neighbors = [
        WalletRawData(
            address="0xabc001", loan_count=8, loans_repaid=8,
            liquidation_count=0, governance_votes_90d=3, lp_days=120,
            tx_count=800, avg_holding_days=30, inflow_7d_eth=0.5, outflow_7d_eth=0.4,
            network_centrality=0.5, verified_counterparty_frac=0.6, unique_protocols=8,
            wallet_age_days=400, max_balance_eth=10, current_balance_eth=3, flashloan_count=0,
            defi_yield_eth=1.0, nft_trades=2, staked_fraction=0.3, cross_chain_txns=2,
            is_sybil_cluster=0.0, governance_proposals=0
        ),
        WalletRawData(
            address="0xabc002", loan_count=3, loans_repaid=2,
            liquidation_count=1, governance_votes_90d=0, lp_days=30,
            tx_count=200, avg_holding_days=5, inflow_7d_eth=5, outflow_7d_eth=4.8,
            network_centrality=0.2, verified_counterparty_frac=0.3, unique_protocols=2,
            wallet_age_days=90, max_balance_eth=2, current_balance_eth=0.2, flashloan_count=3,
            defi_yield_eth=0.1, nft_trades=40, staked_fraction=0.0, cross_chain_txns=0,
            is_sybil_cluster=0.6, governance_proposals=0
        ),
    ]

    edges = [
        TransactionEdge(target.address, "0xabc001", tx_count=42, total_volume=18.5),
        TransactionEdge(target.address, "0xabc002", tx_count=3,  total_volume=0.8),
        TransactionEdge("0xabc001", "0xabc002", tx_count=1,  total_volume=0.2),
    ]

    # Build graph and score
    pipeline = SBCSInferencePipeline(model_path=None)
    result   = pipeline.score_and_report(target, neighbors, edges)

    print("\n📄 Solidity Calldata (for ZKScoreVerifier.verifyProof):")
    calldata = result["solidity_calldata"]
    print(f"  proof.a:       [{calldata['proof']['a'][0]}, ...]")
    print(f"  publicInputs:  [{calldata['publicInputs'][0]}, ...]")
    print("\n✓ Ready to submit to ZKScoreVerifier contract.\n")


if __name__ == "__main__":
    run_demo()
