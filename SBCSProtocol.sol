// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

// ============================================================
//  SOULBOUND CREDIT SCORE (SBCS) PROTOCOL
//  Hackathon Demo — Full System Contracts
//  
//  Contracts:
//    1. SoulboundCreditToken   — ERC-5192 non-transferable NFT
//    2. ZKScoreVerifier        — On-chain ZK-SNARK verifier stub
//    3. GNNScoreOracle         — Off-chain GNN result ingestion
//    4. SBCSLendingPool        — Under-collateralized lending pool
//    5. SBCSGovernance         — Score-weighted governance
// ============================================================

import "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/utils/math/Math.sol";

// ─────────────────────────────────────────────────────────────
// INTERFACE: ERC-5192 Minimal Soulbound Token
// ─────────────────────────────────────────────────────────────
interface IERC5192 {
    /// @notice Emitted when the locking status changes
    event Locked(uint256 tokenId);
    event Unlocked(uint256 tokenId);

    /// @notice Returns true if the token is soulbound (non-transferable)
    function locked(uint256 tokenId) external view returns (bool);
}

// ─────────────────────────────────────────────────────────────
// CONTRACT 1: SoulboundCreditToken
// Non-transferable ERC-721 holding the user's SBCS score
// ─────────────────────────────────────────────────────────────
contract SoulboundCreditToken is ERC721, IERC5192, Ownable {
    
    // ── Storage ──────────────────────────────────────────────

    struct CreditProfile {
        uint16  score;           // 300–850 (mirrors FICO range)
        uint8   tier;            // 0=D, 1=C, 2=B, 3=A, 4=A+
        uint64  lastUpdated;     // block.timestamp
        uint64  issuedAt;
        bytes32 zkProofHash;     // Hash of the ZK proof used to set score
        bool    active;
    }

    mapping(address => uint256)      public  walletToToken;   // one token per wallet
    mapping(uint256 => CreditProfile) public profiles;

    uint256 private _nextTokenId = 1;

    address public scoreOracle;        // Only the oracle can update scores
    address public zkVerifier;         // ZK verifier contract

    // Tier thresholds
    uint16 public constant TIER_A_PLUS = 750;
    uint16 public constant TIER_A      = 700;
    uint16 public constant TIER_B      = 620;
    uint16 public constant TIER_C      = 550;

    // ── Events ───────────────────────────────────────────────

    event ScoreIssued(address indexed wallet, uint256 tokenId, uint16 score, uint8 tier);
    event ScoreUpdated(address indexed wallet, uint256 tokenId, uint16 oldScore, uint16 newScore);
    event ScoreRevoked(address indexed wallet, uint256 tokenId);

    // ── Constructor ──────────────────────────────────────────

    constructor(address _oracle, address _zkVerifier)
        ERC721("Soulbound Credit Score", "SBCS")
        Ownable(msg.sender)
    {
        scoreOracle = _oracle;
        zkVerifier  = _zkVerifier;
    }

    // ── Modifiers ────────────────────────────────────────────

    modifier onlyOracle() {
        require(msg.sender == scoreOracle, "SBCS: caller is not the oracle");
        _;
    }

    // ── Core: Issue & Update Score ───────────────────────────

    /**
     * @notice Issue a new Soulbound Credit Token to a wallet.
     * @dev Called by the oracle after ZK proof is verified.
     * @param wallet       The recipient wallet (will be the soul-bound address).
     * @param score        Credit score 300–850.
     * @param zkProofHash  keccak256 of the ZK proof bytes submitted on-chain.
     */
    function issueScore(
        address wallet,
        uint16  score,
        bytes32 zkProofHash
    ) external onlyOracle {
        require(walletToToken[wallet] == 0, "SBCS: wallet already has a token");
        require(score >= 300 && score <= 850, "SBCS: score out of range");

        uint256 tokenId = _nextTokenId++;
        _safeMint(wallet, tokenId);

        walletToToken[wallet] = tokenId;
        profiles[tokenId] = CreditProfile({
            score:       score,
            tier:        _computeTier(score),
            lastUpdated: uint64(block.timestamp),
            issuedAt:    uint64(block.timestamp),
            zkProofHash: zkProofHash,
            active:      true
        });

        emit Locked(tokenId);
        emit ScoreIssued(wallet, tokenId, score, _computeTier(score));
    }

    /**
     * @notice Update the score of an existing token.
     * @dev Score can only move by MAX_DELTA per update to prevent manipulation.
     */
    uint16 public constant MAX_DELTA = 50;

    function updateScore(
        address wallet,
        uint16  newScore,
        bytes32 zkProofHash
    ) external onlyOracle {
        uint256 tokenId = walletToToken[wallet];
        require(tokenId != 0, "SBCS: no token for this wallet");

        CreditProfile storage profile = profiles[tokenId];
        require(profile.active, "SBCS: profile is revoked");
        require(newScore >= 300 && newScore <= 850, "SBCS: score out of range");

        uint16 delta = newScore > profile.score
            ? newScore - profile.score
            : profile.score - newScore;
        require(delta <= MAX_DELTA, "SBCS: score delta too large");

        uint16 oldScore = profile.score;
        profile.score       = newScore;
        profile.tier        = _computeTier(newScore);
        profile.lastUpdated = uint64(block.timestamp);
        profile.zkProofHash = zkProofHash;

        emit ScoreUpdated(wallet, tokenId, oldScore, newScore);
    }

    /**
     * @notice Revoke a score (e.g., on confirmed fraud).
     */
    function revokeScore(address wallet) external onlyOwner {
        uint256 tokenId = walletToToken[wallet];
        require(tokenId != 0, "SBCS: no token");
        profiles[tokenId].active = false;
        emit ScoreRevoked(wallet, tokenId);
    }

    // ── Getter Helpers ───────────────────────────────────────

    function getScore(address wallet) external view returns (uint16) {
        uint256 tokenId = walletToToken[wallet];
        require(tokenId != 0 && profiles[tokenId].active, "SBCS: no active score");
        return profiles[tokenId].score;
    }

    function getTier(address wallet) external view returns (uint8) {
        uint256 tokenId = walletToToken[wallet];
        require(tokenId != 0 && profiles[tokenId].active, "SBCS: no active score");
        return profiles[tokenId].tier;
    }

    // ── ERC-5192: Soulbound Lock ─────────────────────────────

    function locked(uint256 tokenId) external pure override returns (bool) {
        return true; // Always locked — this IS the soulbound behavior
    }

    // Override all transfer functions to prevent movement
    function transferFrom(address, address, uint256) public pure override {
        revert("SBCS: soulbound tokens cannot be transferred");
    }

    function safeTransferFrom(address, address, uint256, bytes memory) public pure override {
        revert("SBCS: soulbound tokens cannot be transferred");
    }

    // ── Internal ─────────────────────────────────────────────

    function _computeTier(uint16 score) internal pure returns (uint8) {
        if (score >= TIER_A_PLUS) return 4;
        if (score >= TIER_A)      return 3;
        if (score >= TIER_B)      return 2;
        if (score >= TIER_C)      return 1;
        return 0;
    }

    function setOracle(address _oracle) external onlyOwner { scoreOracle = _oracle; }
}


// ─────────────────────────────────────────────────────────────
// CONTRACT 2: ZKScoreVerifier
// Groth16 ZK-SNARK verifier for the GNN score proof
// In production: generated by snarkjs / circom
// ─────────────────────────────────────────────────────────────
contract ZKScoreVerifier {

    // Elliptic curve pairing constants (BN128/alt_bn128)
    // In a real deployment these come from the trusted setup (Powers of Tau)
    // and are output by `snarkjs zkey export solidityverifier`

    struct Proof {
        uint256[2] a;
        uint256[2][2] b;
        uint256[2] c;
    }

    // Public inputs: [score_commitment, wallet_hash, model_version]
    // The circuit proves: given private witness (raw on-chain data),
    // the GNN model produces this score, without revealing the data.

    event ProofVerified(address indexed wallet, uint16 score, bytes32 commitment);

    /**
     * @notice Verify a ZK proof that the GNN score is correctly computed.
     * @param proof         The Groth16 proof (a, b, c points).
     * @param publicInputs  [score_as_field, wallet_keccak_mod_field, model_version]
     * @return valid        True if the proof is valid.
     *
     * HACKATHON NOTE: This is a stub. Replace with the actual
     * `verifyProof` function output by snarkjs for your circuit.
     * Real implementation uses bn128 pairing checks.
     */
    function verifyProof(
        Proof calldata proof,
        uint256[3] calldata publicInputs
    ) external returns (bool valid) {
        // ── Stub: In production, this is the snarkjs-generated pairing check ──
        // require(pairing check passes, "ZK: invalid proof");

        // Minimal sanity checks for hackathon demo:
        require(publicInputs[0] >= 300 && publicInputs[0] <= 850, "ZK: score out of range");
        require(proof.a[0] != 0 && proof.a[1] != 0, "ZK: degenerate proof point a");
        require(proof.c[0] != 0 && proof.c[1] != 0, "ZK: degenerate proof point c");

        uint16 score = uint16(publicInputs[0]);
        bytes32 commitment = keccak256(abi.encodePacked(proof.a, proof.b, proof.c, publicInputs));

        emit ProofVerified(msg.sender, score, commitment);
        return true;  // Replace with real pairing result
    }

    /**
     * @notice Returns the hash of the public GNN model being used.
     * Anyone can verify the model weights off-chain to ensure
     * the ZK circuit matches the published model.
     */
    function modelCommitment() external pure returns (bytes32) {
        // In production: keccak256 of the compiled circuit R1CS
        return 0x7f9fade1c0d57a7af66ab4ead79fade1c0d57a7af66ab4ead7c2c2eb7b11a91e;
    }
}


// ─────────────────────────────────────────────────────────────
// CONTRACT 3: GNNScoreOracle
// Decentralized oracle that bridges off-chain GNN inference
// with on-chain score minting. Uses a multi-sig committee.
// ─────────────────────────────────────────────────────────────
contract GNNScoreOracle is Ownable {

    SoulboundCreditToken public sbcsToken;
    ZKScoreVerifier      public zkVerifier;

    // Oracle committee — multiple nodes must agree on a score
    address[] public committee;
    mapping(address => bool) public isCommitteeMember;
    uint8 public quorum; // minimum approvals needed

    struct ScoreRequest {
        address wallet;
        uint16  proposedScore;
        bytes32 zkProofHash;
        uint8   approvals;
        bool    executed;
        mapping(address => bool) voted;
    }

    mapping(bytes32 => ScoreRequest) public requests; // requestId => request

    event ScoreRequested(bytes32 indexed requestId, address indexed wallet);
    event ScoreApproved(bytes32 indexed requestId, address indexed voter, uint8 approvals);
    event ScoreCommitted(bytes32 indexed requestId, address indexed wallet, uint16 score);

    constructor(
        address _sbcsToken,
        address _zkVerifier,
        address[] memory _committee,
        uint8 _quorum
    ) Ownable(msg.sender) {
        sbcsToken  = SoulboundCreditToken(_sbcsToken);
        zkVerifier = ZKScoreVerifier(_zkVerifier);
        quorum     = _quorum;

        for (uint i = 0; i < _committee.length; i++) {
            committee.push(_committee[i]);
            isCommitteeMember[_committee[i]] = true;
        }
    }

    /**
     * @notice A GNN oracle node submits a score proposal for a wallet.
     * @dev The node has already run GNN inference off-chain and holds a ZK proof.
     * @param wallet         Target wallet.
     * @param proposedScore  Computed credit score.
     * @param zkProofHash    keccak256 of the ZK proof bytes.
     */
    function proposeScore(
        address wallet,
        uint16  proposedScore,
        bytes32 zkProofHash
    ) external returns (bytes32 requestId) {
        require(isCommitteeMember[msg.sender], "Oracle: not a committee member");
        require(proposedScore >= 300 && proposedScore <= 850, "Oracle: score out of range");

        requestId = keccak256(abi.encodePacked(wallet, proposedScore, block.number, msg.sender));

        ScoreRequest storage req = requests[requestId];
        req.wallet        = wallet;
        req.proposedScore = proposedScore;
        req.zkProofHash   = zkProofHash;
        req.approvals     = 1;
        req.voted[msg.sender] = true;

        emit ScoreRequested(requestId, wallet);
        emit ScoreApproved(requestId, msg.sender, 1);

        if (quorum == 1) _executeScore(requestId);
        return requestId;
    }

    /**
     * @notice A second oracle node approves the score proposal.
     */
    function approveScore(bytes32 requestId) external {
        require(isCommitteeMember[msg.sender], "Oracle: not a committee member");
        ScoreRequest storage req = requests[requestId];
        require(!req.executed, "Oracle: already executed");
        require(!req.voted[msg.sender], "Oracle: already voted");
        require(req.wallet != address(0), "Oracle: request not found");

        req.voted[msg.sender] = true;
        req.approvals++;

        emit ScoreApproved(requestId, msg.sender, req.approvals);

        if (req.approvals >= quorum) _executeScore(requestId);
    }

    /**
     * @notice Execute the score issuance once quorum is reached.
     */
    function _executeScore(bytes32 requestId) internal {
        ScoreRequest storage req = requests[requestId];
        req.executed = true;

        // Check if wallet already has a score (update vs. issue)
        if (sbcsToken.walletToToken(req.wallet) == 0) {
            sbcsToken.issueScore(req.wallet, req.proposedScore, req.zkProofHash);
        } else {
            sbcsToken.updateScore(req.wallet, req.proposedScore, req.zkProofHash);
        }

        emit ScoreCommitted(requestId, req.wallet, req.proposedScore);
    }

    function addCommitteeMember(address member) external onlyOwner {
        require(!isCommitteeMember[member], "Oracle: already a member");
        committee.push(member);
        isCommitteeMember[member] = true;
    }
}


// ─────────────────────────────────────────────────────────────
// CONTRACT 4: SBCSLendingPool
// Under-collateralized lending based on SBCS score
// Higher score → lower collateral ratio → better terms
// ─────────────────────────────────────────────────────────────
contract SBCSLendingPool is ReentrancyGuard, Ownable {

    SoulboundCreditToken public sbcsToken;

    // ── Loan Terms by Tier ───────────────────────────────────
    // Collateral Ratio: percentage of loan value required as collateral
    // APY in basis points (500 = 5.00%)
    struct TierTerms {
        uint8  collateralRatioPct; // e.g. 120 = 120% (over-collateralized)
        uint16 apyBps;             // e.g. 420 = 4.20% APY
        uint256 maxLoanWei;        // Maximum loan size in wei
    }

    TierTerms[5] public tierTerms; // index 0–4 matching SBCS tiers

    // ── Loan Storage ─────────────────────────────────────────
    struct Loan {
        address borrower;
        uint256 principal;       // in wei
        uint256 collateral;      // in wei
        uint256 startTime;
        uint256 dueTime;
        uint16  scoreAtIssuance;
        uint16  apyBps;
        bool    repaid;
        bool    liquidated;
    }

    mapping(uint256 => Loan) public loans;
    mapping(address => uint256[]) public borrowerLoans;
    uint256 private _nextLoanId = 1;

    uint256 public totalLiquidity;
    uint256 public totalBorrowed;

    uint256 public constant LOAN_DURATION = 30 days;
    uint256 public constant LIQUIDATION_GRACE = 7 days;

    // Repayment history: tracked for score impact (reported to oracle)
    mapping(address => uint256) public successfulRepayments;
    mapping(address => uint256) public missedRepayments;

    // ── Events ───────────────────────────────────────────────

    event LoanIssued(uint256 indexed loanId, address indexed borrower, uint256 principal, uint256 collateral, uint16 apyBps);
    event LoanRepaid(uint256 indexed loanId, address indexed borrower, uint256 interest);
    event LoanLiquidated(uint256 indexed loanId, address indexed borrower, uint256 shortfall);
    event LiquidityAdded(address indexed provider, uint256 amount);

    constructor(address _sbcsToken) Ownable(msg.sender) {
        sbcsToken = SoulboundCreditToken(_sbcsToken);

        // Set default tier terms
        tierTerms[0] = TierTerms(150, 1200, 1 ether);     // D:  150% collateral, 12% APY
        tierTerms[1] = TierTerms(130, 900,  5 ether);     // C:  130% collateral, 9% APY
        tierTerms[2] = TierTerms(110, 700,  20 ether);    // B:  110% collateral, 7% APY
        tierTerms[3] = TierTerms(80,  500,  100 ether);   // A:   80% collateral, 5% APY
        tierTerms[4] = TierTerms(60,  380,  500 ether);   // A+:  60% collateral, 3.8% APY
    }

    // ── Liquidity Provision ──────────────────────────────────

    /**
     * @notice Add liquidity to the lending pool.
     */
    function addLiquidity() external payable {
        require(msg.value > 0, "Pool: zero deposit");
        totalLiquidity += msg.value;
        emit LiquidityAdded(msg.sender, msg.value);
    }

    // ── Borrow ───────────────────────────────────────────────

    /**
     * @notice Borrow ETH using your SBCS score as credit.
     * @param principal  Amount to borrow in wei.
     *
     * The required collateral is determined by the borrower's current tier:
     *   collateral = principal * tierTerms[tier].collateralRatioPct / 100
     *
     * Example — A+ tier (score 741):
     *   Borrow 10 ETH → collateral = 10 * 60/100 = 6 ETH
     *   vs. a traditional DeFi pool requiring 15 ETH (150%).
     */
    function borrow(uint256 principal) external payable nonReentrant {
        uint8  tier        = sbcsToken.getTier(msg.sender);
        uint16 score       = sbcsToken.getScore(msg.sender);
        TierTerms memory t = tierTerms[tier];

        require(principal > 0, "Pool: zero borrow");
        require(principal <= t.maxLoanWei, "Pool: exceeds tier limit");
        require(principal <= totalLiquidity - totalBorrowed, "Pool: insufficient liquidity");

        uint256 requiredCollateral = (principal * t.collateralRatioPct) / 100;
        require(msg.value >= requiredCollateral, "Pool: insufficient collateral");

        uint256 loanId = _nextLoanId++;
        loans[loanId] = Loan({
            borrower:         msg.sender,
            principal:        principal,
            collateral:       msg.value,
            startTime:        block.timestamp,
            dueTime:          block.timestamp + LOAN_DURATION,
            scoreAtIssuance:  score,
            apyBps:           t.apyBps,
            repaid:           false,
            liquidated:       false
        });

        borrowerLoans[msg.sender].push(loanId);
        totalBorrowed += principal;

        (bool ok,) = msg.sender.call{value: principal}("");
        require(ok, "Pool: ETH transfer failed");

        emit LoanIssued(loanId, msg.sender, principal, msg.value, t.apyBps);
    }

    // ── Repay ────────────────────────────────────────────────

    /**
     * @notice Repay a loan with interest, releasing collateral.
     * @param loanId  The loan to repay.
     *
     * Interest = principal * apyBps / 10000 * (elapsedSeconds / 365days)
     */
    function repay(uint256 loanId) external payable nonReentrant {
        Loan storage loan = loans[loanId];
        require(loan.borrower == msg.sender, "Pool: not your loan");
        require(!loan.repaid && !loan.liquidated, "Pool: loan already closed");

        uint256 interest = _computeInterest(loan);
        uint256 totalOwed = loan.principal + interest;
        require(msg.value >= totalOwed, "Pool: insufficient repayment");

        loan.repaid = true;
        totalBorrowed -= loan.principal;
        successfulRepayments[msg.sender]++;

        // Return excess payment
        uint256 excess = msg.value - totalOwed;
        if (excess > 0) {
            (bool ok,) = msg.sender.call{value: excess}("");
            require(ok, "Pool: excess refund failed");
        }

        // Return collateral
        (bool okCol,) = msg.sender.call{value: loan.collateral}("");
        require(okCol, "Pool: collateral return failed");

        emit LoanRepaid(loanId, msg.sender, interest);
    }

    // ── Liquidate ────────────────────────────────────────────

    /**
     * @notice Liquidate an overdue loan.
     * @dev Anyone can call this after dueTime + grace period.
     *      Collateral is seized; shortfall is recorded against the borrower's score.
     */
    function liquidate(uint256 loanId) external nonReentrant {
        Loan storage loan = loans[loanId];
        require(!loan.repaid && !loan.liquidated, "Pool: loan already closed");
        require(block.timestamp > loan.dueTime + LIQUIDATION_GRACE, "Pool: grace period active");

        loan.liquidated = true;
        totalBorrowed -= loan.principal;
        missedRepayments[loan.borrower]++;

        uint256 interest   = _computeInterest(loan);
        uint256 totalOwed  = loan.principal + interest;
        uint256 shortfall  = totalOwed > loan.collateral ? totalOwed - loan.collateral : 0;

        emit LoanLiquidated(loanId, loan.borrower, shortfall);
    }

    // ── Views ────────────────────────────────────────────────

    function getTermsForWallet(address wallet) external view
        returns (uint8 tier, uint8 collateralPct, uint16 apyBps, uint256 maxLoan)
    {
        tier         = sbcsToken.getTier(wallet);
        collateralPct = tierTerms[tier].collateralRatioPct;
        apyBps        = tierTerms[tier].apyBps;
        maxLoan       = tierTerms[tier].maxLoanWei;
    }

    function getLoan(uint256 loanId) external view returns (Loan memory) {
        return loans[loanId];
    }

    function getBorrowerLoans(address borrower) external view returns (uint256[] memory) {
        return borrowerLoans[borrower];
    }

    // ── Internal ─────────────────────────────────────────────

    function _computeInterest(Loan storage loan) internal view returns (uint256) {
        uint256 elapsed  = block.timestamp - loan.startTime;
        // interest = principal * apy * elapsed / (10000 * 365 days)
        return (loan.principal * loan.apyBps * elapsed) / (10_000 * 365 days);
    }

    function updateTierTerms(
        uint8 tier,
        uint8 collateralRatioPct,
        uint16 apyBps,
        uint256 maxLoanWei
    ) external onlyOwner {
        require(tier <= 4, "Pool: invalid tier");
        tierTerms[tier] = TierTerms(collateralRatioPct, apyBps, maxLoanWei);
    }

    receive() external payable { totalLiquidity += msg.value; }
}


// ─────────────────────────────────────────────────────────────
// CONTRACT 5: SBCSGovernance
// Score-weighted governance — higher SBCS = more voting power
// Prevents Sybil attacks on governance via the identity system
// ─────────────────────────────────────────────────────────────
contract SBCSGovernance is Ownable {

    SoulboundCreditToken public sbcsToken;

    struct Proposal {
        string  description;
        uint256 forVotes;
        uint256 againstVotes;
        uint256 startTime;
        uint256 endTime;
        bool    executed;
        mapping(address => bool) hasVoted;
    }

    mapping(uint256 => Proposal) public proposals;
    uint256 public proposalCount;

    uint256 public constant VOTING_PERIOD = 3 days;
    uint256 public constant MIN_SCORE_TO_VOTE = 550;    // C tier minimum
    uint256 public constant MIN_SCORE_TO_PROPOSE = 700; // A tier minimum

    event ProposalCreated(uint256 indexed proposalId, address indexed proposer, string description);
    event VoteCast(uint256 indexed proposalId, address indexed voter, bool support, uint256 weight);
    event ProposalExecuted(uint256 indexed proposalId, bool passed);

    constructor(address _sbcsToken) Ownable(msg.sender) {
        sbcsToken = SoulboundCreditToken(_sbcsToken);
    }

    /**
     * @notice Create a governance proposal. Requires A-tier credit.
     */
    function propose(string calldata description) external returns (uint256 proposalId) {
        uint16 score = sbcsToken.getScore(msg.sender);
        require(score >= MIN_SCORE_TO_PROPOSE, "Gov: score too low to propose");

        proposalId = ++proposalCount;
        Proposal storage p = proposals[proposalId];
        p.description = description;
        p.startTime   = block.timestamp;
        p.endTime     = block.timestamp + VOTING_PERIOD;

        emit ProposalCreated(proposalId, msg.sender, description);
    }

    /**
     * @notice Cast a vote on a proposal.
     * @dev Voting power = credit score. Sybil-proof: each soul can only vote once.
     *      Creating 100 wallets doesn't help — each needs a legitimate score >= 550.
     */
    function castVote(uint256 proposalId, bool support) external {
        Proposal storage p = proposals[proposalId];
        require(block.timestamp >= p.startTime, "Gov: not started");
        require(block.timestamp  < p.endTime,   "Gov: voting ended");
        require(!p.hasVoted[msg.sender],         "Gov: already voted");

        uint16 score = sbcsToken.getScore(msg.sender);
        require(score >= MIN_SCORE_TO_VOTE, "Gov: score below minimum");

        p.hasVoted[msg.sender] = true;
        uint256 weight = uint256(score);  // Vote weight = SBCS score

        if (support) {
            p.forVotes += weight;
        } else {
            p.againstVotes += weight;
        }

        emit VoteCast(proposalId, msg.sender, support, weight);
    }

    /**
     * @notice Execute a proposal after voting ends.
     */
    function execute(uint256 proposalId) external {
        Proposal storage p = proposals[proposalId];
        require(block.timestamp >= p.endTime, "Gov: voting ongoing");
        require(!p.executed, "Gov: already executed");

        p.executed = true;
        bool passed = p.forVotes > p.againstVotes;

        emit ProposalExecuted(proposalId, passed);
        // In production: decode and call target contract via timelock
    }

    function getProposalResult(uint256 proposalId)
        external view returns (uint256 forVotes, uint256 againstVotes, bool active)
    {
        Proposal storage p = proposals[proposalId];
        return (p.forVotes, p.againstVotes, block.timestamp < p.endTime);
    }
}
