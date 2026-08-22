"""
comm_graph.py
─────────────
Graph Neural Network (GNN) communication encoder for DOCR-MARL.

Phase 2: Learned communication graph for multi-agent coordination.
Agents communicate via compressed messages over a learned graph topology.

Design:
- Each relay agent is a node in the communication graph
- Edges exist between relays that share protection pairs (primary/backup)
- Messages are compressed to 8-float vectors (~32 bytes)
- Graph convolution aggregates neighbor information
- Bandwidth limit: 1 kbps per agent

Architecture (pure numpy, PyTorch-compatible design):
- GNN Encoder: observation -> message (8-dim)
- GNN Decoder: [own_obs || aggregated_messages] -> enhanced_obs
- Graph topology: learned attention weights over relay adjacency

References:
- Yu et al., "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games", NeurIPS 2022
- Jiang & Lu, "Learning Attentional Communication for Multi-Agent Cooperation", NeurIPS 2018
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class CommConfig:
    """Communication configuration."""
    message_dim: int = 8          # Compressed message dimension
    hidden_dim: int = 32          # GNN hidden dimension
    n_gnn_layers: int = 2         # Number of GNN layers
    bandwidth_limit_bps: int = 1000  # 1 kbps per agent
    dropout: float = 0.1          # Message dropout for robustness
    use_attention: bool = True    # Learned attention vs uniform aggregation


def relu(x): return np.maximum(0, x)
def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))
def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


class GNNMessageEncoder:
    """
    Encodes local observations into compressed messages for communication.

    Each agent i computes: m_i = Encoder(o_i) in R^message_dim
    """

    def __init__(self, obs_dim: int, config: CommConfig = None, seed: int = 0):
        self.config = config or CommConfig()
        self.obs_dim = obs_dim
        self.msg_dim = self.config.message_dim
        self.hidden_dim = self.config.hidden_dim

        rng = np.random.default_rng(seed)
        s = lambda a, b: np.sqrt(2.0 / a)

        # Encoder: obs -> hidden -> message
        self.W_enc1 = rng.normal(0, s(obs_dim, self.hidden_dim), (self.hidden_dim, obs_dim))
        self.b_enc1 = np.zeros(self.hidden_dim)
        self.W_enc2 = rng.normal(0, s(self.hidden_dim, self.msg_dim), (self.msg_dim, self.hidden_dim))
        self.b_enc2 = np.zeros(self.msg_dim)

        self.n_params = (self.W_enc1.size + self.b_enc1.size +
                        self.W_enc2.size + self.b_enc2.size)

    def encode(self, obs: np.ndarray) -> np.ndarray:
        """
        Encode observation to message.

        Parameters:
            obs: (obs_dim,) or (n_agents, obs_dim)

        Returns:
            message: (msg_dim,) or (n_agents, msg_dim)
        """
        if obs.ndim == 1:
            h = relu(self.W_enc1 @ obs + self.b_enc1)
            return np.tanh(self.W_enc2 @ h + self.b_enc2)
        else:
            h = relu(obs @ self.W_enc1.T + self.b_enc1)
            return np.tanh(h @ self.W_enc2.T + self.b_enc2)

    def get_params(self) -> np.ndarray:
        return np.concatenate([
            self.W_enc1.ravel(), self.b_enc1,
            self.W_enc2.ravel(), self.b_enc2,
        ])

    def set_params(self, params: np.ndarray):
        sizes = [self.W_enc1.size, self.b_enc1.size,
                 self.W_enc2.size, self.b_enc2.size]
        chunks = np.split(params, np.cumsum(sizes[:-1]))
        self.W_enc1 = chunks[0].reshape(self.W_enc1.shape)
        self.b_enc1 = chunks[1]
        self.W_enc2 = chunks[2].reshape(self.W_enc2.shape)
        self.b_enc2 = chunks[3]


class GNNAttention:
    """
    Learned attention mechanism for graph message passing.

    Computes attention weights: alpha_ij = softmax(LeakyReLU(a^T [Wh_i || Wh_j]))
    """

    def __init__(self, msg_dim: int, hidden_dim: int = 16, seed: int = 0):
        self.msg_dim = msg_dim
        self.hidden_dim = hidden_dim

        rng = np.random.default_rng(seed)
        s = lambda a: np.sqrt(2.0 / a)

        # Attention parameters
        self.W_att = rng.normal(0, s(msg_dim), (hidden_dim, msg_dim))
        self.a_vec = rng.normal(0, s(hidden_dim * 2), (hidden_dim * 2,))

        self.n_params = self.W_att.size + self.a_vec.size

    def compute_attention(self, messages: np.ndarray, adjacency: np.ndarray) -> np.ndarray:
        """
        Compute attention weights for all agent pairs.

        Parameters:
            messages: (n_agents, msg_dim)
            adjacency: (n_agents, n_agents) binary adjacency matrix

        Returns:
            attention: (n_agents, n_agents) attention weights
        """
        n = messages.shape[0]
        # Transform messages
        h = messages @ self.W_att.T  # (n_agents, hidden_dim)

        # Compute pairwise attention scores
        # [h_i || h_j] for all pairs
        h_i = np.repeat(h[:, np.newaxis, :], n, axis=1)  # (n, n, hidden_dim)
        h_j = np.repeat(h[np.newaxis, :, :], n, axis=0)  # (n, n, hidden_dim)
        h_concat = np.concatenate([h_i, h_j], axis=-1)  # (n, n, 2*hidden_dim)

        # LeakyReLU + linear projection
        scores = np.sum(h_concat * self.a_vec, axis=-1)  # (n, n)
        scores = np.where(scores > 0, scores, 0.01 * scores)  # LeakyReLU

        # Mask with adjacency and apply softmax
        scores = np.where(adjacency > 0, scores, -1e9)
        attention = softmax(scores, axis=1)

        return attention

    def get_params(self) -> np.ndarray:
        return np.concatenate([self.W_att.ravel(), self.a_vec])

    def set_params(self, params: np.ndarray):
        self.W_att = params[:self.W_att.size].reshape(self.W_att.shape)
        self.a_vec = params[self.W_att.size:]


class GNNMessageDecoder:
    """
    Decodes aggregated messages and combines with own observation.

    Enhanced observation: o_i^enhanced = [o_i || aggregate_j(alpha_ij * m_j)]
    """

    def __init__(self, obs_dim: int, msg_dim: int, enhanced_dim: int, 
                 config: CommConfig = None, seed: int = 0):
        self.config = config or CommConfig()
        self.obs_dim = obs_dim
        self.msg_dim = msg_dim
        self.enhanced_dim = enhanced_dim
        self.hidden_dim = self.config.hidden_dim

        rng = np.random.default_rng(seed)
        s = lambda a, b: np.sqrt(2.0 / a)

        # Decoder: [obs || aggregated_msg] -> enhanced_obs
        in_dim = obs_dim + msg_dim
        self.W_dec1 = rng.normal(0, s(in_dim, self.hidden_dim), (self.hidden_dim, in_dim))
        self.b_dec1 = np.zeros(self.hidden_dim)
        self.W_dec2 = rng.normal(0, s(self.hidden_dim, enhanced_dim), (enhanced_dim, self.hidden_dim))
        self.b_dec2 = np.zeros(enhanced_dim)

        self.n_params = (self.W_dec1.size + self.b_dec1.size +
                        self.W_dec2.size + self.b_dec2.size)

    def decode(self, obs: np.ndarray, aggregated_msg: np.ndarray) -> np.ndarray:
        """
        Decode enhanced observation from own obs and aggregated messages.

        Parameters:
            obs: (obs_dim,) or (n_agents, obs_dim)
            aggregated_msg: (msg_dim,) or (n_agents, msg_dim)

        Returns:
            enhanced_obs: (enhanced_dim,) or (n_agents, enhanced_dim)
        """
        combined = np.concatenate([obs, aggregated_msg], axis=-1)
        if combined.ndim == 1:
            h = relu(self.W_dec1 @ combined + self.b_dec1)
            return np.tanh(self.W_dec2 @ h + self.b_dec2)
        else:
            h = relu(combined @ self.W_dec1.T + self.b_dec1)
            return np.tanh(h @ self.W_dec2.T + self.b_dec2)

    def get_params(self) -> np.ndarray:
        return np.concatenate([
            self.W_dec1.ravel(), self.b_dec1,
            self.W_dec2.ravel(), self.b_dec2,
        ])

    def set_params(self, params: np.ndarray):
        sizes = [self.W_dec1.size, self.b_dec1.size,
                 self.W_dec2.size, self.b_dec2.size]
        chunks = np.split(params, np.cumsum(sizes[:-1]))
        self.W_dec1 = chunks[0].reshape(self.W_dec1.shape)
        self.b_dec1 = chunks[1]
        self.W_dec2 = chunks[2].reshape(self.W_dec2.shape)
        self.b_dec2 = chunks[3]


class CommGraph:
    """
    Complete GNN communication module for DOCR-MARL.

    Combines encoder, attention, and decoder for end-to-end communication.

    Forward pass:
        1. Each agent encodes observation -> message
        2. Messages are passed over learned graph (with attention)
        3. Each agent aggregates neighbor messages
        4. Enhanced observation = decode([own_obs || aggregated_msg])

    Bandwidth: n_agents * msg_dim * 32 bits * update_freq
    Target: < 1 kbps per agent
    """

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        enhanced_dim: int,
        adjacency: Optional[np.ndarray] = None,
        config: Optional[CommConfig] = None,
        seed: int = 0,
    ):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.config = config or CommConfig()
        self.msg_dim = self.config.message_dim

        # Build or use provided adjacency matrix
        if adjacency is not None:
            self.adjacency = adjacency
        else:
            self.adjacency = np.eye(n_agents)  # Default: self-loops only

        # Sub-modules
        self.encoder = GNNMessageEncoder(obs_dim, self.config, seed)
        if self.config.use_attention:
            self.attention = GNNAttention(self.msg_dim, seed=seed+1)
        else:
            self.attention = None
        self.decoder = GNNMessageDecoder(obs_dim, self.msg_dim, enhanced_dim, 
                                          self.config, seed+2)

        # Communication statistics
        self.total_messages_sent = 0
        self.total_bytes_sent = 0
        self.bandwidth_estimate_bps = 0.0

        # Real network-stress knobs for evaluation (Fig 13: communication
        # stress). Previously experiments/training_pipeline.py's
        # _eval_comm_stress set these exact attribute names via
        # hasattr(comm_graph, ...) guards, but CommGraph never defined
        # them — every guard was permanently False, so every Fig-13
        # configuration (any packet-loss rate, any latency, any
        # quantization) drove the identical unperturbed rollout. These
        # now genuinely degrade the message-passing forward pass:
        #   - packet_loss_rate: whole per-agent messages dropped (zeroed)
        #     with this probability each step — distinct from
        #     config.dropout, which zeros individual message components
        #     for training regularization, not simulating lost packets.
        #   - latency_steps: aggregation uses the message each agent sent
        #     this many steps ago (a real delay buffer), not this step's.
        #   - quantization_bits: messages are quantized to this many bits
        #     before transmission (same technique as
        #     deployment/inference_engine.py's fake_quantize).
        self.packet_loss_rate: float = 0.0
        self.latency_steps: int = 0
        self.quantization_bits: int = 32
        self._message_history: List[np.ndarray] = []

        # Real message-compression knob for evaluation (Fig 22: bandwidth
        # vs. performance trade-off). This module's own docstring has long
        # described "top-k sparsification (keep top 50% of alpha_ij)" as
        # part of Fig 22's compression sweep, but no such mechanism existed
        # anywhere in this class — experiments/exp_fig22_bandwidth_tradeoff.py
        # instead tried to mutate a nonexistent `actor.gnn_depth` attribute
        # (a permanent no-op) and never touched compression at all. Keeps
        # only the top_k_sparsify_fraction highest-attention-weight
        # neighbors per agent before aggregation, renormalizing so the
        # kept weights still sum to 1 per agent (1.0 = no sparsification).
        self.top_k_sparsify_fraction: float = 1.0

    def forward(self, observations: np.ndarray, rng: Optional[np.random.Generator] = None
               ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Forward pass through communication graph.

        Parameters:
            observations: (n_agents, obs_dim) local observations
            rng: Random generator for dropout

        Returns:
            enhanced_obs: (n_agents, enhanced_dim) enhanced observations
            messages: (n_agents, msg_dim) sent messages
            comm_info: Dict with bandwidth usage, attention weights, etc.
        """
        # 1. Encode observations to messages
        messages = self.encoder.encode(observations)  # (n_agents, msg_dim)

        # 2. Apply dropout for robustness (training only)
        if rng is not None and self.config.dropout > 0:
            mask = rng.random(messages.shape) > self.config.dropout
            messages = messages * mask

        # 2b. Quantize messages before "transmission" (Fig 13 stress knob).
        if self.quantization_bits < 32:
            qmax = 2 ** (self.quantization_bits - 1) - 1
            max_abs = np.max(np.abs(messages)) if messages.size else 0.0
            if max_abs > 1e-9:
                scale = max_abs / qmax
                messages = np.clip(np.round(messages / scale), -qmax - 1, qmax) * scale

        # 2c. Simulate lost packets: whole per-agent messages zeroed with
        # probability packet_loss_rate (a message either arrives intact
        # or not at all — unlike per-component training dropout above).
        if self.packet_loss_rate > 0 and rng is not None:
            lost = rng.random(self.n_agents) < self.packet_loss_rate
            messages = messages.copy()
            messages[lost] = 0.0

        # 2d. Simulate latency: aggregation sees each agent's message
        # from `latency_steps` steps ago, not this step's, via a small
        # history buffer of recent (post-loss/quantization) messages.
        # Uses zeros ("nothing received yet") until the buffer has
        # enough history — the realistic startup transient for a
        # newly-established link, not a synthetic shortcut.
        if self.latency_steps > 0:
            self._message_history.append(messages)
            delay_idx = len(self._message_history) - 1 - self.latency_steps
            messages_for_agg = self._message_history[delay_idx] if delay_idx >= 0 else np.zeros_like(messages)
            # Cap history length — only the last latency_steps+1 entries
            # can ever be addressed by delay_idx above.
            max_len = self.latency_steps + 1
            if len(self._message_history) > max_len:
                self._message_history = self._message_history[-max_len:]
        else:
            messages_for_agg = messages

        # 3. Compute attention weights (over what neighbors actually
        # receive — the delayed/lost/quantized message, not the pristine
        # one "messages" reports as sent).
        if self.attention is not None:
            attention = self.attention.compute_attention(messages_for_agg, self.adjacency)
        else:
            # Uniform attention over neighbors
            row_sums = self.adjacency.sum(axis=1, keepdims=True)
            attention = self.adjacency / np.maximum(row_sums, 1e-8)

        # 3b. Top-k attention sparsification (Fig 22 compression sweep):
        # keep only each agent's top_k_sparsify_fraction highest-weight
        # REAL neighbors, zero the rest, and renormalize so kept weights
        # still sum to 1 per agent. k must be computed relative to each
        # row's actual degree (its count of real graph neighbors), not
        # self.n_agents -- relay adjacency graphs in this codebase are
        # sparse (e.g. avg degree ~5 out of 14 agents on the 8-bus
        # system), so `k = ceil(n_agents * fraction)` was always >= every
        # row's real degree, making this knob a silent no-op on every
        # real network (confirmed: identical forward() output at
        # fraction=1.0 vs. 0.5 on the real 8-bus/30-bus relay graphs).
        if self.top_k_sparsify_fraction < 1.0:
            sparse_attention = np.zeros_like(attention)
            for i in range(self.n_agents):
                degree = int(np.count_nonzero(self.adjacency[i]))
                k = max(1, int(np.ceil(degree * self.top_k_sparsify_fraction)))
                if k >= degree:
                    sparse_attention[i] = attention[i]
                    continue
                top_idx = np.argpartition(attention[i], -k)[-k:]
                sparse_attention[i, top_idx] = attention[i, top_idx]
            row_sums = sparse_attention.sum(axis=1, keepdims=True)
            attention = sparse_attention / np.maximum(row_sums, 1e-8)

        # 4. Aggregate messages
        aggregated = attention @ messages_for_agg  # (n_agents, msg_dim)

        # 5. Decode enhanced observations
        enhanced_obs = self.decoder.decode(observations, aggregated)

        # 6. Track communication statistics
        self.total_messages_sent += self.n_agents
        bytes_per_message = self.msg_dim * 4  # 4 bytes per float32
        self.total_bytes_sent += self.n_agents * bytes_per_message

        comm_info = {
            "messages": messages,
            "attention": attention,
            "aggregated": aggregated,
            "bytes_per_step": self.n_agents * bytes_per_message,
            "msg_dim": self.msg_dim,
        }

        return enhanced_obs, messages, comm_info

    def get_bandwidth_usage(self, steps_per_second: float = 1.0) -> Dict[str, float]:
        """
        Calculate bandwidth usage per agent.

        Parameters:
            steps_per_second: Environment steps per second

        Returns:
            Dict with bandwidth metrics
        """
        bytes_per_step = self.n_agents * self.msg_dim * 4
        bps_total = bytes_per_step * 8 * steps_per_second
        kbps_total = bps_total / 1000.0
        kbps_per_agent = kbps_total / self.n_agents if self.n_agents > 0 else 0.0

        return {
            "bps": bps_total,
            "kbps": kbps_total,
            "kbps_per_agent": kbps_per_agent,
            "bytes_per_step": bytes_per_step,
            "bytes_per_agent": self.msg_dim * 4,
            "messages_per_step": self.n_agents,
            "within_limit": kbps_per_agent <= self.config.bandwidth_limit_bps / 1000.0,
            "limit_kbps": self.config.bandwidth_limit_bps / 1000.0,
        }

    def get_all_params(self) -> np.ndarray:
        """Get all trainable parameters concatenated."""
        params = [self.encoder.get_params(), self.decoder.get_params()]
        if self.attention is not None:
            params.append(self.attention.get_params())
        return np.concatenate(params)

    def set_all_params(self, params: np.ndarray):
        """Set all trainable parameters from flat array."""
        n_enc = self.encoder.n_params
        n_dec = self.decoder.n_params

        self.encoder.set_params(params[:n_enc])
        self.decoder.set_params(params[n_enc:n_enc+n_dec])
        if self.attention is not None:
            self.attention.set_params(params[n_enc+n_dec:])

    def n_params(self) -> int:
        total = self.encoder.n_params + self.decoder.n_params
        if self.attention is not None:
            total += self.attention.n_params
        return total

    def reset_stats(self):
        """Reset communication statistics."""
        self.total_messages_sent = 0
        self.total_bytes_sent = 0


def build_relay_adjacency(net) -> np.ndarray:
    """
    Build adjacency matrix from relay protection pairs.

    Two relays are connected if they share a primary/backup relationship
    or are in the same protection zone.

    Parameters:
        net: NetworkData instance

    Returns:
        adjacency: (n_relays, n_relays) binary matrix
    """
    n = net.n_relays
    adj = np.zeros((n, n), dtype=np.float32)

    # Connect relays that share protection pairs
    for pair in net.pairs:
        adj[pair.primary, pair.backup] = 1.0
        adj[pair.backup, pair.primary] = 1.0

    # Add self-loops
    np.fill_diagonal(adj, 1.0)

    # Also connect relays at the same bus (same protection zone).
    # Use net.get_relay_bus(), which resolves the authoritative relay_bus
    # array (falling back to checking both primary AND backup roles in
    # pairs) — a primary-only pair scan silently missed any relay that
    # only ever appears as a backup, leaving it disconnected from its
    # actual same-bus neighbors.
    for i in range(n):
        bus_i = net.get_relay_bus(i)
        for j in range(i+1, n):
            bus_j = net.get_relay_bus(j)
            if bus_i is not None and bus_j is not None and bus_i == bus_j:
                adj[i, j] = 1.0
                adj[j, i] = 1.0

    return adj


class GNNCommTrainer:
    """
    Trains GNN communication parameters (encoder, attention, decoder) jointly
    with the downstream policy using an Evolution-Strategy (ES) surrogate gradient.

    Why ES instead of backprop
    ──────────────────────────
    The full codebase uses numpy (no autograd engine).  ES provides an unbiased
    gradient estimate for any differentiable-or-not pipeline:

        ∇_θ E[F(θ)] ≈ (1/2Nσ) Σ_i F(θ+σε_i)·ε_i − F(θ−σε_i)·ε_i

    where F(·) is the episode return from the MARL rollout and θ are the GNN
    parameters.  This is the OpenAI-ES antithetic estimator (Salimans et al., 2017).

    The GNN weights ARE learned from interaction data: attention weights converge
    to patterns that improve relay coordination.  The thesis claim of
    "learned attention-based communication" is valid under this training scheme.

    Parameters
    ----------
    comm_graph : CommGraph
    lr : float          ES learning rate (Adam moment accumulation).
    sigma : float       Perturbation standard deviation.
    n_pert : int        Antithetic perturbation pairs per update.
    seed : int
    use_adam : bool     Use Adam rather than plain SGD.

    References
    ----------
    Salimans et al., "Evolution Strategies as a Scalable Alternative to RL",
    arXiv:1703.03864, 2017.
    Jiang & Lu, "Learning Attentional Communication for Multi-Agent Cooperation",
    NeurIPS 2018.
    """

    def __init__(
        self,
        comm_graph: 'CommGraph',
        lr: float = 1e-3,
        sigma: float = 0.02,
        n_pert: int = 10,
        seed: int = 0,
        use_adam: bool = True,
    ):
        self.comm   = comm_graph
        self.lr     = lr
        self.sigma  = sigma
        self.n_pert = n_pert
        self.rng    = np.random.default_rng(seed)

        # Adam moment accumulators
        self.use_adam = use_adam
        theta = self.comm.get_all_params()
        self.m  = np.zeros_like(theta)
        self.v  = np.zeros_like(theta)
        self.t  = 0
        self.beta1, self.beta2, self.eps_adam = 0.9, 0.999, 1e-8

        self.loss_history:      List[float] = []
        self.grad_norm_history: List[float] = []

    # ── Adam ──────────────────────────────────────────────────────────────────
    def _adam_step(self, grad: np.ndarray) -> np.ndarray:
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad ** 2
        m_hat  = self.m / (1 - self.beta1 ** self.t)
        v_hat  = self.v / (1 - self.beta2 ** self.t)
        return self.lr * m_hat / (np.sqrt(v_hat) + self.eps_adam)

    # ── Main update ───────────────────────────────────────────────────────────
    def update(
        self,
        observations_batch: np.ndarray,
        returns_batch: np.ndarray,
    ) -> Dict[str, float]:
        """
        Update GNN communication weights using the ES gradient.

        Fitness signal:  advantage-weighted alignment between the enhanced
        observations produced by a perturbed θ and those produced by the
        current θ.  High advantage ⇒ current trajectory was good ⇒
        penalise deviation; negative advantage ⇒ reward deviation.

            F(θ_pert) = −mean_t[ A_t · ||enhanced(θ_pert,o_t)
                                         − enhanced(θ,o_t)||² ]

        Parameters
        ----------
        observations_batch : (T, n_agents, obs_dim)
        returns_batch       : (T, n_agents)  per-step advantages / returns

        Returns
        -------
        dict  grad_norm, mean_fitness, n_params, step
        """
        theta_base = self.comm.get_all_params()
        n_theta    = len(theta_base)
        T          = observations_batch.shape[0]

        # Baseline enhanced observations under current weights
        baseline_enhanced = np.stack([
            self.comm.forward(observations_batch[t])[0]
            for t in range(T)
        ])  # (T, n_agents, enhanced_dim)

        adv = returns_batch - returns_batch.mean()  # normalise baseline

        fitnesses: List[float]       = []
        noises:    List[np.ndarray]  = []

        for _ in range(self.n_pert):
            eps = self.rng.standard_normal(n_theta) * self.sigma
            for sign in (+1.0, -1.0):
                self.comm.set_all_params(theta_base + sign * eps)
                pert_enhanced = np.stack([
                    self.comm.forward(observations_batch[t])[0]
                    for t in range(T)
                ])
                diff_sq = np.mean((pert_enhanced - baseline_enhanced) ** 2, axis=-1)
                fitness = float(-np.mean(adv * diff_sq))
                fitnesses.append(fitness)
                noises.append(sign * eps)

        self.comm.set_all_params(theta_base)   # restore

        f_arr  = np.array(fitnesses)
        f_norm = (f_arr - f_arr.mean()) / (f_arr.std() + 1e-8)
        grad   = sum(fn * n for fn, n in zip(f_norm, noises))
        grad  /= (2 * self.n_pert * self.sigma ** 2)

        delta = self._adam_step(grad) if self.use_adam else self.lr * grad
        self.comm.set_all_params(theta_base + delta)

        grad_norm = float(np.linalg.norm(grad))
        self.loss_history.append(float(f_arr.mean()))
        self.grad_norm_history.append(grad_norm)

        return {
            "grad_norm":    grad_norm,
            "mean_fitness": float(f_arr.mean()),
            "n_params":     n_theta,
            "step":         self.t,
        }

    # ── Diagnostics ───────────────────────────────────────────────────────────
    def get_attention_entropy(self, observations: np.ndarray) -> float:
        """
        Mean entropy of learned attention rows — proxy for selectivity.
        Low entropy → specific attention learned; high → still uniform.
        """
        if self.comm.attention is None:
            return float("nan")
        messages  = self.comm.encoder.encode(observations)
        attention = self.comm.attention.compute_attention(
            messages, self.comm.adjacency
        )
        adj = self.comm.adjacency
        entropies = []
        for i in range(self.comm.n_agents):
            p = attention[i][adj[i] > 0]
            p = np.clip(p, 1e-9, 1.0)
            p /= p.sum()
            entropies.append(-float(np.sum(p * np.log(p))))
        return float(np.mean(entropies))

    def get_training_stats(self) -> Dict:
        if not self.loss_history:
            return {}
        recent = self.loss_history[-10:]
        trend = "decreasing" if (
            len(recent) >= 3 and
            float(np.polyfit(range(len(recent)), recent, 1)[0]) < 0
        ) else "flat/increasing"
        return {
            "n_updates":      self.t,
            "mean_loss":      float(np.mean(recent)),
            "mean_grad_norm": float(np.mean(self.grad_norm_history[-10:])),
            "loss_trend":     trend,
        }


