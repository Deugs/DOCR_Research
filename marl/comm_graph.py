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

# torch is required as of 2026-10-22 (see marl/marl_agents.py's
# module-level comment for the full history of the ES-to-backprop
# rewrite this module is part of). Import order matters on at least one
# real development machine used on this project: torch imported before
# pandapower segfaults with zero diagnostic output (native OpenMP/MKL
# runtime conflict) — every real entry point already imports
# network/marl_env (which transitively imports pandapower) before this
# module, so this is safe in practice, but a fresh script/notebook must
# preserve that order too.
import torch


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


class _FlatParamsMixin:
    """get_params()/set_params()/n_params adapters over a torch.nn.Module's
    real parameters -- kept for backward compatibility with ES-era
    tests/fixtures written against GNNMessageEncoder/GNNAttention/
    GNNMessageDecoder's old hand-rolled-NumPy flat-vector API (see
    tests/test_phase2.py). n_params is a plain attribute-style property
    (no parens at call sites), matching the old code's `self.n_params = ...`
    set once in __init__ rather than a method."""

    def get_params(self) -> np.ndarray:
        with torch.no_grad():
            return torch.nn.utils.parameters_to_vector(self.parameters()).cpu().numpy()

    def set_params(self, params: np.ndarray):
        p_t = torch.as_tensor(np.asarray(params), dtype=torch.float32)
        torch.nn.utils.vector_to_parameters(p_t, self.parameters())

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def _device(self):
        """This module's actual parameter device -- NumPy wrapper methods
        (encode/compute_attention/decode) must build their input tensors
        on this device, not default to CPU, or they crash the moment the
        module has been moved to CUDA (mismatched-device matmul)."""
        return next(self.parameters()).device


class GNNMessageEncoder(_FlatParamsMixin, torch.nn.Module):
    """
    Encodes local observations into compressed messages for communication.

    Each agent i computes: m_i = Encoder(o_i) in R^message_dim

    2026-10-22: rewritten from hand-rolled NumPy to a real torch.nn.Module
    (see marl/marl_agents.py's module-level comment for the full ES-to-
    backprop rewrite this is part of). encode() keeps the original
    NumPy-in/NumPy-out, no-grad public interface for the rollout-collection
    hot path; encode_t() is the new differentiable version used by
    CommGraph.forward_t() for joint training with the downstream policy.
    """

    def __init__(self, obs_dim: int, config: CommConfig = None, seed: int = 0):
        super().__init__()
        self.config = config or CommConfig()
        self.obs_dim = obs_dim
        self.msg_dim = self.config.message_dim
        self.hidden_dim = self.config.hidden_dim

        gen = torch.Generator().manual_seed(seed)
        self.fc1 = torch.nn.Linear(obs_dim, self.hidden_dim)
        self.fc2 = torch.nn.Linear(self.hidden_dim, self.msg_dim)
        for layer, fan_in in ((self.fc1, obs_dim), (self.fc2, self.hidden_dim)):
            torch.nn.init.normal_(layer.weight, mean=0.0, std=(2.0 / fan_in) ** 0.5, generator=gen)
            torch.nn.init.zeros_(layer.bias)

    def encode_t(self, obs_t: torch.Tensor) -> torch.Tensor:
        """Differentiable encode: obs_t (..., obs_dim) -> message (..., msg_dim)."""
        h = torch.relu(self.fc1(obs_t))
        return torch.tanh(self.fc2(h))

    def encode(self, obs: np.ndarray) -> np.ndarray:
        """NumPy in/out, no-grad -- the rollout-collection hot path."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self._device)
        with torch.no_grad():
            return self.encode_t(obs_t).cpu().numpy()


class GNNAttention(_FlatParamsMixin, torch.nn.Module):
    """
    Learned attention mechanism for graph message passing.

    Computes attention weights: alpha_ij = softmax(LeakyReLU(a^T [Wh_i || Wh_j]))
    """

    def __init__(self, msg_dim: int, hidden_dim: int = 16, seed: int = 0):
        super().__init__()
        self.msg_dim = msg_dim
        self.hidden_dim = hidden_dim

        gen = torch.Generator().manual_seed(seed)
        self.W_att = torch.nn.Linear(msg_dim, hidden_dim, bias=False)
        torch.nn.init.normal_(self.W_att.weight, mean=0.0, std=(2.0 / msg_dim) ** 0.5, generator=gen)
        self.a_vec = torch.nn.Parameter(
            torch.empty(hidden_dim * 2).normal_(0.0, (2.0 / (hidden_dim * 2)) ** 0.5, generator=gen))

    def compute_attention_t(self, messages_t: torch.Tensor, adjacency_t: torch.Tensor) -> torch.Tensor:
        """Differentiable attention: messages_t (n, msg_dim), adjacency_t
        (n, n) -> attention (n, n)."""
        n = messages_t.shape[0]
        h = self.W_att(messages_t)  # (n, hidden_dim)
        h_i = h.unsqueeze(1).expand(n, n, -1)
        h_j = h.unsqueeze(0).expand(n, n, -1)
        h_concat = torch.cat([h_i, h_j], dim=-1)  # (n, n, 2*hidden_dim)
        scores = (h_concat * self.a_vec).sum(dim=-1)
        scores = torch.where(scores > 0, scores, 0.01 * scores)  # LeakyReLU
        scores = torch.where(adjacency_t > 0, scores, torch.full_like(scores, -1e9))
        return torch.softmax(scores, dim=1)

    def compute_attention(self, messages: np.ndarray, adjacency: np.ndarray) -> np.ndarray:
        """NumPy in/out, no-grad -- the rollout-collection hot path."""
        messages_t = torch.as_tensor(messages, dtype=torch.float32, device=self._device)
        adjacency_t = torch.as_tensor(adjacency, dtype=torch.float32, device=self._device)
        with torch.no_grad():
            return self.compute_attention_t(messages_t, adjacency_t).cpu().numpy()


class GNNMessageDecoder(_FlatParamsMixin, torch.nn.Module):
    """
    Decodes aggregated messages and combines with own observation.

    Enhanced observation: o_i^enhanced = [o_i || aggregate_j(alpha_ij * m_j)]
    """

    def __init__(self, obs_dim: int, msg_dim: int, enhanced_dim: int,
                 config: CommConfig = None, seed: int = 0):
        super().__init__()
        self.config = config or CommConfig()
        self.obs_dim = obs_dim
        self.msg_dim = msg_dim
        self.enhanced_dim = enhanced_dim
        self.hidden_dim = self.config.hidden_dim

        gen = torch.Generator().manual_seed(seed)
        in_dim = obs_dim + msg_dim
        self.fc1 = torch.nn.Linear(in_dim, self.hidden_dim)
        self.fc2 = torch.nn.Linear(self.hidden_dim, enhanced_dim)
        for layer, fan_in in ((self.fc1, in_dim), (self.fc2, self.hidden_dim)):
            torch.nn.init.normal_(layer.weight, mean=0.0, std=(2.0 / fan_in) ** 0.5, generator=gen)
            torch.nn.init.zeros_(layer.bias)

    def decode_t(self, obs_t: torch.Tensor, aggregated_msg_t: torch.Tensor) -> torch.Tensor:
        """Differentiable decode."""
        combined = torch.cat([obs_t, aggregated_msg_t], dim=-1)
        h = torch.relu(self.fc1(combined))
        return torch.tanh(self.fc2(h))

    def decode(self, obs: np.ndarray, aggregated_msg: np.ndarray) -> np.ndarray:
        """NumPy in/out, no-grad -- the rollout-collection hot path."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self._device)
        agg_t = torch.as_tensor(aggregated_msg, dtype=torch.float32, device=self._device)
        with torch.no_grad():
            return self.decode_t(obs_t, agg_t).cpu().numpy()


class CommGraph(torch.nn.Module):
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

    2026-10-22: now a real torch.nn.Module (encoder/attention/decoder are
    registered submodules), trained via backprop + Adam through update()
    instead of the old ES-based GNNCommTrainer (removed -- see module
    docstring history and marl/marl_agents.py's module-level comment).
    forward() keeps its original NumPy-in/NumPy-out, no-grad interface,
    including all its degradation knobs (packet loss, latency,
    quantization, top-k sparsification), for the rollout-collection hot
    path and robustness-evaluation sweeps. forward_t() is the new
    differentiable path (no degradation -- see update()'s docstring for
    why) used for joint training with the downstream policy.
    """

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        enhanced_dim: int,
        adjacency: Optional[np.ndarray] = None,
        config: Optional[CommConfig] = None,
        seed: int = 0,
        device=None,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.config = config or CommConfig()
        self.msg_dim = self.config.message_dim
        self.device = torch.device(device) if device is not None else torch.device("cpu")

        # Build or use provided adjacency matrix
        if adjacency is not None:
            self.adjacency = adjacency
        else:
            self.adjacency = np.eye(n_agents)  # Default: self-loops only
        self._adjacency_t = torch.as_tensor(self.adjacency, dtype=torch.float32, device=self.device)

        # Sub-modules
        self.encoder = GNNMessageEncoder(obs_dim, self.config, seed).to(self.device)
        if self.config.use_attention:
            self.attention = GNNAttention(self.msg_dim, seed=seed+1).to(self.device)
        else:
            self.attention = None
        self.decoder = GNNMessageDecoder(obs_dim, self.msg_dim, enhanced_dim,
                                          self.config, seed+2).to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

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
            # max(..., 1): quantization_bits=1 would otherwise give
            # qmax=0 and divide by zero below. Every current caller only
            # uses 32/16/8/4/2, so this is unreached today, but guards
            # against the stress-sweep range ever being widened to 1 bit.
            qmax = max(2 ** (self.quantization_bits - 1) - 1, 1)
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

    def forward_t(self, observations_t: torch.Tensor) -> torch.Tensor:
        """Differentiable counterpart of forward(): obs_t (n_agents, obs_dim)
        -> enhanced_obs_t (n_agents, enhanced_dim), real autograd graph
        through encoder -> attention -> aggregate -> decode. Deliberately
        does NOT apply forward()'s stochastic degradation knobs (packet
        loss, latency, quantization, top-k sparsification) -- those
        simulate channel imperfections for robustness *evaluation*, not a
        training-time regularizer, and backprop-ing through randomly
        zeroed/delayed/quantized messages would make the gradient signal
        noisier without a clear benefit here. config.dropout (a real
        training regularizer, distinct from the robustness knobs) is not
        applied either, matching forward()'s own "rng is None" no-dropout
        behavior; a future pass could add it back as F.dropout if desired.
        Used only by update() below, called once per rollout -- not by
        _collect_rollout, which still uses forward()."""
        messages_t = self.encoder.encode_t(observations_t)  # (n_agents, msg_dim)
        if self.attention is not None:
            attention_t = self.attention.compute_attention_t(messages_t, self._adjacency_t)
        else:
            row_sums = self._adjacency_t.sum(dim=1, keepdim=True)
            attention_t = self._adjacency_t / torch.clamp(row_sums, min=1e-8)
        aggregated_t = attention_t @ messages_t
        return self.decoder.decode_t(observations_t, aggregated_t)

    def update(self, raw_obs_batch: np.ndarray, act_batch: np.ndarray,
               adv_batch: np.ndarray, old_lp_batch: np.ndarray, actor,
               clip_eps: float = 0.2) -> Dict[str, float]:
        """Real PPO-clip gradient step for the GNN (encoder/attention/
        decoder), via backprop through the SAME downstream objective the
        actor itself is trained on -- replacing the old ES-based
        GNNCommTrainer's advantage-weighted-alignment proxy objective
        (removed; see this module's git history for the ES-era math).

        Mechanics: runs the real (shared, parameter-tied) actor network
        on this module's freshly-recomputed enhanced observations, so
        gradients flow from the actor's own log-prob computation back
        into this module's parameters -- but never calls actor.optimizer
        .step(), so the actor's own weights are only ever updated by its
        own update() calls, not by this one. This makes the GNN's
        training signal the actual policy objective instead of a
        hand-designed proxy.

        raw_obs_batch: (T, n_agents, obs_dim). act_batch/old_lp_batch:
        (T, n_agents, act_dim)/(T, n_agents). adv_batch: (T, n_agents).
        actor: the SharedGaussianActor these enhanced observations feed.
        """
        obs_t = torch.as_tensor(raw_obs_batch, dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(act_batch, dtype=torch.float32, device=self.device)
        adv_t = torch.as_tensor(adv_batch, dtype=torch.float32, device=self.device)
        old_lp_t = torch.as_tensor(old_lp_batch, dtype=torch.float32, device=self.device)
        T = obs_t.shape[0]

        self.optimizer.zero_grad()
        # forward_t operates on one timestep's (n_agents, obs_dim) graph at
        # a time (attention is a per-timestep graph op across agents); stack
        # over T into one batched call via vmap-free manual looping would be
        # O(T) Python overhead, so instead flatten T into the sequence the
        # encoder/decoder (plain per-row MLPs) can already batch over, and
        # only loop T for the O(n_agents^2) attention step itself.
        enhanced_list = [self.forward_t(obs_t[t]) for t in range(T)]
        enhanced_t = torch.stack(enhanced_list, dim=0)  # (T, n_agents, enhanced_dim)

        # Shared actor is parameter-shared across agents -- one batched
        # (T*n_agents, enhanced_dim) forward call evaluates every agent at
        # once instead of a Python loop over agents.
        flat_enhanced = enhanced_t.reshape(T * self.n_agents, -1)
        flat_act = act_t.reshape(T * self.n_agents, -1)
        new_lp = actor._log_prob_t(flat_enhanced, flat_act).reshape(T, self.n_agents)

        ratio = torch.exp(torch.clamp(new_lp - old_lp_t, -5, 5))
        clip_r = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
        loss = -torch.minimum(ratio * adv_t, clip_r * adv_t).mean()
        loss.backward()
        self.optimizer.step()

        return {"loss": float(loss.item()), "n_params": sum(p.numel() for p in self.parameters())}

    def get_all_params(self) -> np.ndarray:
        """Flat parameter vector -- kept for backward compatibility with
        ES-era tests/fixtures, see SharedGaussianActor.get_all_params."""
        with torch.no_grad():
            return torch.nn.utils.parameters_to_vector(self.parameters()).cpu().numpy()

    def set_all_params(self, params: np.ndarray):
        p_t = torch.as_tensor(np.asarray(params), dtype=torch.float32, device=self.device)
        torch.nn.utils.vector_to_parameters(p_t, self.parameters())

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

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


# GNNCommTrainer (the old ES-based trainer for this module's GNN) was
# removed 2026-10-22 -- its functionality is now CommGraph.update() above,
# a real PPO-clip backprop step through the actual downstream policy
# objective instead of an ES-estimated advantage-weighted-alignment proxy.
# See git history for the removed class if the old math is ever needed for
# reference. The dormant marl/comm_graph_gradient.py fork (its own,
# separate GNNCommTrainer) was retired the same day, now fully superseded
# by this module -- see git history for that file too if ever needed.
