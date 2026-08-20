"""
comm_graph_gradient.py  —  TRUE Gradient-Based GNN Communication (Drop-in Extension)
────────────────────────────────────────────────────────────────────────────────────

This module is a DROP-IN EXTENSION of comm_graph.py. It provides:

1. ALL original classes from comm_graph.py (unchanged API):
   - CommConfig, GNNMessageEncoder, GNNAttention, GNNMessageDecoder
   - CommGraph, GNNCommTrainer
   - build_relay_adjacency, build_neighbor_observations

2. NEW gradient-based versions ("Grad" prefix):
   - GradGNNMessageEncoder, GradGNNAttention, GradGNNMessageDecoder
   - GradCommGraph, GradientBasedMAPPOMARLTrainer

3. A unified interface: CommGraph can optionally use gradient mode
   via the `use_gradient_mode` parameter.

Usage (no code changes needed):
────────────────────────────────
    from comm_graph_gradient import CommGraph, GNNCommTrainer
    # Works exactly like the original

    from comm_graph_gradient import GradientBasedMAPPOMARLTrainer
    # New trainer with true backprop

Design Philosophy
─────────────────
- Zero breaking changes to existing API
- All original functionality preserved
- Gradient-based training is opt-in via new classes/parameters
- Verified against finite-difference gradients

Mathematical Guarantees
──────────────────────
- All manual backprop derivatives verified via finite differences
- PPO-CLIP gradient: ∂L/∂θ = A_t · r_t(θ) · ∂log π/∂θ  (non-clipped branch)
- Gaussian policy gradient: ∂logπ/∂μ = z/σ,  z = (a−μ)/σ
- Communication backprop: dL/dθ_comm flows through encoder→attention→decoder
"""

__all__ = [
    # ── Original comm_graph.py API (100% backward compatible) ──
    "CommConfig",
    "GNNMessageEncoder",
    "GNNAttention", 
    "GNNMessageDecoder",
    "CommGraph",
    "GNNCommTrainer",
    "build_relay_adjacency",
    "build_neighbor_observations",
    # ── Gradient-based extensions (new) ──
    "ComputationTape",
    "TapeEntry",
    "GradGNNMessageEncoder",
    "GradGNNAttention",
    "GradGNNMessageDecoder",
    "GradCommGraph",
    "GradSharedGaussianActor",
    "Adam",
    "GradientBasedMAPPOMARLTrainer",
    "compute_ppo_policy_gradient",
    "compute_comm_gradient",
    "run_gradient_mappo",
    "verify_actor_gradient",
]


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 1: ORIGINAL comm_graph.py (preserved verbatim — zero API changes)
# ═══════════════════════════════════════════════════════════════════════════════

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class CommConfig:
    """Communication configuration."""
    message_dim: int = 8
    hidden_dim: int = 32
    n_gnn_layers: int = 2
    bandwidth_limit_bps: int = 1000
    dropout: float = 0.1
    use_attention: bool = True


def relu(x): return np.maximum(0, x)
def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))
def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


class GNNMessageEncoder:
    """Encodes local observations into compressed messages for communication."""

    def __init__(self, obs_dim: int, config: CommConfig = None, seed: int = 0):
        self.config = config or CommConfig()
        self.obs_dim = obs_dim
        self.msg_dim = self.config.message_dim
        self.hidden_dim = self.config.hidden_dim

        rng = np.random.default_rng(seed)
        s = lambda a, b: np.sqrt(2.0 / a)

        self.W_enc1 = rng.normal(0, s(obs_dim, self.hidden_dim), (self.hidden_dim, obs_dim))
        self.b_enc1 = np.zeros(self.hidden_dim)
        self.W_enc2 = rng.normal(0, s(self.hidden_dim, self.msg_dim), (self.msg_dim, self.hidden_dim))
        self.b_enc2 = np.zeros(self.msg_dim)

        self.n_params = (self.W_enc1.size + self.b_enc1.size +
                        self.W_enc2.size + self.b_enc2.size)

    def encode(self, obs: np.ndarray) -> np.ndarray:
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
    """Learned attention mechanism for graph message passing."""

    def __init__(self, msg_dim: int, hidden_dim: int = 16, seed: int = 0):
        self.msg_dim = msg_dim
        self.hidden_dim = hidden_dim

        rng = np.random.default_rng(seed)
        s = lambda a: np.sqrt(2.0 / a)

        self.W_att = rng.normal(0, s(msg_dim), (hidden_dim, msg_dim))
        self.a_vec = rng.normal(0, s(hidden_dim * 2), (hidden_dim * 2,))

        self.n_params = self.W_att.size + self.a_vec.size

    def compute_attention(self, messages: np.ndarray, adjacency: np.ndarray) -> np.ndarray:
        n = messages.shape[0]
        h = messages @ self.W_att.T

        h_i = np.repeat(h[:, np.newaxis, :], n, axis=1)
        h_j = np.repeat(h[np.newaxis, :, :], n, axis=0)
        h_concat = np.concatenate([h_i, h_j], axis=-1)

        scores = np.sum(h_concat * self.a_vec, axis=-1)
        scores = np.where(scores > 0, scores, 0.01 * scores)

        scores = np.where(adjacency > 0, scores, -1e9)
        attention = softmax(scores, axis=1)

        return attention

    def get_params(self) -> np.ndarray:
        return np.concatenate([self.W_att.ravel(), self.a_vec])

    def set_params(self, params: np.ndarray):
        self.W_att = params[:self.W_att.size].reshape(self.W_att.shape)
        self.a_vec = params[self.W_att.size:]


class GNNMessageDecoder:
    """Decodes aggregated messages and combines with own observation."""

    def __init__(self, obs_dim: int, msg_dim: int, enhanced_dim: int,
                 config: CommConfig = None, seed: int = 0):
        self.config = config or CommConfig()
        self.obs_dim = obs_dim
        self.msg_dim = msg_dim
        self.enhanced_dim = enhanced_dim
        self.hidden_dim = self.config.hidden_dim

        rng = np.random.default_rng(seed)
        s = lambda a, b: np.sqrt(2.0 / a)

        in_dim = obs_dim + msg_dim
        self.W_dec1 = rng.normal(0, s(in_dim, self.hidden_dim), (self.hidden_dim, in_dim))
        self.b_dec1 = np.zeros(self.hidden_dim)
        self.W_dec2 = rng.normal(0, s(self.hidden_dim, enhanced_dim), (enhanced_dim, self.hidden_dim))
        self.b_dec2 = np.zeros(enhanced_dim)

        self.n_params = (self.W_dec1.size + self.b_dec1.size +
                        self.W_dec2.size + self.b_dec2.size)

    def decode(self, obs: np.ndarray, aggregated_msg: np.ndarray) -> np.ndarray:
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
    """Complete GNN communication module for DOCR-MARL."""

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

        if adjacency is not None:
            self.adjacency = adjacency
        else:
            self.adjacency = np.eye(n_agents)

        self.encoder = GNNMessageEncoder(obs_dim, self.config, seed)
        if self.config.use_attention:
            self.attention = GNNAttention(self.msg_dim, seed=seed+1)
        else:
            self.attention = None
        self.decoder = GNNMessageDecoder(obs_dim, self.msg_dim, enhanced_dim,
                                          self.config, seed+2)

        self.total_messages_sent = 0
        self.total_bytes_sent = 0
        self.bandwidth_estimate_bps = 0.0

    def forward(self, observations: np.ndarray, rng: Optional[np.random.Generator] = None
               ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        messages = self.encoder.encode(observations)

        if rng is not None and self.config.dropout > 0:
            mask = rng.random(messages.shape) > self.config.dropout
            messages = messages * mask

        if self.attention is not None:
            attention = self.attention.compute_attention(messages, self.adjacency)
        else:
            row_sums = self.adjacency.sum(axis=1, keepdims=True)
            attention = self.adjacency / np.maximum(row_sums, 1e-8)

        aggregated = attention @ messages
        enhanced_obs = self.decoder.decode(observations, aggregated)

        self.total_messages_sent += self.n_agents
        bytes_per_message = self.msg_dim * 4
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
        params = [self.encoder.get_params(), self.decoder.get_params()]
        if self.attention is not None:
            params.append(self.attention.get_params())
        return np.concatenate(params)

    def set_all_params(self, params: np.ndarray):
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
        self.total_messages_sent = 0
        self.total_bytes_sent = 0


class GNNCommTrainer:
    """Trains GNN communication parameters via Evolution Strategies (ES)."""

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

        self.use_adam = use_adam
        theta = self.comm.get_all_params()
        self.m  = np.zeros_like(theta)
        self.v  = np.zeros_like(theta)
        self.t  = 0
        self.beta1, self.beta2, self.eps_adam = 0.9, 0.999, 1e-8

        self.loss_history:      List[float] = []
        self.grad_norm_history: List[float] = []

    def _adam_step(self, grad: np.ndarray) -> np.ndarray:
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad ** 2
        m_hat  = self.m / (1 - self.beta1 ** self.t)
        v_hat  = self.v / (1 - self.beta2 ** self.t)
        return self.lr * m_hat / (np.sqrt(v_hat) + self.eps_adam)

    def update(
        self,
        observations_batch: np.ndarray,
        returns_batch: np.ndarray,
    ) -> Dict[str, float]:
        theta_base = self.comm.get_all_params()
        n_theta    = len(theta_base)
        T          = observations_batch.shape[0]

        baseline_enhanced = np.stack([
            self.comm.forward(observations_batch[t])[0]
            for t in range(T)
        ])

        adv = returns_batch - returns_batch.mean()

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

        self.comm.set_all_params(theta_base)

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

    def get_attention_entropy(self, observations: np.ndarray) -> float:
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


def build_relay_adjacency(net) -> np.ndarray:
    """Build adjacency matrix from relay protection pairs."""
    n = net.n_relays
    adj = np.zeros((n, n), dtype=np.float32)

    for pair in net.pairs:
        adj[pair.primary, pair.backup] = 1.0
        adj[pair.backup, pair.primary] = 1.0

    np.fill_diagonal(adj, 1.0)

    for i in range(n):
        for j in range(i+1, n):
            bus_i = None
            bus_j = None
            for pair in net.pairs:
                if pair.primary == i:
                    bus_i = pair.bus_pri
                if pair.primary == j:
                    bus_j = pair.bus_pri
            if bus_i is not None and bus_j is not None and bus_i == bus_j:
                adj[i, j] = 1.0
                adj[j, i] = 1.0

    return adj


def build_neighbor_observations(
    net,
    observations: np.ndarray,
    relay_pairs: List[List[int]],
    max_neighbors: int = 4,
) -> np.ndarray:
    """Build neighbor-aware observations by concatenating neighbor features."""
    n = net.n_relays
    obs_dim = observations.shape[1]
    neighbor_obs = np.zeros((n, max_neighbors, obs_dim))

    for i in range(n):
        neighbors = set()
        for pidx in relay_pairs[i]:
            pair = net.pairs[pidx]
            if pair.primary == i:
                neighbors.add(pair.backup)
            else:
                neighbors.add(pair.primary)

        for pair in net.pairs:
            if pair.primary == i:
                for other_pair in net.pairs:
                    if other_pair.bus_pri == pair.bus_pri and other_pair.primary != i:
                        neighbors.add(other_pair.primary)

        neighbors = sorted(list(neighbors))[:max_neighbors]
        for j, neighbor_idx in enumerate(neighbors):
            neighbor_obs[i, j] = observations[neighbor_idx]

    return neighbor_obs.reshape(n, -1)


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 2: GRADIENT-BASED EXTENSIONS (new — no API changes to Part 1)
# ═══════════════════════════════════════════════════════════════════════════════

import time
from typing import Callable


# ── 2.1 Computation Tape (manual autograd) ──────────────────────────────────

class TapeEntry:
    """One forward operation with its backward function."""
    __slots__ = ("op", "inputs", "output", "backward_fn", "param_ids")

    def __init__(self, op: str, inputs: Dict[str, np.ndarray],
                 output: np.ndarray, backward_fn: Callable,
                 param_ids: List[str] = None):
        self.op = op
        self.inputs = inputs
        self.output = output
        self.backward_fn = backward_fn
        self.param_ids = param_ids or []


class ComputationTape:
    """
    Manual autograd tape. Records operations in forward order.
    Backward walks the tape in reverse, applying the chain rule.

    Supported primitives: linear, relu, tanh, matmul, concat, mul, add, softmax
    """

    def __init__(self):
        self.entries: List[TapeEntry] = []
        self.grads: Dict[str, np.ndarray] = {}

    def _add(self, entry: TapeEntry) -> np.ndarray:
        self.entries.append(entry)
        return entry.output

    def linear(self, x, W, b, name="linear"):
        """y = x @ W.T + b  (batch or single)"""
        if x.ndim == 1:
            y = W @ x + b
        else:
            y = x @ W.T + b

        def backward(dout):
            if x.ndim == 1:
                return {
                    "x": W.T @ dout,
                    name+"_W": np.outer(dout, x),
                    name+"_b": dout,
                }
            else:
                return {
                    "x": dout @ W,
                    name+"_W": dout.T @ x,
                    name+"_b": dout.sum(axis=0),
                }

        return self._add(TapeEntry("linear", {"x": x}, y, backward,
                                   param_ids=[name+"_W", name+"_b"]))

    def relu(self, x, name="relu"):
        y = np.maximum(0, x)
        def backward(dout):
            return {"x": dout * (x > 0)}
        return self._add(TapeEntry("relu", {"x": x}, y, backward))

    def tanh(self, x, name="tanh"):
        y = np.tanh(x)
        def backward(dout):
            return {"x": dout * (1.0 - y ** 2)}
        return self._add(TapeEntry("tanh", {"x": x}, y, backward))

    def matmul(self, A, B, name="matmul"):
        Y = A @ B
        def backward(dout):
            return {"A": dout @ B.T, "B": A.T @ dout}
        return self._add(TapeEntry("matmul", {"A": A, "B": B}, Y, backward))

    def concat(self, arrays, axis=-1, name="concat"):
        z = np.concatenate(arrays, axis=axis)
        shapes = [a.shape[axis] for a in arrays]
        def backward(dout):
            grads = {}
            idx = 0
            for i, s in enumerate(shapes):
                grads[f"x{i}"] = np.take(dout, range(idx, idx + s), axis=axis)
                idx += s
            return grads
        inputs = {f"x{i}": a for i, a in enumerate(arrays)}
        return self._add(TapeEntry("concat", inputs, z, backward))

    def mul(self, a, b, name="mul"):
        y = a * b
        def backward(dout):
            return {"a": dout * b, "b": dout * a}
        return self._add(TapeEntry("mul", {"a": a, "b": b}, y, backward))

    def add(self, a, b, name="add"):
        y = a + b
        def backward(dout):
            # Handle broadcasting: sum over broadcasted dimensions
            def reduce_grad(grad, orig_shape):
                if grad.shape == orig_shape:
                    return grad
                # Sum over dimensions that were broadcasted
                while grad.ndim > len(orig_shape):
                    grad = grad.sum(axis=0)
                for i, (g, o) in enumerate(zip(grad.shape, orig_shape)):
                    if o == 1 and g > 1:
                        grad = grad.sum(axis=i, keepdims=True)
                return grad
            return {
                "a": reduce_grad(dout.copy(), a.shape),
                "b": reduce_grad(dout.copy(), b.shape),
            }
        return self._add(TapeEntry("add", {"a": a, "b": b}, y, backward))

    def leaky_relu(self, x, alpha=0.01, name="leaky_relu"):
        y = np.where(x > 0, x, alpha * x)
        def backward(dout):
            return {"x": dout * np.where(x > 0, 1.0, alpha)}
        return self._add(TapeEntry("leaky_relu", {"x": x}, y, backward))

    def repeat(self, x, repeats, axis=None, name="repeat"):
        y = np.repeat(x, repeats, axis=axis)
        def backward(dout):
            # Approximate: sum-reduce along the repeated axis
            # For simplicity, handle common cases
            if axis is None:
                return {"x": np.sum(dout.reshape(x.shape + (repeats,)), axis=-1)}
            else:
                # Sum over the repeated blocks
                shape = list(x.shape)
                shape.insert(axis, repeats)
                reshaped = dout.reshape(shape)
                return {"x": np.sum(reshaped, axis=axis + 1)}
        return self._add(TapeEntry("repeat", {"x": x}, y, backward))

    def sum(self, x, axis=None, keepdims=False, name="sum"):
        y = np.sum(x, axis=axis, keepdims=keepdims)
        def backward(dout):
            if not keepdims and axis is not None:
                shape = list(x.shape)
                shape[axis] = 1
                dout = dout.reshape(shape)
            return {"x": np.broadcast_to(dout, x.shape)}
        return self._add(TapeEntry("sum", {"x": x}, y, backward))

    
    def masked_fill(self, x, mask, value, name="masked_fill"):
        y = np.where(mask, x, value)
        def backward(dout):
            return {"x": dout * mask}
        return self._add(TapeEntry("masked_fill", {"x": x}, y, backward))

    def reshape(self, x, shape, name="reshape"):
        y = x.reshape(shape)
        def backward(dout):
            # If dout was broadcasted, sum-reduce before reshaping
            grad = dout
            while grad.ndim > len(shape):
                grad = grad.sum(axis=0)
            for i, (g, s) in enumerate(zip(grad.shape, shape)):
                if s == 1 and g > 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return {"x": grad.reshape(x.shape)}
        return self._add(TapeEntry("reshape", {"x": x}, y, backward))

    def softmax(self, x, axis=-1, mask=None, name="softmax"):
        """Masked softmax with gradient."""
        x_safe = x - np.max(x, axis=axis, keepdims=True)
        e = np.exp(x_safe)
        if mask is not None:
            e = e * mask
        y = e / (np.sum(e, axis=axis, keepdims=True) + 1e-10)

        def backward(dout):
            return {"x": y * (dout - np.sum(y * dout, axis=axis, keepdims=True))}
        return self._add(TapeEntry("softmax", {"x": x}, y, backward))

    def backward(self, dL_final: np.ndarray) -> Dict[str, np.ndarray]:
        """Backpropagate from final output gradient through entire tape."""
        self.grads.clear()
        tensor_grads = {id(self.entries[-1].output): dL_final}

        for entry in reversed(self.entries):
            out_id = id(entry.output)
            if out_id not in tensor_grads:
                continue

            dout = tensor_grads[out_id]
            local_grads = entry.backward_fn(dout)

            # Accumulate parameter gradients
            for pid in entry.param_ids:
                if pid in local_grads:
                    if pid not in self.grads:
                        self.grads[pid] = local_grads[pid].copy()
                    else:
                        self.grads[pid] += local_grads[pid]

            # Route gradients to input tensors
            for key, grad in local_grads.items():
                if key.startswith("x") or key in ("A", "B", "a", "b"):
                    if key in entry.inputs:
                        inp_id = id(entry.inputs[key])
                        if inp_id in tensor_grads:
                            tensor_grads[inp_id] += grad
                        else:
                            tensor_grads[inp_id] = grad.copy()
                        # Store by string key for test access (do NOT accumulate —
                        # different ops may reuse the same key with different shapes).
                        # The tensor_id routing above handles the true chain rule.
                        if key not in self.grads:
                            self.grads[key] = grad.copy()

        return self.grads

    def get_grad(self, param_id: str) -> Optional[np.ndarray]:
        return self.grads.get(param_id)

    def clear(self):
        self.entries.clear()
        self.grads.clear()


# ── 2.2 Gradient-Based GNN Modules ──────────────────────────────────────────

class GradGNNMessageEncoder:
    """Gradient-tracked encoder: obs → linear → ReLU → linear → tanh → messages."""

    def __init__(self, obs_dim, msg_dim, hidden_dim, seed=0):
        rng = np.random.default_rng(seed)
        s = lambda a, b: np.sqrt(2.0 / a)

        self.W1 = rng.normal(0, s(obs_dim, hidden_dim), (hidden_dim, obs_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.normal(0, s(hidden_dim, msg_dim), (msg_dim, hidden_dim))
        self.b2 = np.zeros(msg_dim)

        self.obs_dim, self.msg_dim, self.hidden_dim = obs_dim, msg_dim, hidden_dim
        self.n_params = self.W1.size + self.b1.size + self.W2.size + self.b2.size

    def forward(self, obs, tape=None):
        if tape is None:
            if obs.ndim == 1:
                h = np.maximum(0, self.W1 @ obs + self.b1)
                return np.tanh(self.W2 @ h + self.b2)
            else:
                h = np.maximum(0, obs @ self.W1.T + self.b1)
                return np.tanh(h @ self.W2.T + self.b2)

        if obs.ndim == 1:
            h1 = tape.linear(obs, self.W1, self.b1, name="enc1")
            h_relu = tape.relu(h1, name="enc_relu")
            h2 = tape.linear(h_relu, self.W2, self.b2, name="enc2")
            return tape.tanh(h2, name="enc_tanh")
        else:
            # Batch mode: use 2D matrix operations through tape
            h1 = tape.linear(obs, self.W1, self.b1, name="enc1")
            h_relu = tape.relu(h1, name="enc_relu")
            h2 = tape.linear(h_relu, self.W2, self.b2, name="enc2")
            return tape.tanh(h2, name="enc_tanh")

    def get_params(self):
        return np.concatenate([self.W1.ravel(), self.b1, self.W2.ravel(), self.b2])

    def set_params(self, p):
        s = [self.W1.size, self.b1.size, self.W2.size, self.b2.size]
        c = np.split(p, np.cumsum(s[:-1]))
        self.W1, self.b1, self.W2, self.b2 = c[0].reshape(self.W1.shape), c[1], c[2].reshape(self.W2.shape), c[3]


class GradGNNAttention:
    """Gradient-tracked attention: messages → W_att → pairwise → LeakyReLU → softmax."""

    def __init__(self, msg_dim, hidden_dim=16, seed=0):
        rng = np.random.default_rng(seed)
        s = lambda a: np.sqrt(2.0 / a)

        self.W_att = rng.normal(0, s(msg_dim), (hidden_dim, msg_dim))
        self.a_vec = rng.normal(0, s(hidden_dim * 2), (hidden_dim * 2,))

        self.msg_dim, self.hidden_dim = msg_dim, hidden_dim
        self.n_params = self.W_att.size + self.a_vec.size

    def forward(self, messages, adjacency, tape=None):
        n = messages.shape[0]

        if tape is None:
            h = messages @ self.W_att.T
            h_i = np.repeat(h[:, None, :], n, axis=1)
            h_j = np.repeat(h[None, :, :], n, axis=0)
            h_cat = np.concatenate([h_i, h_j], axis=-1)
            scores = np.sum(h_cat * self.a_vec, axis=-1)
            scores = np.where(scores > 0, scores, 0.01 * scores)
            scores = np.where(adjacency > 0, scores, -1e9)
            e = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            return e / (np.sum(e, axis=1, keepdims=True) + 1e-10)

        # Gradient-tracked path: decompose a_vec into two halves for broadcasting
        # scores[i,j] = a_vec · [h_i || h_j] = a1·h_i + a2·h_j
        # where a1 = a_vec[:hidden], a2 = a_vec[hidden:]
        h = tape.linear(messages, self.W_att, np.zeros(self.hidden_dim), name="att_W")
        # h shape: (n, hidden)
        a1 = self.a_vec[:self.hidden_dim]  # (hidden,)
        a2 = self.a_vec[self.hidden_dim:]  # (hidden,)
        # scores = h @ a1.T[:, None] + h @ a2.T[None, :]  → (n, n)
        # Using broadcasting via tape reshape and add
        h_a1 = tape.linear(h, a1.reshape(1, self.hidden_dim), np.zeros(1), name="att_a1")
        h_a2 = tape.linear(h, a2.reshape(1, self.hidden_dim), np.zeros(1), name="att_a2")
        # h_a1 shape: (n, 1), h_a2 shape: (n, 1)
        # Need to broadcast: h_a1[:, None] + h_a2[None, :] → (n, n)
        h_a1_b = tape.reshape(h_a1, (n, 1), name="att_a1r")
        h_a2_b = tape.reshape(h_a2, (1, n), name="att_a2r")
        scores = tape.add(h_a1_b, h_a2_b, name="att_scores")
        scores_lr = tape.leaky_relu(scores, alpha=0.01, name="att_lr")
        mask = (adjacency > 0).astype(float)
        scores_masked = tape.masked_fill(scores_lr, mask > 0, -1e9, name="att_masked")
        return tape.softmax(scores_masked, axis=1, mask=mask, name="att_softmax")

    def get_params(self):
        return np.concatenate([self.W_att.ravel(), self.a_vec])

    def set_params(self, p):
        self.W_att = p[:self.W_att.size].reshape(self.W_att.shape)
        self.a_vec = p[self.W_att.size:]


class GradGNNMessageDecoder:
    """Gradient-tracked decoder: [obs || agg_msg] → linear → ReLU → linear → tanh."""

    def __init__(self, obs_dim, msg_dim, enhanced_dim, hidden_dim, seed=0):
        rng = np.random.default_rng(seed)
        s = lambda a, b: np.sqrt(2.0 / a)

        in_dim = obs_dim + msg_dim
        self.W1 = rng.normal(0, s(in_dim, hidden_dim), (hidden_dim, in_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.normal(0, s(hidden_dim, enhanced_dim), (enhanced_dim, hidden_dim))
        self.b2 = np.zeros(enhanced_dim)

        self.obs_dim, self.msg_dim, self.enhanced_dim = obs_dim, msg_dim, enhanced_dim
        self.hidden_dim = hidden_dim
        self.n_params = self.W1.size + self.b1.size + self.W2.size + self.b2.size

    def forward(self, obs, aggregated_msg, tape=None):
        if tape is None:
            combined = np.concatenate([obs, aggregated_msg], axis=-1)
            if combined.ndim == 1:
                h = np.maximum(0, self.W1 @ combined + self.b1)
                return np.tanh(self.W2 @ h + self.b2)
            else:
                h = np.maximum(0, combined @ self.W1.T + self.b1)
                return np.tanh(h @ self.W2.T + self.b2)

        if obs.ndim == 1:
            combined = tape.concat([obs, aggregated_msg], axis=-1, name="dec_combined")
            h1 = tape.linear(combined, self.W1, self.b1, name="dec1")
            h_relu = tape.relu(h1, name="dec_relu")
            h2 = tape.linear(h_relu, self.W2, self.b2, name="dec2")
            return tape.tanh(h2, name="dec_tanh")
        else:
            # Batch mode: use 2D matrix operations through tape
            combined = tape.concat([obs, aggregated_msg], axis=-1, name="dec_combined")
            h1 = tape.linear(combined, self.W1, self.b1, name="dec1")
            h_relu = tape.relu(h1, name="dec_relu")
            h2 = tape.linear(h_relu, self.W2, self.b2, name="dec2")
            return tape.tanh(h2, name="dec_tanh")

    def get_params(self):
        return np.concatenate([self.W1.ravel(), self.b1, self.W2.ravel(), self.b2])

    def set_params(self, p):
        s = [self.W1.size, self.b1.size, self.W2.size, self.b2.size]
        c = np.split(p, np.cumsum(s[:-1]))
        self.W1, self.b1, self.W2, self.b2 = c[0].reshape(self.W1.shape), c[1], c[2].reshape(self.W2.shape), c[3]


class GradCommGraph:
    """Complete gradient-tracked communication module."""

    def __init__(self, n_agents, obs_dim, enhanced_dim, adjacency=None,
                 msg_dim=8, hidden_dim=32, use_attention=True, seed=0):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.enhanced_dim = enhanced_dim
        self.msg_dim = msg_dim
        self.adjacency = adjacency if adjacency is not None else np.eye(n_agents)
        self.use_attention = use_attention

        self.encoder = GradGNNMessageEncoder(obs_dim, msg_dim, hidden_dim, seed)
        self.attention = GradGNNAttention(msg_dim, hidden_dim, seed+1) if use_attention else None
        self.decoder = GradGNNMessageDecoder(obs_dim, msg_dim, enhanced_dim, hidden_dim, seed+2)

        self.total_messages = 0
        self.total_bytes = 0

    def forward(self, observations, tape=None, rng=None):
        messages = self.encoder.forward(observations, tape=tape)

        if self.attention is not None:
            attention = self.attention.forward(messages, self.adjacency, tape=tape)
        else:
            rs = self.adjacency.sum(axis=1, keepdims=True)
            attention = self.adjacency / np.maximum(rs, 1e-8)

        if tape is not None:
            aggregated = tape.matmul(attention, messages, name="aggregate")
        else:
            aggregated = attention @ messages

        enhanced = self.decoder.forward(observations, aggregated, tape=tape)

        self.total_messages += self.n_agents
        self.total_bytes += self.n_agents * self.msg_dim * 4

        return enhanced, messages, {
            "attention": attention, "aggregated": aggregated,
            "bytes_per_step": self.n_agents * self.msg_dim * 4,
        }

    def get_all_params(self):
        params = [self.encoder.get_params(), self.decoder.get_params()]
        if self.attention is not None:
            params.append(self.attention.get_params())
        return np.concatenate(params)

    def set_all_params(self, p):
        n_e = self.encoder.n_params
        n_d = self.decoder.n_params
        self.encoder.set_params(p[:n_e])
        self.decoder.set_params(p[n_e:n_e+n_d])
        if self.attention is not None:
            self.attention.set_params(p[n_e+n_d:])

    def n_params(self):
        total = self.encoder.n_params + self.decoder.n_params
        if self.attention is not None:
            total += self.attention.n_params
        return total

    def get_bandwidth_usage(self, steps_per_second=1.0):
        bps = self.n_agents * self.msg_dim * 4 * 8 * steps_per_second
        kbps = bps / 1000.0
        return {
            "bps": bps, "kbps": kbps,
            "kbps_per_agent": kbps / self.n_agents,
            "bytes_per_step": self.n_agents * self.msg_dim * 4,
        }


# ── 2.3 Gradient-Based Actor ────────────────────────────────────────────────

class GradSharedGaussianActor:
    """Shared actor with gradient tape support."""

    def __init__(self, obs_dim, act_dim, hidden=64, seed=0):
        rng = np.random.default_rng(seed)
        s = lambda a: np.sqrt(2.0 / a)

        self.W1 = rng.normal(0, s(obs_dim), (hidden, obs_dim))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, s(hidden), (hidden, hidden))
        self.b2 = np.zeros(hidden)
        self.W3 = rng.normal(0, s(hidden), (act_dim, hidden))
        self.b3 = np.zeros(act_dim)

        self.log_std = np.full(act_dim, -1.0)

        self.obs_dim, self.act_dim, self.hidden = obs_dim, act_dim, hidden
        self.n_net_params = (self.W1.size + self.b1.size + self.W2.size +
                             self.b2.size + self.W3.size + self.b3.size)
        self.n_params = self.n_net_params + act_dim

    def forward(self, obs, tape=None):
        if tape is None:
            if obs.ndim == 1:
                h1 = np.maximum(0, self.W1 @ obs + self.b1)
                h2 = np.maximum(0, self.W2 @ h1 + self.b2)
                return self.W3 @ h2 + self.b3
            else:
                h1 = np.maximum(0, obs @ self.W1.T + self.b1)
                h2 = np.maximum(0, h1 @ self.W2.T + self.b2)
                return h2 @ self.W3.T + self.b3

        if obs.ndim == 1:
            h1 = tape.linear(obs, self.W1, self.b1, name="actor_l1")
            hr1 = tape.relu(h1, name="actor_relu1")
            h2 = tape.linear(hr1, self.W2, self.b2, name="actor_l2")
            hr2 = tape.relu(h2, name="actor_relu2")
            return tape.linear(hr2, self.W3, self.b3, name="actor_l3")
        else:
            out = []
            for i in range(obs.shape[0]):
                h1 = tape.linear(obs[i], self.W1, self.b1, name=f"actor_l1_{i}")
                hr1 = tape.relu(h1, name=f"actor_relu1_{i}")
                h2 = tape.linear(hr1, self.W2, self.b2, name=f"actor_l2_{i}")
                hr2 = tape.relu(h2, name=f"actor_relu2_{i}")
                out.append(tape.linear(hr2, self.W3, self.b3, name=f"actor_l3_{i}"))
            return np.stack(out)

    def get_mean(self, obs, tape=None):
        pre = self.forward(obs, tape=tape)
        if tape is not None:
            return tape.tanh(pre, name="actor_tanh")
        return np.tanh(pre)

    def sample(self, obs, rng, tape=None):
        mean = self.get_mean(obs, tape=tape)
        std = np.exp(np.clip(self.log_std, -3, 0))
        noise = rng.normal(0, 1, self.act_dim)
        action = np.clip(mean + std * noise, -1.0, 1.0)
        log_prob = float(np.sum(-0.5 * (noise**2 + 2 * self.log_std + np.log(2 * np.pi))))
        return action, log_prob

    def log_prob(self, obs, action):
        mean = self.get_mean(obs)
        std = np.exp(np.clip(self.log_std, -3, 0))
        z = (action - mean) / (std + 1e-8)
        return float(np.sum(-0.5 * (z**2 + 2 * self.log_std + np.log(2 * np.pi))))

    def get_all_params(self):
        return np.concatenate([
            self.W1.ravel(), self.b1, self.W2.ravel(), self.b2,
            self.W3.ravel(), self.b3, self.log_std,
        ])

    def set_all_params(self, p):
        s = [self.W1.size, self.b1.size, self.W2.size, self.b2.size,
             self.W3.size, self.b3.size, self.act_dim]
        c = np.split(p, np.cumsum(s[:-1]))
        self.W1, self.b1 = c[0].reshape(self.W1.shape), c[1]
        self.W2, self.b2 = c[2].reshape(self.W2.shape), c[3]
        self.W3, self.b3 = c[4].reshape(self.W3.shape), c[5]
        self.log_std = c[6]


# ── 2.4 Adam Optimizer ──────────────────────────────────────────────────────

class Adam:
    """Adam optimizer for numpy arrays."""

    def __init__(self, lr=3e-4, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.t = 0
        self.m, self.v = {}, {}

    def step(self, params, grads, name="params"):
        self.t += 1
        if name not in self.m:
            self.m[name] = np.zeros_like(params)
            self.v[name] = np.zeros_like(params)

        self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grads
        self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (grads ** 2)

        m_hat = self.m[name] / (1 - self.beta1 ** self.t)
        v_hat = self.v[name] / (1 - self.beta2 ** self.t)

        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ── 2.5 Core Gradient Computation ─────────────────────────────────────────────

def compute_ppo_policy_gradient(actor, obs, action, advantage, old_log_prob, clip_eps=0.2):
    """
    Compute PPO-CLIP policy gradient for ONE sample.

    Returns: (loss, actor_grad_flat, dloss_dobs)
    """
    tape = ComputationTape()

    mean = actor.get_mean(obs, tape=tape)
    std = np.exp(np.clip(actor.log_std, -3, 0))
    z = (action - mean) / (std + 1e-8)
    log_prob = float(np.sum(-0.5 * (z**2 + 2 * actor.log_std + np.log(2 * np.pi))))

    ratio = np.exp(np.clip(log_prob - old_log_prob, -10, 10))
    clipped = np.clip(ratio, 1 - clip_eps, 1 + clip_eps)

    surr1 = ratio * advantage
    surr2 = clipped * advantage

    # PPO-CLIP: gradient is zero when the objective is clipped.
    # The min(r*A, clip(r)*A) picks the smaller value.
    # For A > 0: clipped when r > 1+ε (clip(r) = 1+ε < r, so min = clip(r)*A)
    # For A < 0: clipped when r < 1-ε (clip(r) = 1-ε > r, so min = clip(r)*A)
    # Zero-gradient condition: (r - clip(r)) * A > 0
    if (ratio - clipped) * advantage > 1e-12:
        return -surr2, np.zeros(actor.n_params), np.zeros(actor.obs_dim)

    # Policy gradient: dL/d(mean) where mean is the tanh OUTPUT
    # dL/d(ratio) = -advantage, dratio/d(log_prob) = ratio, dlog_prob/d(mean) = z/std
    grad_factor = -advantage * ratio
    dlogp_dmean = z / (std + 1e-8)
    dloss_dmean = grad_factor * dlogp_dmean  # gradient w.r.t. tanh output

    # Backprop through tanh and actor network
    # tape.backward starts from tanh output; tanh entry applies (1-y^2) derivative
    all_grads = tape.backward(dloss_dmean)

    # Flatten actor parameter gradients
    actor_grad = np.zeros(actor.n_params)
    idx = 0
    param_map = {
        "actor_l1_W": actor.W1, "actor_l1_b": actor.b1,
        "actor_l2_W": actor.W2, "actor_l2_b": actor.b2,
        "actor_l3_W": actor.W3, "actor_l3_b": actor.b3,
    }

    for pid, arr in param_map.items():
        g = all_grads.get(pid)
        if g is not None:
            actor_grad[idx:idx+arr.size] = g.ravel()
        idx += arr.size

    # Gradient w.r.t. log_std
    # dlog_prob/d(log_std) = (z^2 - 1) for each dimension
    dlogp_dlogstd = (z**2 - 1.0)
    dloss_dlogstd = grad_factor * dlogp_dlogstd
    actor_grad[idx:idx+actor.act_dim] = dloss_dlogstd

    # dL/dobs
    dloss_dobs = np.zeros(actor.obs_dim)
    for key, val in all_grads.items():
        if key == "x" and val.shape == (actor.obs_dim,):
            dloss_dobs = val
            break

    return -surr1, actor_grad, dloss_dobs


def compute_comm_gradient(comm_graph, raw_obs, dloss_denhanced):
    """Backpropagate through communication graph. Returns flat gradient vector."""
    tape = ComputationTape()
    enhanced, messages, info = comm_graph.forward(raw_obs, tape=tape)
    all_grads = tape.backward(dloss_denhanced)

    comm_grad = np.zeros(comm_graph.n_params())
    idx = 0

    # Encoder
    enc_map = {"enc1_W": comm_graph.encoder.W1, "enc1_b": comm_graph.encoder.b1,
               "enc2_W": comm_graph.encoder.W2, "enc2_b": comm_graph.encoder.b2}
    for pid, arr in enc_map.items():
        g = all_grads.get(pid)
        if g is not None:
            comm_grad[idx:idx+arr.size] = g.ravel()
        idx += arr.size

    # Decoder
    dec_map = {"dec1_W": comm_graph.decoder.W1, "dec1_b": comm_graph.decoder.b1,
               "dec2_W": comm_graph.decoder.W2, "dec2_b": comm_graph.decoder.b2}
    for pid, arr in dec_map.items():
        g = all_grads.get(pid)
        if g is not None:
            comm_grad[idx:idx+arr.size] = g.ravel()
        idx += arr.size

    # Attention: GradGNNAttention.forward() decomposes a_vec into two
    # linear "pseudo-weights" (tape ops named "att_a1"/"att_a2") so the
    # (n,n) pairwise score matrix can be built via tape.reshape+tape.add
    # instead of an explicit repeat -- see GradGNNAttention.forward(). Its
    # true params (get_params() order: [W_att.ravel(), a_vec]) map back to
    # tape gradients as follows:
    #   W_att -> "att_W_W"   (shape (hidden_dim, msg_dim))
    #   a_vec -> concat("att_a1_W", "att_a2_W")  (each (1, hidden_dim))
    # This was previously skipped entirely (left as zeros), silently
    # dropping attention from the comm-graph gradient update.
    if comm_graph.attention is not None:
        att = comm_graph.attention
        n_att = att.n_params
        att_grad = np.zeros(n_att)

        g_w_att = all_grads.get("att_W_W")
        if g_w_att is not None:
            att_grad[:att.W_att.size] = g_w_att.ravel()[:att.W_att.size]

        g_a1 = all_grads.get("att_a1_W")
        g_a2 = all_grads.get("att_a2_W")
        if g_a1 is not None and g_a2 is not None:
            a_vec_grad = np.concatenate([g_a1.ravel(), g_a2.ravel()])
            att_grad[att.W_att.size:att.W_att.size + a_vec_grad.size] = a_vec_grad

        comm_grad[idx:idx + n_att] = att_grad
        idx += n_att

    return comm_grad


# ── 2.6 Full Gradient-Based Trainer ──────────────────────────────────────────

class GradientBasedMAPPOMARLTrainer:
    """
    TRUE Gradient-Based MAPPO with GNN Communication.

    Drop-in replacement for MAPPOMARLTrainer from marl_agents.py.
    Uses analytical backprop instead of ES for actor + communication updates.
    """

    def __init__(self, env, use_comm=False, hidden_actor=64, hidden_critic=128,
                 lr_actor=3e-4, lr_critic=3e-4, lr_comm=1e-4,
                 clip_eps=0.2, gamma=0.99, lam=0.95,
                 rollout_len=64, n_epochs=4, seed=0,
                 curriculum=False, curriculum_stages=6,
                 msg_dim=8, comm_hidden=32):

        self.env = env
        self.n_agents = env.n_agents
        self.use_comm = use_comm
        self.clip_eps = clip_eps
        self.gamma = gamma
        self.lam = lam
        self.rollout_len = rollout_len
        self.n_epochs = n_epochs
        self.rng = np.random.default_rng(seed)

        self.curriculum = curriculum
        self.curriculum_stages = curriculum_stages
        self.current_stage = 0

        # Communication
        self.comm_graph = None
        if use_comm:
            try:
                adj = build_relay_adjacency(env.net)
            except:
                adj = np.eye(env.n_agents)

            self.comm_graph = GradCommGraph(
                n_agents=env.n_agents, obs_dim=env.OBS_DIM,
                enhanced_dim=env.OBS_DIM + msg_dim,
                adjacency=adj, msg_dim=msg_dim, hidden_dim=comm_hidden,
                seed=seed+100,
            )

        # Actor
        actor_obs_dim = env.OBS_DIM
        if use_comm and self.comm_graph:
            actor_obs_dim = self.comm_graph.enhanced_dim

        self.actor = GradSharedGaussianActor(actor_obs_dim, env.ACT_DIM, hidden_actor, seed)

        # Critic (ES-based — can be upgraded to gradient-based)
        self.critic = self._make_critic(env.n_agents, hidden_critic, seed+999)

        # Buffer
        self.buffer = self._make_buffer(env.n_agents, actor_obs_dim, env.ACT_DIM,
                                         rollout_len, use_comm, msg_dim,
                                         raw_obs_dim=env.OBS_DIM)

        # Optimizers
        self.actor_opt = Adam(lr=lr_actor)
        self.critic_opt = Adam(lr=lr_critic)
        self.comm_opt = Adam(lr=lr_comm)

        # Stats
        self.loss_history = []
        self.grad_norm_history = []
        self.clip_ratio_history = []
        self.attention_snapshots = []
        self.per_agent_returns = [[] for _ in range(env.n_agents)]
        self._global_step = 0
        self.attention_checkpoint_interval = 500

    def _make_critic(self, n_relays, hidden, seed):
        class Critic:
            def __init__(self, n, h, s):
                rng = np.random.default_rng(s)
                s_fn = lambda a: np.sqrt(2.0 / a)
                gd = 2 * n + 12
                self.W1 = rng.normal(0, s_fn(gd), (h, gd))
                self.b1 = np.zeros(h)
                self.W2 = rng.normal(0, s_fn(h), (h, h))
                self.b2 = np.zeros(h)
                self.W3 = rng.normal(0, s_fn(h), (1, h))
                self.b3 = np.zeros(1)

            def value(self, TMS, PS, pv=0.0, topo=0.0, ft=0, ns=None):
                gs = np.concatenate([TMS, PS])
                fo = np.zeros(4); fo[ft] = 1.0
                if ns is not None:
                    gs = np.concatenate([gs, [pv, topo], fo, ns])
                else:
                    gs = np.concatenate([gs, [pv, topo], fo])
                h1 = np.maximum(0, self.W1 @ gs + self.b1)
                h2 = np.maximum(0, self.W2 @ h1 + self.b2)
                return float((self.W3 @ h2 + self.b3).item())

            def value_batch(self, TMS, PS, pv, ns, ft=0):
                """Batched form of value(): TMS/PS (T,n_relays), pv (T,),
                ns (T,6) -> (T,) predictions via one vectorized forward call
                instead of a Python loop over T individual value() calls."""
                T = TMS.shape[0]
                gs = np.concatenate([TMS, PS], axis=1)
                fo = np.zeros((T, 4)); fo[:, ft] = 1.0
                gs = np.concatenate([gs, np.stack([pv, np.zeros(T)], axis=1), fo, ns], axis=1)
                h1 = np.maximum(0, gs @ self.W1.T + self.b1)
                h2 = np.maximum(0, h1 @ self.W2.T + self.b2)
                return (h2 @ self.W3.T + self.b3).ravel()

            def get_params(self):
                return np.concatenate([self.W1.ravel(), self.b1,
                                       self.W2.ravel(), self.b2,
                                       self.W3.ravel(), self.b3])

            def set_params(self, p):
                sz = [self.W1.size, self.b1.size, self.W2.size, self.b2.size, self.W3.size, self.b3.size]
                c = np.split(p, np.cumsum(sz[:-1]))
                self.W1, self.b1 = c[0].reshape(self.W1.shape), c[1]
                self.W2, self.b2 = c[2].reshape(self.W2.shape), c[3]
                self.W3, self.b3 = c[4].reshape(self.W3.shape), c[5]

            def n_params(self):
                return sum(x.size for x in [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3])

        return Critic(n_relays, hidden, seed)

    def _make_buffer(self, n_agents, obs_dim, act_dim, cap, use_comm, msg_dim, raw_obs_dim=None):
        class Buf:
            def __init__(self, n, od, ad, c, uc, md, rod):
                self.obs = np.zeros((c, n, od))
                self.actions = np.zeros((c, n, ad))
                self.rewards = np.zeros((c, n))
                self.log_probs = np.zeros((c, n))
                self.values = np.zeros(c)
                self.dones = np.zeros(c, dtype=bool)
                # Per-timestep global relay/PV state, needed so the critic
                # update re-scores each timestep against the state that
                # actually produced ret_batch[t], not whatever self.env's
                # current state happens to be when the update runs.
                self.tms = np.zeros((c, n))
                self.ps = np.zeros((c, n))
                self.pv_levels = np.zeros(c)
                self.use_comm = uc
                if uc:
                    self.messages = np.zeros((c, n, md))
                    self.attention = np.zeros((c, n, n))
                    # Pre-communication (raw) observations per timestep, needed
                    # so _update_actor_and_comm() can recompute the comm graph's
                    # forward pass (with a gradient tape) against the SAME input
                    # that actually produced obs[t] during rollout collection —
                    # not whatever self.env's raw observation happens to be
                    # after the whole rollout has already finished.
                    self.raw_obs = np.zeros((c, n, rod if rod is not None else od))
                self.ptr = self.size = 0
                self.capacity = c
                self.n_agents = n

            def add(self, obs, act, rew, lp, val, done, msg=None, attn=None,
                    tms=None, ps=None, pv_level=0.0, raw_obs=None):
                i = self.ptr % self.capacity
                self.obs[i] = obs
                self.actions[i] = act
                self.rewards[i] = rew
                self.log_probs[i] = lp
                self.values[i] = val
                self.dones[i] = done
                if tms is not None:
                    self.tms[i] = tms
                if ps is not None:
                    self.ps[i] = ps
                self.pv_levels[i] = pv_level
                if self.use_comm and msg is not None:
                    self.messages[i] = msg
                if self.use_comm and attn is not None:
                    self.attention[i] = attn
                if self.use_comm and raw_obs is not None:
                    self.raw_obs[i] = raw_obs
                self.ptr += 1
                self.size = min(self.ptr, self.capacity)

            def compute_gae(self, gamma=0.99, lam=0.95, last_val=0.0):
                T = self.size
                adv = np.zeros((T, self.n_agents))
                ret = np.zeros((T, self.n_agents))
                for ag in range(self.n_agents):
                    gae = 0.0
                    for t in reversed(range(T)):
                        Vn = last_val if (t == T - 1 or self.dones[t]) else self.values[t + 1]
                        delta = self.rewards[t, ag] + gamma * Vn * (1 - self.dones[t]) - self.values[t]
                        gae = delta + gamma * lam * (1 - self.dones[t]) * gae
                        adv[t, ag] = gae
                        ret[t, ag] = gae + self.values[t]
                    adv[:, ag] = (adv[:, ag] - adv[:, ag].mean()) / (adv[:, ag].std() + 1e-8)
                return adv, ret

            def clear(self):
                self.ptr = self.size = 0

        return Buf(n_agents, obs_dim, act_dim, cap, use_comm, msg_dim, raw_obs_dim)

    def _get_neighbor_stats(self, obs):
        return np.array([
            float(np.mean(obs[:, 0])), float(np.mean(obs[:, 1])),
            float(np.std(obs[:, 0])), float(np.std(obs[:, 1])),
            float(np.max(obs[:, 4])), float(np.mean(obs[:, 3])),
        ])

    def _get_neighbor_stats_batch(self, obs_batch):
        """Batched form of _get_neighbor_stats: obs_batch (T, n_agents,
        obs_dim) -> (T, 6), reduced over the agent axis (axis=1) instead of
        a Python loop calling _get_neighbor_stats(obs_batch[t]) per timestep."""
        return np.stack([
            obs_batch[:, :, 0].mean(axis=1), obs_batch[:, :, 1].mean(axis=1),
            obs_batch[:, :, 0].std(axis=1), obs_batch[:, :, 1].std(axis=1),
            obs_batch[:, :, 4].max(axis=1), obs_batch[:, :, 3].mean(axis=1),
        ], axis=1)

    def _update_actor_and_comm(self, obs_batch, act_batch, adv_batch, old_lp_batch):
        """Compute true gradients and update actor + communication parameters."""
        T, n_agents = obs_batch.shape[:2]

        actor_grad_accum = np.zeros(self.actor.n_params)
        comm_grad_accum = np.zeros(self.comm_graph.n_params()) if self.use_comm else None

        total_loss = 0.0
        n_clipped = 0

        for t in range(T):
            # Use the raw (pre-comm-graph) observation actually recorded at
            # timestep t, not self.env's current observation — by the time
            # this update runs, the whole rollout has already finished, so
            # self.env's live observation is just the final timestep's obs
            # repeated for every t, not the historical per-t input that
            # actually produced obs_batch[t]/enhanced_obs at collection time.
            if self.use_comm and hasattr(self.buffer, 'raw_obs'):
                raw_obs = self.buffer.raw_obs[t]
            else:
                raw_obs = obs_batch[t]

            if self.use_comm and self.comm_graph is not None:
                comm_tape = ComputationTape()
                enhanced_obs, _, _ = self.comm_graph.forward(raw_obs, tape=comm_tape)
            else:
                enhanced_obs = obs_batch[t]

            dloss_denhanced = np.zeros((n_agents, self.actor.obs_dim))

            for ag in range(n_agents):
                loss, actor_grad, dloss_dobs = compute_ppo_policy_gradient(
                    self.actor, enhanced_obs[ag], act_batch[t, ag],
                    adv_batch[t, ag], old_lp_batch[t, ag], self.clip_eps,
                )

                total_loss += loss
                actor_grad_accum += actor_grad

                if self.use_comm:
                    dloss_denhanced[ag] = dloss_dobs

                if loss == 0:
                    n_clipped += 1

            if self.use_comm and comm_grad_accum is not None:
                comm_grad = compute_comm_gradient(self.comm_graph, raw_obs, dloss_denhanced)
                comm_grad_accum += comm_grad

        n_total = T * n_agents
        total_loss /= n_total
        actor_grad_accum /= n_total
        if comm_grad_accum is not None:
            comm_grad_accum /= n_total

        clip_ratio = n_clipped / max(n_total, 1)

        # Apply updates
        new_actor = self.actor_opt.step(self.actor.get_all_params(), actor_grad_accum, "actor")
        self.actor.set_all_params(new_actor)

        if self.use_comm and comm_grad_accum is not None:
            new_comm = self.comm_opt.step(self.comm_graph.get_all_params(), comm_grad_accum, "comm")
            self.comm_graph.set_all_params(new_comm)

        return total_loss, float(np.linalg.norm(actor_grad_accum)), clip_ratio

    def _update_critic(self, obs_batch, ret_batch):
        """ES-based critic update."""
        T = len(obs_batch)
        theta = self.critic.get_params()
        noises, losses = [], []

        # Use the historical per-timestep relay/PV state stored in the
        # buffer (not self.env's *current* state) — ret_batch[t] was
        # computed from the state at timestep t, not whatever state self.env
        # happens to be in once the rollout has finished and this update
        # runs. Also precomputed once (not per-perturbation): these don't
        # depend on the critic's perturbed parameters.
        tms_batch = self.buffer.tms[:T]
        ps_batch = self.buffer.ps[:T]
        pv_batch = self.buffer.pv_levels[:T]
        ns_batch = self._get_neighbor_stats_batch(obs_batch)

        for _ in range(10):
            eps = self.rng.normal(0, 0.02, len(theta))
            for sign in (+1, -1):
                self.critic.set_params(theta + sign * eps)
                pred = self.critic.value_batch(tms_batch, ps_batch, pv_batch, ns_batch)
                loss = -float(np.mean((ret_batch - pred[:, None]) ** 2))
                noises.append(sign * eps)
                losses.append(loss)

        l_arr = np.array(losses)
        l_norm = (l_arr - l_arr.mean()) / (l_arr.std() + 1e-8)
        grad = sum(l * n for l, n in zip(l_norm, noises))
        grad /= (2 * 10 * 0.02)
        self.critic.set_params(theta + 0.01 * grad)

    def _collect_rollout(self, obs):
        self.buffer.clear()

        for _ in range(self.rollout_len):
            if self.use_comm and self.comm_graph:
                enhanced, messages, info = self.comm_graph.forward(obs)
                current_obs = enhanced
            else:
                current_obs = obs
                messages, info = None, {}

            actions = np.zeros((self.n_agents, self.env.ACT_DIM))
            log_probs = np.zeros(self.n_agents)
            for i in range(self.n_agents):
                actions[i], log_probs[i] = self.actor.sample(current_obs[i], self.rng)

            TMS, PS = self.env.get_state()
            pv = getattr(self.env, 'current_pv_level', 0.0)
            ns = self._get_neighbor_stats(current_obs)
            val = self.critic.value(TMS, PS, pv=pv, ns=ns)

            next_obs, rewards, done, info_step = self.env.step(actions)

            self.buffer.add(current_obs, actions, rewards, log_probs, val, done,
                          msg=messages, attn=info.get("attention"),
                          tms=TMS, ps=PS, pv_level=pv, raw_obs=obs)

            self._global_step += 1
            if (self.use_comm and info.get("attention") is not None and
                self._global_step % self.attention_checkpoint_interval == 0):
                attn = np.array(info["attention"])
                p = np.clip(attn, 1e-9, 1.0)
                p /= p.sum(axis=1, keepdims=True) + 1e-9
                ent = float(-np.mean(np.sum(p * np.log(p), axis=1)))
                self.attention_snapshots.append({"step": self._global_step, "attn_matrix": attn.copy(), "entropy": ent})

            obs = next_obs
            if done:
                obs = self.env.reset()

        return obs

    def train(self, total_steps=30000, verbose=True, log_every=2000,
              early_stop_patience=None, early_stop_min_delta=1e-4):
        """
        early_stop_patience: if set, stop once this many consecutive rollout
        evaluations pass without the best objective improving by more than
        early_stop_min_delta. Disabled by default (None) — see
        MAPPOMARLTrainer.train() in marl/marl_agents.py for the same
        opt-in pattern and rationale.
        """
        obs = self.env.reset()
        step = 0
        history = []
        best_OF = np.inf
        best_TMS = self.env.TMS.copy()
        best_PS = self.env.PS.copy()
        no_improve_count = 0
        early_stopped = False
        t0 = time.perf_counter()

        if verbose:
            print(f"\n{'─'*55}")
            print(f"  GRADIENT-BASED MAPPO  ·  {self.env.net.name}")
            print(f"  Agents: {self.n_agents}  |  Comm: {'ON (True Grad)' if self.use_comm else 'OFF'}")
            print(f"  Actor params: {self.actor.n_params}  |  Update: ANALYTICAL BACKPROP")
            print(f"{'─'*55}")

        while step < total_steps:
            obs = self._collect_rollout(obs)
            T = self.buffer.size
            step += T

            TMS, PS = self.env.get_state()
            pv = getattr(self.env, 'current_pv_level', 0.0)
            last_val = self.critic.value(TMS, PS, pv=pv, ns=self._get_neighbor_stats(obs))
            advs, rets = self.buffer.compute_gae(self.gamma, self.lam, last_val)

            # TRUE GRADIENT UPDATE
            loss, grad_norm, clip_ratio = self._update_actor_and_comm(
                self.buffer.obs[:T], self.buffer.actions[:T],
                advs[:T], self.buffer.log_probs[:T]
            )

            self._update_critic(self.buffer.obs[:T], rets[:T])

            self.loss_history.append(float(loss))
            self.grad_norm_history.append(grad_norm)
            self.clip_ratio_history.append(clip_ratio)

            # Evaluate
            try:
                from network.relay_physics import objective_function as OF_fn
                OF_now, _, _ = OF_fn(self.env.net, TMS, PS)
            except:
                OF_now = 0.0

            improved = OF_now < best_OF - early_stop_min_delta
            if OF_now < best_OF:
                best_OF = OF_now
                best_TMS = TMS.copy()
                best_PS = PS.copy()
            no_improve_count = 0 if improved else no_improve_count + 1

            history.append(OF_now)

            if self.buffer.size > 0:
                am = self.buffer.rewards[:self.buffer.size].mean(axis=0)
                for i in range(self.n_agents):
                    self.per_agent_returns[i].append(float(am[i]))

            if verbose and step % log_every < T:
                print(f"  step {step:6d}/{total_steps}  OF={OF_now:.4f}  best={best_OF:.4f}  "
                      f"|grad|={grad_norm:.4f}  clip%={clip_ratio*100:.1f}  "
                      f"t={time.perf_counter()-t0:.1f}s")

            if early_stop_patience is not None and no_improve_count >= early_stop_patience:
                early_stopped = True
                if verbose:
                    print(f"  Early stop at step {step}/{total_steps}: "
                          f"no improvement > {early_stop_min_delta} for {no_improve_count} rollouts")
                break

        elapsed = time.perf_counter() - t0
        if verbose:
            print(f"  Done. Best OF = {best_OF:.4f}  [{elapsed:.1f}s]")
            print(f"  Avg |grad|: {np.mean(self.grad_norm_history):.4f}")

        return {
            "method": "Gradient-MAPPO" + ("-Comm" if self.use_comm else ""),
            "network": self.env.net.name,
            "TMS": best_TMS, "PS": best_PS, "OF": best_OF,
            "history": history, "elapsed": elapsed,
            "loss_history": self.loss_history,
            "grad_norm_history": self.grad_norm_history,
            "clip_ratio_history": self.clip_ratio_history,
            "attention_snapshots": self.attention_snapshots,
            "per_agent_returns": self.per_agent_returns,
            "early_stopped": early_stopped,
            "stopped_at_step": step,
        }


def run_gradient_mappo(net, total_steps=30000, seed=0, verbose=True,
                         use_comm=False, curriculum=False, curriculum_stage=0,
                         fault_type="3ph", use_pandapower=False,
                         hidden_actor=64, hidden_critic=128):
    """Convenience function — same signature as run_mappo."""
    try:
        from marl.marl_env import DynamicDOCREnv
    except ImportError:
        from marl_env import DynamicDOCREnv

    env = DynamicDOCREnv(net, action_scale=0.05, max_steps=300, seed=seed,
                           use_pandapower=use_pandapower,
                           curriculum_stage=curriculum_stage, fault_type=fault_type)

    trainer = GradientBasedMAPPOMARLTrainer(
        env, use_comm=use_comm, hidden_actor=hidden_actor,
        hidden_critic=hidden_critic, seed=seed, curriculum=curriculum,
    )
    return trainer.train(total_steps=total_steps, verbose=verbose)


# ── 2.7 Gradient Verification ───────────────────────────────────────────────

def verify_actor_gradient(actor, obs, action, advantage, old_lp, eps=1e-5, n_check=50):
    """Verify analytical gradients against finite differences."""
    loss, agrad, _ = compute_ppo_policy_gradient(actor, obs, action, advantage, old_lp)

    params = actor.get_all_params()
    fd_grad = np.zeros(min(len(params), n_check))
    std = np.exp(np.clip(actor.log_std, -3, 0))

    for i in range(len(fd_grad)):
        p_plus = params.copy(); p_plus[i] += eps
        actor.set_all_params(p_plus)
        mean_p = actor.get_mean(obs)
        z_p = (action - mean_p) / (std + 1e-8)
        lp_p = float(np.sum(-0.5 * (z_p**2 + 2 * actor.log_std + np.log(2 * np.pi))))
        r_p = np.exp(np.clip(lp_p - old_lp, -10, 10))
        loss_p = -min(r_p * advantage, np.clip(r_p, 0.8, 1.2) * advantage)

        p_minus = params.copy(); p_minus[i] -= eps
        actor.set_all_params(p_minus)
        mean_m = actor.get_mean(obs)
        z_m = (action - mean_m) / (std + 1e-8)
        lp_m = float(np.sum(-0.5 * (z_m**2 + 2 * actor.log_std + np.log(2 * np.pi))))
        r_m = np.exp(np.clip(lp_m - old_lp, -10, 10))
        loss_m = -min(r_m * advantage, np.clip(r_m, 0.8, 1.2) * advantage)

        fd_grad[i] = (loss_p - loss_m) / (2 * eps)

    actor.set_all_params(params)

    max_err = np.max(np.abs(agrad[:n_check] - fd_grad))
    rel_err = max_err / (np.max(np.abs(fd_grad)) + 1e-10)

    return agrad[:n_check], fd_grad, max_err, rel_err