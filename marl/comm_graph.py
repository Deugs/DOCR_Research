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

        # 3. Compute attention weights
        if self.attention is not None:
            attention = self.attention.compute_attention(messages, self.adjacency)
        else:
            # Uniform attention over neighbors
            row_sums = self.adjacency.sum(axis=1, keepdims=True)
            attention = self.adjacency / np.maximum(row_sums, 1e-8)

        # 4. Aggregate messages
        aggregated = attention @ messages  # (n_agents, msg_dim)

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

    # Also connect relays at the same bus (same protection zone)
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
    """
    Build neighbor-aware observations by concatenating neighbor features.

    Parameters:
        net: NetworkData instance
        observations: (n_agents, obs_dim) current observations
        relay_pairs: List of pair indices per relay
        max_neighbors: Maximum neighbors to include

    Returns:
        neighbor_obs: (n_agents, max_neighbors * obs_dim) padded neighbor features
    """
    n = net.n_relays
    obs_dim = observations.shape[1]
    neighbor_obs = np.zeros((n, max_neighbors, obs_dim))

    for i in range(n):
        # Find unique neighbors from relay pairs
        neighbors = set()
        for pidx in relay_pairs[i]:
            pair = net.pairs[pidx]
            if pair.primary == i:
                neighbors.add(pair.backup)
            else:
                neighbors.add(pair.primary)

        # Also add relays at same bus
        for pair in net.pairs:
            if pair.primary == i:
                for other_pair in net.pairs:
                    if other_pair.bus_pri == pair.bus_pri and other_pair.primary != i:
                        neighbors.add(other_pair.primary)

        neighbors = sorted(list(neighbors))[:max_neighbors]
        for j, neighbor_idx in enumerate(neighbors):
            neighbor_obs[i, j] = observations[neighbor_idx]

    return neighbor_obs.reshape(n, -1)
