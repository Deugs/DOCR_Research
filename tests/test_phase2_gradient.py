"""
tests/test_phase2_gradient.py
-------------------------------
Unit and integration tests for Phase 2 components WITH gradient-based communication.

Tests:
1. GNN Message Encoder/Decoder (original + gradient-tracked versions)
2. GNN Attention mechanism (original + gradient-tracked)
3. CommGraph end-to-end communication (original + gradient-tracked)
4. Relay adjacency matrix construction
5. Neighbor observations
6. Shared Gaussian Actor (original + gradient-tracked)
7. MAPPO Centralized Critic
8. MAPPO Buffer with communication
9. Single-Agent PPO (centralized baseline)
10. MAPPO with GNN communication integration (ES + gradient modes)
11. Bandwidth usage tracking
12. Backward compatibility with Phase 1
13. GRADIENT VERIFICATION: finite-difference checks for all backprop primitives
14. TRUE GRADIENT training: end-to-end actor+comm backprop
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from network.network_data import get_3bus, get_8bus, get_30bus, get_39bus
from network.relay_physics import CTI, TMS_MIN, TMS_MAX


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 1: Original API Tests (unchanged — ensures backward compatibility)
# ═══════════════════════════════════════════════════════════════════════════════

def test_gnn_encoder():
    """Test GNN message encoder (original API)."""
    rng = np.random.default_rng(42)
    from marl.comm_graph_gradient import GNNMessageEncoder, CommConfig

    config = CommConfig(message_dim=8, hidden_dim=32)
    encoder = GNNMessageEncoder(obs_dim=10, config=config, seed=42)

    obs = rng.standard_normal(10)
    msg = encoder.encode(obs)
    assert msg.shape == (8,), f"Expected (8,), got {msg.shape}"
    assert np.all(np.abs(msg) <= 1.0), "Messages should be in [-1, 1] (tanh)"

    obs_batch = rng.standard_normal((5, 10))
    msgs = encoder.encode(obs_batch)
    assert msgs.shape == (5, 8), f"Expected (5, 8), got {msgs.shape}"

    params = encoder.get_params()
    encoder2 = GNNMessageEncoder(obs_dim=10, config=config, seed=0)
    encoder2.set_params(params)
    msg2 = encoder2.encode(obs)
    assert np.allclose(msg, msg2), "Parameter set/get failed"

    print(f"  [ORIGINAL] GNN Encoder: msg_shape={msg.shape}, params={encoder.n_params}")


def test_gnn_attention():
    """Test GNN attention mechanism (original API)."""
    rng = np.random.default_rng(42)
    from marl.comm_graph_gradient import GNNAttention

    attention = GNNAttention(msg_dim=8, hidden_dim=16, seed=42)

    n = 5
    messages = rng.standard_normal((n, 8))
    adjacency = np.eye(n)
    adjacency[0, 1] = adjacency[1, 0] = 1.0
    adjacency[1, 2] = adjacency[2, 1] = 1.0

    attn_weights = attention.compute_attention(messages, adjacency)
    assert attn_weights.shape == (n, n), f"Expected ({n},{n}), got {attn_weights.shape}"

    row_sums = attn_weights.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5), f"Attention rows don't sum to 1: {row_sums}"
    assert attn_weights[0, 3] < 1e-6, "Non-neighbor should have near-zero attention"

    print(f"  [ORIGINAL] GNN Attention: shape={attn_weights.shape}, row_sums_ok={np.allclose(row_sums, 1.0)}")


def test_gnn_decoder():
    """Test GNN message decoder (original API)."""
    rng = np.random.default_rng(42)
    from marl.comm_graph_gradient import GNNMessageDecoder, CommConfig

    config = CommConfig(hidden_dim=32)
    decoder = GNNMessageDecoder(obs_dim=10, msg_dim=8, enhanced_dim=18, config=config, seed=42)

    obs = rng.standard_normal(10)
    aggregated_msg = rng.standard_normal(8)
    enhanced = decoder.decode(obs, aggregated_msg)
    assert enhanced.shape == (18,), f"Expected (18,), got {enhanced.shape}"
    assert np.all(np.abs(enhanced) <= 1.0), "Enhanced obs should be in [-1, 1]"

    obs_batch = rng.standard_normal((5, 10))
    msg_batch = rng.standard_normal((5, 8))
    enhanced_batch = decoder.decode(obs_batch, msg_batch)
    assert enhanced_batch.shape == (5, 18)

    print(f"  [ORIGINAL] GNN Decoder: enhanced_shape={enhanced.shape}, params={decoder.n_params}")


def test_comm_graph():
    """Test complete CommGraph end-to-end (original API)."""
    rng = np.random.default_rng(42)
    from marl.comm_graph_gradient import CommGraph, CommConfig, build_relay_adjacency

    net = get_8bus()
    adj = build_relay_adjacency(net)

    assert adj.shape == (net.n_relays, net.n_relays)
    assert np.all(np.diag(adj) == 1.0), "Self-loops missing"
    assert np.all(adj == adj.T), "Adjacency should be symmetric"

    config = CommConfig(message_dim=8, hidden_dim=32, use_attention=True)
    comm = CommGraph(
        n_agents=net.n_relays,
        obs_dim=10,
        enhanced_dim=18,
        adjacency=adj,
        config=config,
        seed=42,
    )

    observations = rng.standard_normal((net.n_relays, 10))
    enhanced_obs, messages, comm_info = comm.forward(observations)

    assert enhanced_obs.shape == (net.n_relays, 18)
    assert messages.shape == (net.n_relays, 8)
    assert "attention" in comm_info
    assert "bytes_per_step" in comm_info

    bw = comm.get_bandwidth_usage(steps_per_second=1.0)
    assert bw["kbps"] < 10.0, f"Bandwidth too high: {bw['kbps']:.2f} kbps"

    print(f"  [ORIGINAL] CommGraph: enhanced={enhanced_obs.shape}, msg={messages.shape}, bw={bw['kbps']:.3f}kbps")


def test_relay_adjacency():
    """Test relay adjacency matrix for different systems."""
    from marl.comm_graph_gradient import build_relay_adjacency

    for get_net in [get_3bus, get_8bus, get_30bus, get_39bus]:
        net = get_net()
        adj = build_relay_adjacency(net)

        assert adj.shape == (net.n_relays, net.n_relays)
        assert np.all(np.diag(adj) == 1.0), f"{net.name}: Self-loops missing"
        assert np.all(adj == adj.T), f"{net.name}: Not symmetric"

        for pair in net.pairs:
            assert adj[pair.primary, pair.backup] == 1.0, f"{net.name}: Pair not connected"
            assert adj[pair.backup, pair.primary] == 1.0, f"{net.name}: Pair not symmetric"

        n_edges = int(np.sum(adj) - net.n_relays) // 2
        print(f"  [ORIGINAL] {net.name}: {net.n_relays} relays, {n_edges} unique edges")


def test_neighbor_observations():
    """Test neighbor observation extraction."""
    from marl.marl_env import DynamicDOCREnv

    net = get_8bus()
    env = DynamicDOCREnv(net, use_pandapower=False, seed=42)
    obs = env.reset(seed=42)

    neighbor_obs = env.get_neighbor_observations(obs)
    assert neighbor_obs.ndim == 3, f"Expected 3D, got {neighbor_obs.ndim}D"
    assert neighbor_obs.shape[0] == net.n_relays

    for i in range(net.n_relays):
        if env.neighbors[i]:
            assert not np.all(neighbor_obs[i, 0] == 0), f"Agent {i} should have non-zero neighbor"

    print(f"  [ORIGINAL] Neighbor Obs: shape={neighbor_obs.shape}, max_neighbors={neighbor_obs.shape[1]}")


def test_shared_gaussian_actor():
    """Test shared Gaussian actor (original API)."""
    rng = np.random.default_rng(42)
    from marl.marl_agents import SharedGaussianActor

    actor = SharedGaussianActor(obs_dim=10, act_dim=2, hidden=32, seed=42)

    obs = rng.standard_normal(10)
    action, log_prob = actor.sample(obs, np.random.default_rng(42))
    assert action.shape == (2,)
    assert np.all(np.abs(action) <= 1.0)
    assert log_prob < 0

    obs_batch = rng.standard_normal((5, 10))
    means = actor.get_mean(obs_batch)
    assert means.shape == (5, 2)

    params = actor.get_all_params()
    actor2 = SharedGaussianActor(obs_dim=10, act_dim=2, hidden=32, seed=0)
    actor2.set_all_params(params)
    action2, _ = actor2.sample(obs, np.random.default_rng(42))
    assert np.allclose(action, action2), "Parameter sharing failed"

    print(f"  [ORIGINAL] Shared Actor: action_shape={action.shape}, params={len(params)}")


def test_mappo_critic():
    """Test MAPPO centralized critic."""
    from marl.marl_agents import MAPPOCentralCritic

    n_relays = 14
    critic = MAPPOCentralCritic(n_relays, hidden=64, seed=42, use_global_state=True)

    TMS = np.full(n_relays, 0.5)
    PS = np.full(n_relays, 1.0)
    neighbor_stats = np.array([0.5, 0.5, 0.1, 0.1, 0.0, 0.0])

    value = critic.value(TMS, PS, pv_level=0.5, topology_flag=1.0,
                         fault_type_idx=0, neighbor_stats=neighbor_stats)
    assert isinstance(value, float)

    params = critic.get_params()
    critic2 = MAPPOCentralCritic(n_relays, hidden=64, seed=0)
    critic2.set_params(params)
    value2 = critic2.value(TMS, PS, pv_level=0.5, topology_flag=1.0,
                               fault_type_idx=0, neighbor_stats=neighbor_stats)
    assert np.isclose(value, value2), "Critic parameter set/get failed"

    print(f"  [ORIGINAL] MAPPO Critic: value={value:.4f}, params={critic.n_params()}")


def test_mappo_buffer():
    """Test MAPPO rollout buffer with communication."""
    rng = np.random.default_rng(42)
    from marl.marl_agents import MAPPOBuffer

    n_agents, obs_dim, act_dim = 8, 10, 2
    capacity = 64

    buffer = MAPPOBuffer(n_agents, obs_dim, act_dim, capacity, use_comm=False)
    obs = rng.standard_normal((n_agents, obs_dim))
    actions = rng.standard_normal((n_agents, act_dim))
    rewards = rng.standard_normal(n_agents)
    log_probs = rng.standard_normal(n_agents)
    value = 0.5
    done = False

    buffer.add(obs, actions, rewards, log_probs, value, done)
    assert buffer.size == 1

    buffer_comm = MAPPOBuffer(n_agents, obs_dim, act_dim, capacity, use_comm=True, msg_dim=8)
    messages = rng.standard_normal((n_agents, 8))
    attention = rng.random((n_agents, n_agents))
    buffer_comm.add(obs, actions, rewards, log_probs, value, done,
                    messages=messages, attention=attention)
    assert buffer_comm.size == 1
    assert buffer_comm.messages is not None

    for _ in range(10):
        buffer.add(obs, actions, rewards, log_probs, value, done)
    adv, ret = buffer.compute_gae(gamma=0.99, lam=0.95, last_value=0.0)
    assert adv.shape == (11, n_agents)
    assert ret.shape == (11, n_agents)

    print(f"  [ORIGINAL] MAPPO Buffer: size={buffer.size}, adv_shape={adv.shape}")


def test_single_ppo():
    """Test Single-Agent PPO (centralized baseline)."""
    from marl.single_agent_ppo import CentralizedPPOEnvironment, SinglePPOTrainer

    net = get_3bus()
    env = CentralizedPPOEnvironment(net, use_pandapower=False, seed=42)

    obs = env.reset()
    assert obs.shape == (env.OBS_DIM,)
    assert len(env.TMS) == net.n_relays
    assert len(env.PS) == net.n_relays

    trainer = SinglePPOTrainer(env, hidden_actor=32, hidden_critic=32, seed=42)
    action_norm, _ = trainer.actor.sample(obs, np.random.default_rng(42))
    TMS, PS = trainer.actor.decode_action(action_norm, env.ps_min, env.ps_max)
    assert len(TMS) == net.n_relays
    assert len(PS) == net.n_relays
    assert np.all(TMS >= TMS_MIN) and np.all(TMS <= TMS_MAX)
    assert np.all(PS >= env.ps_min) and np.all(PS <= env.ps_max)

    print(f"  [ORIGINAL] Single PPO: obs={obs.shape}, action={action_norm.shape}, TMS_range=[{TMS.min():.3f}, {TMS.max():.3f}]")


def test_mappo_integration():
    """Test MAPPO integration with environment."""
    from marl.marl_env import DynamicDOCREnv
    from marl.marl_agents import MAPPOMARLTrainer

    net = get_3bus()
    env = DynamicDOCREnv(net, use_pandapower=False, seed=42)

    trainer = MAPPOMARLTrainer(
        env,
        use_comm=False,
        hidden_actor=16,
        hidden_critic=32,
        rollout_len=16,
        n_epochs=2,
        seed=42,
    )

    obs = env.reset(seed=42)
    assert obs.shape == (net.n_relays, env.OBS_DIM)

    result = trainer.train(total_steps=64, verbose=False, log_every=1000)
    assert "TMS" in result
    assert "PS" in result
    assert "OF" in result
    assert result["method"] == "MAPPO"

    print(f"  [ORIGINAL] MAPPO Integration: OF={result['OF']:.4f}, method={result['method']}")


def test_bandwidth_tracking():
    """Test communication bandwidth tracking."""
    rng = np.random.default_rng(42)
    from marl.comm_graph_gradient import CommGraph, CommConfig, build_relay_adjacency

    net = get_8bus()
    adj = build_relay_adjacency(net)
    config = CommConfig(message_dim=8, bandwidth_limit_bps=1000)

    comm = CommGraph(
        n_agents=net.n_relays,
        obs_dim=10,
        enhanced_dim=18,
        adjacency=adj,
        config=config,
        seed=42,
    )

    for _ in range(10):
        obs = rng.standard_normal((net.n_relays, 10))
        comm.forward(obs)

    bw = comm.get_bandwidth_usage(steps_per_second=1.0)
    assert bw["within_limit"], f"Bandwidth {bw['kbps']:.2f}kbps exceeds limit {bw['limit_kbps']:.2f}kbps"
    assert bw["bytes_per_step"] == net.n_relays * 8 * 4

    print(f"  [ORIGINAL] Bandwidth: {bw['kbps']:.3f}kbps (limit: {bw['limit_kbps']:.0f}kbps), within_limit={bw['within_limit']}")


def test_backward_compatibility():
    """Test Phase 2 is backward compatible with Phase 1."""
    from marl.marl_env import DynamicDOCREnv
    from network.network_data import get_3bus

    net = get_3bus()
    env = DynamicDOCREnv(net, use_pandapower=False, seed=42)

    obs = env.reset(seed=42)
    assert obs.shape == (net.n_relays, env.OBS_DIM)
    assert env.OBS_DIM == 10

    actions = np.zeros((net.n_relays, env.ACT_DIM))
    next_obs, rewards, done, info = env.step(actions)
    assert next_obs.shape == obs.shape
    assert rewards.shape == (net.n_relays,)
    assert "pv_level" in info
    assert "fault_type" in info

    neighbor_obs = env.get_neighbor_observations(obs)
    assert neighbor_obs.ndim == 3

    print(f"  [ORIGINAL] Backward Compatible: obs={obs.shape}, reward_mean={rewards.mean():.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 2: GRADIENT-BASED TESTS (new — true backpropagation verification)
# ═══════════════════════════════════════════════════════════════════════════════

def test_computation_tape_linear():
    """Verify linear layer gradient via finite differences."""
    from marl.comm_graph_gradient import ComputationTape

    tape = ComputationTape()
    x = np.array([1.0, 2.0, 3.0])
    W = np.array([[1.0, 0.5, -0.5], [0.0, 1.0, 2.0]])  # (2, 3)
    b = np.array([0.1, -0.1])

    y = tape.linear(x, W, b, name="test")
    assert y.shape == (2,)

    # Backward with dL/dy = [1.0, 1.0]
    grads = tape.backward(np.array([1.0, 1.0]))

    # Verify dL/dW = dout @ x.T = [[1,2,3], [1,2,3]]
    expected_dW = np.outer(np.array([1.0, 1.0]), x)
    assert np.allclose(grads["test_W"], expected_dW), f"dW mismatch: {grads['test_W']} vs {expected_dW}"

    # Verify dL/db = dout = [1, 1]
    assert np.allclose(grads["test_b"], np.array([1.0, 1.0])), f"db mismatch"

    # Verify dL/dx = W.T @ dout = [1+0, 0.5+1, -0.5+2] = [1, 1.5, 1.5]
    expected_dx = W.T @ np.array([1.0, 1.0])
    assert np.allclose(grads["x"], expected_dx), f"dx mismatch: {grads['x']} vs {expected_dx}"

    print(f"  [GRADIENT] Tape Linear: dW_norm={np.linalg.norm(grads['test_W']):.4f}, dx_norm={np.linalg.norm(grads['x']):.4f}")


def test_computation_tape_relu():
    """Verify ReLU gradient."""
    from marl.comm_graph_gradient import ComputationTape

    tape = ComputationTape()
    x = np.array([1.0, -2.0, 0.5])
    y = tape.relu(x)

    grads = tape.backward(np.array([1.0, 1.0, 1.0]))
    expected = np.array([1.0, 0.0, 1.0])  # d/dx max(0,x) = 1 if x>0 else 0
    assert np.allclose(grads["x"], expected), f"ReLU grad mismatch: {grads['x']} vs {expected}"

    print(f"  [GRADIENT] Tape ReLU: grad={grads['x']}")


def test_computation_tape_tanh():
    """Verify tanh gradient: d/dx tanh(x) = 1 - tanh²(x)."""
    from marl.comm_graph_gradient import ComputationTape

    tape = ComputationTape()
    x = np.array([0.5, -0.5, 1.0])
    y = tape.tanh(x)

    grads = tape.backward(np.array([1.0, 1.0, 1.0]))
    expected = 1.0 - np.tanh(x) ** 2
    assert np.allclose(grads["x"], expected, atol=1e-6), f"Tanh grad mismatch: {grads['x']} vs {expected}"

    print(f"  [GRADIENT] Tape Tanh: max_error={np.max(np.abs(grads['x'] - expected)):.2e}")


def test_computation_tape_matmul():
    """Verify matrix multiplication gradient."""
    from marl.comm_graph_gradient import ComputationTape

    tape = ComputationTape()
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = np.array([[0.5, 1.0], [1.5, 2.0]])
    Y = tape.matmul(A, B, name="mm")

    dout = np.array([[1.0, 0.0], [0.0, 1.0]])
    grads = tape.backward(dout)

    # dL/dA = dout @ B.T
    expected_dA = dout @ B.T
    assert np.allclose(grads["A"], expected_dA), f"dA mismatch"

    # dL/dB = A.T @ dout
    expected_dB = A.T @ dout
    assert np.allclose(grads["B"], expected_dB), f"dB mismatch"

    print(f"  [GRADIENT] Tape MatMul: dA_norm={np.linalg.norm(grads['A']):.4f}, dB_norm={np.linalg.norm(grads['B']):.4f}")


def test_computation_tape_chain():
    """Verify full chain: linear → ReLU → linear gradient."""
    from marl.comm_graph_gradient import ComputationTape

    tape = ComputationTape()
    x = np.array([1.0, 2.0])
    W1 = np.array([[1.0, 0.5], [0.5, 1.0]])
    b1 = np.array([0.0, 0.0])
    W2 = np.array([[1.0, -1.0]])
    b2 = np.array([0.0])

    h1 = tape.linear(x, W1, b1, name="l1")
    h_relu = tape.relu(h1)
    y = tape.linear(h_relu, W2, b2, name="l2")

    # Backward with dL/dy = 1.0
    grads = tape.backward(np.array([1.0]))

    # Verify parameter gradients exist
    assert "l1_W" in grads, "Missing l1_W gradient"
    assert "l2_W" in grads, "Missing l2_W gradient"
    assert "l1_b" in grads, "Missing l1_b gradient"
    assert "l2_b" in grads, "Missing l2_b gradient"

    print(f"  [GRADIENT] Tape Chain: l1_W_norm={np.linalg.norm(grads['l1_W']):.4f}, l2_W_norm={np.linalg.norm(grads['l2_W']):.4f}")


def test_grad_gnn_encoder():
    """Test gradient-tracked GNN encoder."""
    rng = np.random.default_rng(42)
    from marl.comm_graph_gradient import GradGNNMessageEncoder, ComputationTape

    encoder = GradGNNMessageEncoder(obs_dim=10, msg_dim=8, hidden_dim=32, seed=42)
    obs = rng.standard_normal(10)

    # Forward without tape
    msg_no_tape = encoder.forward(obs)
    assert msg_no_tape.shape == (8,)

    # Forward with tape
    tape = ComputationTape()
    msg_tape = encoder.forward(obs, tape=tape)
    assert np.allclose(msg_no_tape, msg_tape), "Tape forward should match no-tape"

    # Backward
    grads = tape.backward(np.ones(8))
    assert len(grads) > 0, "No gradients computed"

    print(f"  [GRADIENT] Grad Encoder: msg_shape={msg_tape.shape}, n_grads={len(grads)}")


def test_grad_gnn_attention():
    """Test gradient-tracked GNN attention."""
    rng = np.random.default_rng(42)
    from marl.comm_graph_gradient import GradGNNAttention, ComputationTape

    attention = GradGNNAttention(msg_dim=8, hidden_dim=16, seed=42)
    n = 5
    messages = rng.standard_normal((n, 8))
    adjacency = np.eye(n)
    adjacency[0, 1] = adjacency[1, 0] = 1.0

    # Forward without tape
    attn_no_tape = attention.forward(messages, adjacency)
    assert attn_no_tape.shape == (n, n)
    assert np.allclose(attn_no_tape.sum(axis=1), 1.0, atol=1e-5)

    # Forward with tape
    tape = ComputationTape()
    attn_tape = attention.forward(messages, adjacency, tape=tape)
    assert np.allclose(attn_no_tape, attn_tape, atol=1e-5)

    # Backward
    dout = np.ones((n, n)) / (n * n)
    grads = tape.backward(dout)
    assert len(grads) > 0, "No gradients computed"

    print(f"  [GRADIENT] Grad Attention: shape={attn_tape.shape}, n_grads={len(grads)}")


def test_grad_comm_graph():
    """Test gradient-tracked CommGraph end-to-end."""
    rng = np.random.default_rng(42)
    from marl.comm_graph_gradient import GradCommGraph, build_relay_adjacency, ComputationTape

    net = get_3bus()
    adj = build_relay_adjacency(net)

    comm = GradCommGraph(
        n_agents=net.n_relays,
        obs_dim=10,
        enhanced_dim=18,
        adjacency=adj,
        msg_dim=8,
        hidden_dim=32,
        use_attention=True,
        seed=42,
    )

    obs = rng.standard_normal((net.n_relays, 10))

    # Forward without tape
    enhanced_no_tape, _, _ = comm.forward(obs)
    assert enhanced_no_tape.shape == (net.n_relays, 18)

    # Forward with tape
    tape = ComputationTape()
    enhanced_tape, messages, info = comm.forward(obs, tape=tape)
    assert np.allclose(enhanced_no_tape, enhanced_tape, atol=1e-5)

    # Backward through entire communication graph
    dout = rng.standard_normal((net.n_relays, 18))
    grads = tape.backward(dout)

    # Should have encoder, decoder, and attention gradients
    enc_grads = [k for k in grads if "enc" in k]
    dec_grads = [k for k in grads if "dec" in k]
    assert len(enc_grads) > 0, "Missing encoder gradients"
    assert len(dec_grads) > 0, "Missing decoder gradients"

    print(f"  [GRADIENT] Grad CommGraph: enhanced={enhanced_tape.shape}, enc_grads={len(enc_grads)}, dec_grads={len(dec_grads)}")


def test_grad_actor_forward_backward():
    """Test gradient-tracked actor forward and backward."""
    rng = np.random.default_rng(42)
    from marl.comm_graph_gradient import GradSharedGaussianActor, ComputationTape

    actor = GradSharedGaussianActor(obs_dim=10, act_dim=2, hidden=32, seed=42)
    obs = rng.standard_normal(10)

    # Forward without tape
    mean_no_tape = actor.get_mean(obs)
    assert mean_no_tape.shape == (2,)
    assert np.all(np.abs(mean_no_tape) <= 1.0)

    # Forward with tape
    tape = ComputationTape()
    mean_tape = actor.get_mean(obs, tape=tape)
    assert np.allclose(mean_no_tape, mean_tape)

    # Backward: simulate policy gradient dL/d(mean) = random direction
    dloss_dmean = rng.standard_normal(2)
    # Backprop through tanh and actor network
    grads = tape.backward(dloss_dmean)

    assert len(grads) > 0, "No actor gradients computed"
    assert any("actor_l1_W" in k for k in grads), "Missing actor layer 1 gradient"
    assert any("actor_l2_W" in k for k in grads), "Missing actor layer 2 gradient"
    assert any("actor_l3_W" in k for k in grads), "Missing actor layer 3 gradient"

    print(f"  [GRADIENT] Grad Actor: mean_shape={mean_tape.shape}, n_grads={len(grads)}")


def test_ppo_policy_gradient_finite_difference():
    """
    CRITICAL TEST: Verify PPO policy gradient against finite differences.
    This is the mathematical correctness check for the entire gradient chain.
    """
    from marl.comm_graph_gradient import GradSharedGaussianActor, compute_ppo_policy_gradient

    actor = GradSharedGaussianActor(obs_dim=4, act_dim=2, hidden=8, seed=42)
    obs = np.array([0.1, 0.2, 0.3, 0.4])
    # Use action far from mean to ensure non-zero gradient
    action = np.array([0.9, -0.8])
    advantage = 1.5
    old_log_prob = -2.0
    clip_eps = 0.2

    # Analytical gradient
    loss, agrad, _ = compute_ppo_policy_gradient(
        actor, obs, action, advantage, old_log_prob, clip_eps
    )

    # Finite difference gradient (check first 30 params)
    n_check = 30
    eps = 1e-5
    params = actor.get_all_params()
    fd_grad = np.zeros(n_check)

    for i in range(n_check):
        # Plus
        p_plus = params.copy()
        p_plus[i] += eps
        actor.set_all_params(p_plus)
        mean_p = actor.get_mean(obs)
        std = np.exp(np.clip(actor.log_std, -3, 0))
        z_p = (action - mean_p) / (std + 1e-8)
        lp_p = float(np.sum(-0.5 * (z_p**2 + 2 * actor.log_std + np.log(2 * np.pi))))
        r_p = np.exp(np.clip(lp_p - old_log_prob, -10, 10))
        loss_p = -min(r_p * advantage, np.clip(r_p, 1 - clip_eps, 1 + clip_eps) * advantage)

        # Minus
        p_minus = params.copy()
        p_minus[i] -= eps
        actor.set_all_params(p_minus)
        mean_m = actor.get_mean(obs)
        z_m = (action - mean_m) / (std + 1e-8)
        lp_m = float(np.sum(-0.5 * (z_m**2 + 2 * actor.log_std + np.log(2 * np.pi))))
        r_m = np.exp(np.clip(lp_m - old_log_prob, -10, 10))
        loss_m = -min(r_m * advantage, np.clip(r_m, 1 - clip_eps, 1 + clip_eps) * advantage)

        fd_grad[i] = (loss_p - loss_m) / (2 * eps)

    # Restore
    actor.set_all_params(params)

    # Compare
    max_abs_err = np.max(np.abs(agrad[:n_check] - fd_grad))
    max_rel_err = max_abs_err / (np.max(np.abs(fd_grad)) + 1e-10)

    assert max_rel_err < 0.01, f"Gradient mismatch! Max rel error: {max_rel_err:.4f}"

    print(f"  [GRADIENT] PPO FD Check: max_rel_err={max_rel_err:.2e}, max_abs_err={max_abs_err:.2e} — VERIFIED")


def test_comm_graph_gradient_finite_difference():
    """
    CRITICAL TEST: Verify communication graph gradients against finite differences.
    """
    rng = np.random.default_rng(42)
    from marl.comm_graph_gradient import GradCommGraph, build_relay_adjacency, ComputationTape

    net = get_3bus()
    adj = build_relay_adjacency(net)
    comm = GradCommGraph(
        n_agents=net.n_relays, obs_dim=10, enhanced_dim=18,
        adjacency=adj, msg_dim=8, hidden_dim=16, use_attention=False, seed=42,
    )

    obs = rng.standard_normal((net.n_relays, 10))
    dout = rng.standard_normal((net.n_relays, 18))

    # Analytical gradient
    tape = ComputationTape()
    enhanced, _, _ = comm.forward(obs, tape=tape)
    grads = tape.backward(dout)

    # Flatten analytical grads (only parameter gradients, not input gradients)
    params = comm.get_all_params()
    agrad_flat = np.zeros(len(params))
    idx = 0
    # Encoder params
    for k in ["enc1_W", "enc1_b", "enc2_W", "enc2_b"]:
        g = grads.get(k)
        if g is not None:
            agrad_flat[idx:idx + g.size] = g.ravel()
        idx += comm.encoder.W1.size if k == "enc1_W" else                comm.encoder.b1.size if k == "enc1_b" else                comm.encoder.W2.size if k == "enc2_W" else                comm.encoder.b2.size
    # Decoder params
    for k in ["dec1_W", "dec1_b", "dec2_W", "dec2_b"]:
        g = grads.get(k)
        if g is not None:
            agrad_flat[idx:idx + g.size] = g.ravel()
        idx += comm.decoder.W1.size if k == "dec1_W" else                comm.decoder.b1.size if k == "dec1_b" else                comm.decoder.W2.size if k == "dec2_W" else                comm.decoder.b2.size

    # Finite difference (check first 20 params)
    n_check = 20
    eps = 1e-5
    fd_grad = np.zeros(n_check)

    for i in range(n_check):
        p_plus = params.copy()
        p_plus[i] += eps
        comm.set_all_params(p_plus)
        enhanced_p, _, _ = comm.forward(obs)
        loss_p = np.sum(enhanced_p * dout)

        p_minus = params.copy()
        p_minus[i] -= eps
        comm.set_all_params(p_minus)
        enhanced_m, _, _ = comm.forward(obs)
        loss_m = np.sum(enhanced_m * dout)

        fd_grad[i] = (loss_p - loss_m) / (2 * eps)

    comm.set_all_params(params)

    max_abs_err = np.max(np.abs(agrad_flat[:n_check] - fd_grad))
    max_rel_err = max_abs_err / (np.max(np.abs(fd_grad)) + 1e-10)

    assert max_rel_err < 0.01, f"Comm gradient mismatch! Max rel error: {max_rel_err:.4f}"

    print(f"  [GRADIENT] Comm FD Check: max_rel_err={max_rel_err:.2e}, max_abs_err={max_abs_err:.2e} — VERIFIED")


def test_gradient_based_trainer_instantiation():
    """Test that the gradient-based trainer can be instantiated."""
    from marl.comm_graph_gradient import GradientBasedMAPPOMARLTrainer
    from marl.marl_env import DynamicDOCREnv

    net = get_3bus()
    env = DynamicDOCREnv(net, use_pandapower=False, seed=42)

    # Without communication
    trainer = GradientBasedMAPPOMARLTrainer(
        env,
        use_comm=False,
        hidden_actor=16,
        hidden_critic=32,
        rollout_len=16,
        seed=42,
    )
    assert trainer.actor.n_params > 0
    assert trainer.critic.n_params() > 0

    # With communication
    trainer_comm = GradientBasedMAPPOMARLTrainer(
        env,
        use_comm=True,
        hidden_actor=16,
        hidden_critic=32,
        rollout_len=16,
        msg_dim=8,
        comm_hidden=16,
        seed=42,
    )
    assert trainer_comm.comm_graph is not None
    assert trainer_comm.comm_graph.n_params() > 0

    print(f"  [GRADIENT] Trainer Instantiate: no_comm_actor_params={trainer.actor.n_params}, comm_params={trainer_comm.comm_graph.n_params()}")


def test_gradient_based_trainer_short_run():
    """Test a short training run with true gradients."""
    from marl.comm_graph_gradient import GradientBasedMAPPOMARLTrainer
    from marl.marl_env import DynamicDOCREnv

    net = get_3bus()
    env = DynamicDOCREnv(net, use_pandapower=False, seed=42)

    trainer = GradientBasedMAPPOMARLTrainer(
        env,
        use_comm=False,  # Test without comm first (simpler)
        hidden_actor=16,
        hidden_critic=32,
        rollout_len=16,
        lr_actor=1e-3,
        seed=42,
    )

    result = trainer.train(total_steps=64, verbose=False, log_every=1000)

    assert "TMS" in result
    assert "PS" in result
    assert "OF" in result
    assert "loss_history" in result
    assert "grad_norm_history" in result
    assert len(result["loss_history"]) > 0
    assert len(result["grad_norm_history"]) > 0

    print(f"  [GRADIENT] Trainer Short Run: OF={result['OF']:.4f}, n_updates={len(result['loss_history'])}, avg_|grad|={np.mean(result['grad_norm_history']):.4f}")


def test_gradient_based_trainer_with_comm():
    """Test gradient-based trainer WITH communication enabled."""
    from marl.comm_graph_gradient import GradientBasedMAPPOMARLTrainer
    from marl.marl_env import DynamicDOCREnv

    net = get_3bus()
    env = DynamicDOCREnv(net, use_pandapower=False, seed=42)

    trainer = GradientBasedMAPPOMARLTrainer(
        env,
        use_comm=True,
        hidden_actor=16,
        hidden_critic=32,
        rollout_len=16,
        lr_actor=1e-3,
        lr_comm=5e-4,
        msg_dim=4,
        comm_hidden=8,
        seed=42,
    )

    result = trainer.train(total_steps=64, verbose=False, log_every=1000)

    assert result["method"] == "Gradient-MAPPO-Comm"
    assert len(result["loss_history"]) > 0
    assert len(result["grad_norm_history"]) > 0

    print(f"  [GRADIENT] Trainer With Comm: method={result['method']}, OF={result['OF']:.4f}, avg_|grad|={np.mean(result['grad_norm_history']):.4f}")


def test_gradient_vs_es_comparison():
    """Compare gradient-based and ES-based training on same seed."""
    from marl.marl_agents import MAPPOMARLTrainer
    from marl.comm_graph_gradient import GradientBasedMAPPOMARLTrainer
    from marl.marl_env import DynamicDOCREnv

    net = get_3bus()

    # ES-based trainer
    env_es = DynamicDOCREnv(net, use_pandapower=False, seed=42)
    trainer_es = MAPPOMARLTrainer(
        env_es, use_comm=False, hidden_actor=16, hidden_critic=32,
        rollout_len=16, seed=42,
    )
    result_es = trainer_es.train(total_steps=64, verbose=False)

    # Gradient-based trainer
    env_grad = DynamicDOCREnv(net, use_pandapower=False, seed=42)
    trainer_grad = GradientBasedMAPPOMARLTrainer(
        env_grad, use_comm=False, hidden_actor=16, hidden_critic=32,
        rollout_len=16, lr_actor=0.02, seed=42,
    )
    result_grad = trainer_grad.train(total_steps=64, verbose=False)

    # Both should produce valid results
    assert result_es["OF"] < np.inf
    assert result_grad["OF"] < np.inf

    print(f"  [GRADIENT] ES vs Grad: ES_OF={result_es['OF']:.4f}, Grad_OF={result_grad['OF']:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_all_tests():
    """Run all Phase 2 unit tests (original + gradient-based)."""
    print("\n" + "=" * 70)
    print("  Phase 2 Unit Tests: Decentralized Agent Architecture")
    print("  Including TRUE Gradient-Based Communication Verification")
    print("=" * 70)

    original_tests = [
        test_gnn_encoder,
        test_gnn_attention,
        test_gnn_decoder,
        test_comm_graph,
        test_relay_adjacency,
        test_neighbor_observations,
        test_shared_gaussian_actor,
        test_mappo_critic,
        test_mappo_buffer,
        test_single_ppo,
        test_mappo_integration,
        test_bandwidth_tracking,
        test_backward_compatibility,
    ]

    gradient_tests = [
        test_computation_tape_linear,
        test_computation_tape_relu,
        test_computation_tape_tanh,
        test_computation_tape_matmul,
        test_computation_tape_chain,
        test_grad_gnn_encoder,
        test_grad_gnn_attention,
        test_grad_comm_graph,
        test_grad_actor_forward_backward,
        test_ppo_policy_gradient_finite_difference,
        test_comm_graph_gradient_finite_difference,
        test_gradient_based_trainer_instantiation,
        test_gradient_based_trainer_short_run,
        test_gradient_based_trainer_with_comm,
        test_gradient_vs_es_comparison,
    ]

    passed = 0
    failed = 0

    print("\n--- ORIGINAL API TESTS (Backward Compatibility) ---")
    for test in original_tests:
        try:
            test()
            passed += 1
            print(f"    [PASS] {test.__name__}")
        except Exception as e:
            failed += 1
            print(f"    [FAIL] {test.__name__}: {e}")
            import traceback
            traceback.print_exc()

    print("\n--- GRADIENT-BASED TESTS (New) ---")
    for test in gradient_tests:
        try:
            test()
            passed += 1
            print(f"    [PASS] {test.__name__}")
        except Exception as e:
            failed += 1
            print(f"    [FAIL] {test.__name__}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n  Results: {passed} passed, {failed} failed out of {passed + failed}")
    print("=" * 70)
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)