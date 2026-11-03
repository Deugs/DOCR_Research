"""
tests/test_phase2.py
--------------------
Unit and integration tests for Phase 2 components.

Tests:
1. GNN Message Encoder/Decoder
2. GNN Attention mechanism
3. CommGraph end-to-end communication
4. Relay adjacency matrix construction
5. Neighbor observations
6. Shared Gaussian Actor (parameter sharing)
7. MAPPO Centralized Critic
8. MAPPO Buffer with communication
9. Single-Agent PPO (centralized baseline)
10. MAPPO with GNN communication integration
11. Bandwidth usage tracking
12. Backward compatibility with Phase 1
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from network.network_data import get_3bus, get_8bus, get_30bus, get_39bus
from network.relay_physics import CTI, TMS_MIN, TMS_MAX


def test_gnn_encoder():
    """Test GNN message encoder."""
    rng = np.random.default_rng(42)
    from marl.comm_graph import GNNMessageEncoder, CommConfig

    config = CommConfig(message_dim=8, hidden_dim=32)
    encoder = GNNMessageEncoder(obs_dim=10, config=config, seed=42)

    # Single observation
    obs = rng.standard_normal(10)
    msg = encoder.encode(obs)
    assert msg.shape == (8,), f"Expected (8,), got {msg.shape}"
    assert np.all(np.abs(msg) <= 1.0), "Messages should be in [-1, 1] (tanh)"

    # Batch observations
    obs_batch = rng.standard_normal((5, 10))
    msgs = encoder.encode(obs_batch)
    assert msgs.shape == (5, 8), f"Expected (5, 8), got {msgs.shape}"

    # Parameter get/set
    params = encoder.get_params()
    encoder2 = GNNMessageEncoder(obs_dim=10, config=config, seed=0)
    encoder2.set_params(params)
    msg2 = encoder2.encode(obs)
    assert np.allclose(msg, msg2), "Parameter set/get failed"

    print(f"  GNN Encoder: msg_shape={msg.shape}, params={encoder.n_params}")


def test_gnn_attention():
    """Test GNN attention mechanism."""
    rng = np.random.default_rng(42)
    from marl.comm_graph import GNNAttention

    attention = GNNAttention(msg_dim=8, hidden_dim=16, seed=42)

    n = 5
    messages = rng.standard_normal((n, 8))
    adjacency = np.eye(n)
    adjacency[0, 1] = adjacency[1, 0] = 1.0
    adjacency[1, 2] = adjacency[2, 1] = 1.0

    attn_weights = attention.compute_attention(messages, adjacency)
    assert attn_weights.shape == (n, n), f"Expected ({n},{n}), got {attn_weights.shape}"

    # Check row sums = 1 (softmax)
    row_sums = attn_weights.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5), f"Attention rows don't sum to 1: {row_sums}"

    # Check zero attention for non-neighbors
    assert attn_weights[0, 3] < 1e-6, "Non-neighbor should have near-zero attention"

    print(f"  GNN Attention: shape={attn_weights.shape}, row_sums_ok={np.allclose(row_sums, 1.0)}")


def test_gnn_attention_batched():
    """2026-11-xx (Phase V4 of the GPU-vectorized-environment plan):
    GNNAttention.compute_attention_batched_t() -- the batched attention
    path for N parallel training environments, which (unlike the
    encoder/decoder, confirmed to already batch over any leading shape for
    free) needed a real rewrite: naively flattening (n_envs, n_agents,
    msg_dim) to (n_envs*n_agents, msg_dim) and calling the existing
    compute_attention_t() would compute attention scores ACROSS agents
    from different environments, silently mixing unrelated environments'
    graphs. Verifies both: (1) n_envs=1 matches compute_attention_t()
    exactly, and (2) n_envs>1 matches calling compute_attention_t()
    independently per environment -- the actual property that matters here
    (zero cross-environment leakage), not just "it runs"."""
    import torch
    from marl.comm_graph import GNNAttention

    torch.manual_seed(0)
    n_agents, msg_dim = 6, 8
    att = GNNAttention(msg_dim, hidden_dim=16, seed=3)
    adjacency = (torch.rand(n_agents, n_agents) > 0.4).float()
    adjacency.fill_diagonal_(1.0)

    messages = torch.randn(n_agents, msg_dim, dtype=torch.float32)
    single_out = att.compute_attention_t(messages, adjacency)
    batched_out = att.compute_attention_batched_t(messages.unsqueeze(0), adjacency)[0]
    assert (single_out - batched_out).abs().max().item() < 1e-10, "n_envs=1 should exactly match compute_attention_t()"

    n_envs = 4
    messages_batch = torch.randn(n_envs, n_agents, msg_dim, dtype=torch.float32)
    batched_out = att.compute_attention_batched_t(messages_batch, adjacency)
    assert batched_out.shape == (n_envs, n_agents, n_agents)
    assert torch.allclose(batched_out.sum(dim=2), torch.ones(n_envs, n_agents), atol=1e-5), (
        "each environment's attention rows must still sum to 1 (softmax)"
    )
    for e in range(n_envs):
        individual_out = att.compute_attention_t(messages_batch[e], adjacency)
        diff = (batched_out[e] - individual_out).abs().max().item()
        assert diff < 1e-10, f"env {e}: batched vs individual diff {diff:.2e} -- possible cross-environment leakage"

    print(f"  GNN Attention (batched): n_envs={n_envs} shape={batched_out.shape}, no cross-env leakage")


def test_gnn_decoder():
    """Test GNN message decoder."""
    rng = np.random.default_rng(42)
    from marl.comm_graph import GNNMessageDecoder, CommConfig

    config = CommConfig(hidden_dim=32)
    decoder = GNNMessageDecoder(obs_dim=10, msg_dim=8, enhanced_dim=18, config=config, seed=42)

    obs = rng.standard_normal(10)
    aggregated_msg = rng.standard_normal(8)
    enhanced = decoder.decode(obs, aggregated_msg)
    assert enhanced.shape == (18,), f"Expected (18,), got {enhanced.shape}"
    assert np.all(np.abs(enhanced) <= 1.0), "Enhanced obs should be in [-1, 1]"

    # Batch
    obs_batch = rng.standard_normal((5, 10))
    msg_batch = rng.standard_normal((5, 8))
    enhanced_batch = decoder.decode(obs_batch, msg_batch)
    assert enhanced_batch.shape == (5, 18)

    print(f"  GNN Decoder: enhanced_shape={enhanced.shape}, params={decoder.n_params}")


def test_comm_graph():
    """Test complete CommGraph end-to-end."""
    rng = np.random.default_rng(42)
    from marl.comm_graph import CommGraph, CommConfig, build_relay_adjacency

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

    # Bandwidth check
    bw = comm.get_bandwidth_usage(steps_per_second=1.0)
    assert bw["kbps"] < 10.0, f"Bandwidth too high: {bw['kbps']:.2f} kbps"
    print(f"  CommGraph: enhanced={enhanced_obs.shape}, msg={messages.shape}, bw={bw['kbps']:.3f}kbps")


def test_comm_graph_stress_mechanics():
    """Round-5 regression: experiments/training_pipeline.py's
    _eval_comm_stress set comm_graph.packet_loss_rate/latency_steps/
    quantization_bits via hasattr() guards, but CommGraph never defined
    those attributes — every guard was permanently False, so Fig 13's
    "communication stress" evaluation always ran the identical
    unperturbed rollout regardless of the requested stress level. Verify
    each knob now genuinely changes forward()'s output."""
    from marl.comm_graph import CommGraph, CommConfig, build_relay_adjacency

    net = get_8bus()
    adj = build_relay_adjacency(net)
    config = CommConfig(message_dim=8, hidden_dim=32, use_attention=True)

    def make_comm():
        return CommGraph(n_agents=net.n_relays, obs_dim=10, enhanced_dim=18,
                          adjacency=adj, config=config, seed=42)

    observations = np.random.default_rng(0).standard_normal((net.n_relays, 10))
    rng = np.random.default_rng(1)

    # Attributes must exist with sane defaults (no-op stress).
    comm = make_comm()
    assert comm.packet_loss_rate == 0.0
    assert comm.latency_steps == 0
    assert comm.quantization_bits == 32

    # packet_loss_rate=1.0 must zero every agent's message before
    # aggregation, changing the enhanced observation vs. no loss.
    comm_lossy = make_comm()
    comm_lossy.packet_loss_rate = 1.0
    enhanced_base, _, _ = make_comm().forward(observations, rng=np.random.default_rng(1))
    enhanced_lossy, _, _ = comm_lossy.forward(observations, rng=np.random.default_rng(1))
    assert not np.allclose(enhanced_base, enhanced_lossy), \
        "packet_loss_rate=1.0 had no effect on forward() output"

    # quantization_bits must change the aggregated output.
    comm_quant = make_comm()
    comm_quant.quantization_bits = 2
    enhanced_quant, _, _ = comm_quant.forward(observations, rng=np.random.default_rng(1))
    assert not np.allclose(enhanced_base, enhanced_quant), \
        "quantization_bits had no effect on forward() output"

    # latency_steps: first call(s) see zero history (no message received
    # yet), which must differ from the immediate (latency_steps=0) result.
    comm_latent = make_comm()
    comm_latent.latency_steps = 2
    enhanced_latent, _, _ = comm_latent.forward(observations, rng=np.random.default_rng(1))
    assert not np.allclose(enhanced_base, enhanced_latent), \
        "latency_steps had no effect on forward() output"

    print("  CommGraph stress mechanics: packet_loss/quantization/latency all alter forward()")


def test_comm_graph_top_k_sparsify():
    """Fig 22 (bandwidth vs. performance): exp_fig22_bandwidth_tradeoff.py
    used to try mutating a nonexistent `actor.gnn_depth` attribute to
    simulate compression -- a permanent no-op, and it crashed outright on
    a matmul dimension mismatch before ever reaching that code. CommGraph
    now has a real top_k_sparsify_fraction knob (this module's own
    docstring has long described "top-k sparsification (keep top 50% of
    alpha_ij)" as part of Fig 22's compression sweep). On a densely
    connected graph (more real neighbors per agent than k), restricting
    to the top-k highest-attention neighbors must change the aggregated
    message relative to using all neighbors.
    """
    from marl.comm_graph import CommGraph, CommConfig

    n_agents = 10
    # Fully connected (including self-loops) so every agent has n_agents
    # real neighbors -- k=5 (50%) is then a genuine restriction, unlike a
    # sparse graph where every agent already has <= k neighbors.
    adjacency = np.ones((n_agents, n_agents))
    config = CommConfig(message_dim=8, hidden_dim=16, use_attention=True)

    def make_comm():
        return CommGraph(n_agents=n_agents, obs_dim=10, enhanced_dim=18,
                          adjacency=adjacency, config=config, seed=42)

    observations = np.random.default_rng(0).standard_normal((n_agents, 10))

    comm = make_comm()
    assert comm.top_k_sparsify_fraction == 1.0

    enhanced_base, _, _ = make_comm().forward(observations, rng=np.random.default_rng(1))

    comm_sparse = make_comm()
    comm_sparse.top_k_sparsify_fraction = 0.5
    enhanced_sparse, _, _ = comm_sparse.forward(observations, rng=np.random.default_rng(1))

    assert not np.allclose(enhanced_base, enhanced_sparse), \
        "top_k_sparsify_fraction=0.5 had no effect on forward() output on a dense graph"

    print("  CommGraph top-k sparsification: restricting neighbors alters forward()")


def test_comm_graph_top_k_sparsify_uses_real_degree_not_n_agents():
    """Regression: the first implementation of top_k_sparsify_fraction
    computed k = ceil(n_agents * fraction) -- a fraction of the TOTAL
    agent count, not of each agent's actual number of real graph
    neighbors (its degree). Real relay-adjacency graphs in this codebase
    are sparse (e.g. avg degree ~5 out of 14 agents on the 8-bus system),
    so that k was always >= every row's real degree, making the knob a
    silent no-op on every real network -- exp_fig22_bandwidth_tradeoff.py's
    "top50" compression arm would report identical performance to
    uncompressed while claiming a 50% bandwidth cut. k must be computed
    relative to each row's own degree instead.
    """
    from marl.comm_graph import CommGraph, CommConfig, build_relay_adjacency
    from network.network_data import get_8bus

    net = get_8bus()
    adjacency = build_relay_adjacency(net)
    # Confirm this graph is genuinely sparse relative to n_agents, so a
    # fraction-of-n_agents k would trivially keep every real neighbor.
    degrees = adjacency.sum(axis=1)
    assert degrees.max() < net.n_relays, \
        "test fixture is not sparse -- rewrite with a sparser adjacency"

    config = CommConfig(message_dim=8, hidden_dim=16, use_attention=True)

    def make_comm():
        return CommGraph(n_agents=net.n_relays, obs_dim=10, enhanced_dim=18,
                          adjacency=adjacency, config=config, seed=42)

    observations = np.random.default_rng(0).standard_normal((net.n_relays, 10))

    enhanced_base, _, _ = make_comm().forward(observations, rng=np.random.default_rng(1))

    comm_sparse = make_comm()
    comm_sparse.top_k_sparsify_fraction = 0.5
    enhanced_sparse, _, _ = comm_sparse.forward(observations, rng=np.random.default_rng(1))

    assert not np.allclose(enhanced_base, enhanced_sparse), \
        "top_k_sparsify_fraction=0.5 had no effect on the real, sparse 8-bus relay graph"

    print("  CommGraph top-k sparsification: real per-node degree, not n_agents, drives k")


def test_comm_graph_forward_batched_t_matches_single():
    """Phase V11 of the GPU-vectorized-environment plan: CommGraph.
    forward_batched_t() (differentiable path, no degradation) must exactly
    match forward_t() at n_envs=1, and match n_envs independent forward_t()
    calls per environment at n_envs>1 (zero cross-environment leakage in
    the attention step) -- same verification bar Phase V4 established for
    compute_attention_batched_t alone, now extended to the full
    encode->attend->decode pipeline."""
    import torch
    from marl.comm_graph import CommGraph, CommConfig, build_relay_adjacency

    net = get_8bus()
    adj = build_relay_adjacency(net)
    config = CommConfig(message_dim=8, hidden_dim=32, use_attention=True)
    comm = CommGraph(n_agents=net.n_relays, obs_dim=10, enhanced_dim=18,
                      adjacency=adj, config=config, seed=7)

    rng = np.random.default_rng(0)
    obs_single = rng.standard_normal((net.n_relays, 10))
    obs_t = torch.as_tensor(obs_single, dtype=torch.float32)

    single_out = comm.forward_t(obs_t)
    batched_out = comm.forward_batched_t(obs_t.unsqueeze(0))[0]
    diff = (single_out - batched_out).abs().max().item()
    assert diff < 1e-10, f"n_envs=1 should exactly match forward_t(): diff={diff:.2e}"

    n_envs = 3
    obs_batch = rng.standard_normal((n_envs, net.n_relays, 10))
    obs_batch_t = torch.as_tensor(obs_batch, dtype=torch.float32)
    batched_out = comm.forward_batched_t(obs_batch_t)
    assert batched_out.shape == (n_envs, net.n_relays, 18)
    for e in range(n_envs):
        individual_out = comm.forward_t(obs_batch_t[e])
        diff = (batched_out[e] - individual_out).abs().max().item()
        # float32 tolerance (not 1e-10 like the n_envs=1 case above): torch.bmm
        # over a real (n_envs>1) batch can select a different BLAS kernel than
        # per-call 2D matmul, same float32 rounding-noise category this
        # project has accepted elsewhere (e.g. Phase V5's actor sample_batch,
        # ~6e-8) -- not evidence of cross-environment leakage, which would
        # show up as a difference many orders of magnitude larger.
        assert diff < 1e-5, f"env {e}: batched vs individual diff {diff:.2e} -- possible cross-environment leakage"

    print(f"  CommGraph.forward_batched_t: n_envs={n_envs}, exact match, no cross-env leakage")


def test_comm_graph_forward_batched_matches_single_n_envs_1():
    """forward_batched() (Phase V11, full degradation model -- dropout,
    quantization, packet loss, latency) must exactly match forward() at
    n_envs=1 across several sequential steps (latency_steps needs a
    multi-step history buffer to exercise meaningfully)."""
    from marl.comm_graph import CommGraph, CommConfig, build_relay_adjacency

    net = get_8bus()
    adj = build_relay_adjacency(net)
    config = CommConfig(message_dim=8, hidden_dim=32, use_attention=True, dropout=0.2)

    def make_comm():
        c = CommGraph(n_agents=net.n_relays, obs_dim=10, enhanced_dim=18,
                       adjacency=adj, config=config, seed=11)
        c.packet_loss_rate = 0.3
        c.latency_steps = 2
        c.quantization_bits = 4
        return c

    comm_single = make_comm()
    comm_batched = make_comm()
    rng_single = np.random.default_rng(5)
    rng_batched = np.random.default_rng(5)
    data_rng = np.random.default_rng(1)

    for step in range(4):
        obs = data_rng.standard_normal((net.n_relays, 10))
        enhanced_single, messages_single, info_single = comm_single.forward(obs, rng=rng_single)
        enhanced_batched, messages_batched, info_batched = comm_batched.forward_batched(
            obs[None, :, :], rng=rng_batched)

        assert np.abs(enhanced_single - enhanced_batched[0]).max() < 1e-10, f"step {step}: enhanced_obs mismatch"
        assert np.abs(messages_single - messages_batched[0]).max() < 1e-10, f"step {step}: messages mismatch"
        assert np.abs(info_single["attention"] - info_batched["attention"][0]).max() < 1e-10, \
            f"step {step}: attention mismatch"

    print("  CommGraph.forward_batched: n_envs=1 exactly matches forward() across 4 steps "
          "(dropout+packet_loss+quantization+latency all exercised)")


def test_comm_graph_forward_batched_no_cross_env_leakage():
    """forward_batched()'s batch-reduction ops (quantization's per-env max,
    packet-loss's per-env draw, latency's per-env history slot) must never
    mix data across environments. Verified by comparing two batched runs
    that are IDENTICAL except for one environment's observations: with the
    rng reseeded identically before each run (so every random mask is
    drawn identically regardless of observation content), every OTHER
    environment's output must stay bit-for-bit identical despite env 1's
    input changing."""
    from marl.comm_graph import CommGraph, CommConfig, build_relay_adjacency

    net = get_8bus()
    adj = build_relay_adjacency(net)
    config = CommConfig(message_dim=8, hidden_dim=32, use_attention=True, dropout=0.2)
    n_envs = 3

    def make_comm():
        c = CommGraph(n_agents=net.n_relays, obs_dim=10, enhanced_dim=18,
                       adjacency=adj, config=config, seed=11)
        c.packet_loss_rate = 0.3
        c.latency_steps = 2
        c.quantization_bits = 4
        return c

    data_rng = np.random.default_rng(2)
    obs_steps = [data_rng.standard_normal((n_envs, net.n_relays, 10)) for _ in range(4)]
    obs_steps_perturbed = [o.copy() for o in obs_steps]
    for o in obs_steps_perturbed:
        o[1] += 50.0  # large perturbation to env 1 only

    comm_a = make_comm()
    comm_b = make_comm()
    rng_a = np.random.default_rng(9)
    rng_b = np.random.default_rng(9)

    for step in range(4):
        enhanced_a, _, _ = comm_a.forward_batched(obs_steps[step], rng=rng_a)
        enhanced_b, _, _ = comm_b.forward_batched(obs_steps_perturbed[step], rng=rng_b)
        for e in (0, 2):
            diff = np.abs(enhanced_a[e] - enhanced_b[e]).max()
            assert diff < 1e-10, (
                f"step {step}, env {e}: output changed when only env 1's observations "
                f"were perturbed -- cross-environment leakage (diff={diff:.2e})"
            )
        assert not np.allclose(enhanced_a[1], enhanced_b[1]), \
            f"step {step}: env 1's own output should differ after its observations were perturbed"

    print("  CommGraph.forward_batched: perturbing one environment's observations leaves "
          "every other environment's output unchanged (no cross-env leakage)")


def test_comm_graph_update_batched_matches_single_n_envs_1():
    """update_batched() (Phase V11) must exactly match update() at n_envs=1
    -- same PPO-clip loss formula and gradient step on the GNN, verified
    via the resulting loss value and the GNN's post-update parameters."""
    from marl.comm_graph import CommGraph, CommConfig, build_relay_adjacency
    from marl.marl_agents import SharedGaussianActor

    net = get_8bus()
    adj = build_relay_adjacency(net)
    config = CommConfig(message_dim=8, hidden_dim=16, use_attention=True)
    n_agents = net.n_relays
    T = 3

    def make_comm():
        return CommGraph(n_agents=n_agents, obs_dim=10, enhanced_dim=18,
                          adjacency=adj, config=config, seed=13, lr=1e-3)

    comm_single = make_comm()
    comm_batched = make_comm()
    actor_single = SharedGaussianActor(18, 2, hidden=16, seed=4)
    actor_batched = SharedGaussianActor(18, 2, hidden=16, seed=4)

    rng = np.random.default_rng(3)
    raw_obs = rng.standard_normal((T, n_agents, 10))
    act = rng.uniform(-1, 1, (T, n_agents, 2))
    adv = rng.standard_normal((T, n_agents))
    old_lp = rng.standard_normal((T, n_agents))

    result_single = comm_single.update(raw_obs, act, adv, old_lp, actor_single, clip_eps=0.2)
    result_batched = comm_batched.update_batched(
        raw_obs[:, None, :, :], act[:, None, :, :], adv[:, None, :], old_lp[:, None, :],
        actor_batched, clip_eps=0.2,
    )

    assert abs(result_single["loss"] - result_batched["loss"]) < 1e-6, (
        f"loss mismatch: single={result_single['loss']:.6f} batched={result_batched['loss']:.6f}"
    )
    params_single = comm_single.get_all_params()
    params_batched = comm_batched.get_all_params()
    assert np.abs(params_single - params_batched).max() < 1e-6, \
        "post-update GNN parameters diverge between update() and update_batched() at n_envs=1"

    print(f"  CommGraph.update_batched: n_envs=1 exactly matches update() "
          f"(loss={result_batched['loss']:.4f})")


def test_relay_adjacency():
    """Test relay adjacency matrix for different systems."""
    from marl.comm_graph import build_relay_adjacency

    for get_net in [get_3bus, get_8bus, get_30bus, get_39bus]:
        net = get_net()
        adj = build_relay_adjacency(net)

        assert adj.shape == (net.n_relays, net.n_relays)
        assert np.all(np.diag(adj) == 1.0), f"{net.name}: Self-loops missing"
        assert np.all(adj == adj.T), f"{net.name}: Not symmetric"

        # Check that relays in pairs are connected
        for pair in net.pairs:
            assert adj[pair.primary, pair.backup] == 1.0, f"{net.name}: Pair not connected"
            assert adj[pair.backup, pair.primary] == 1.0, f"{net.name}: Pair not symmetric"

        # Count edges
        n_edges = int(np.sum(adj) - net.n_relays) // 2
        print(f"  {net.name}: {net.n_relays} relays, {n_edges} unique edges")


def test_neighbor_observations():
    """Test neighbor observation extraction."""
    from marl.marl_env import DynamicDOCREnv

    net = get_8bus()
    env = DynamicDOCREnv(net, use_pandapower=False, seed=42)
    obs = env.reset(seed=42)

    neighbor_obs = env.get_neighbor_observations(obs)
    assert neighbor_obs.ndim == 3, f"Expected 3D, got {neighbor_obs.ndim}D"
    assert neighbor_obs.shape[0] == net.n_relays

    # Check that neighbors are populated
    for i in range(net.n_relays):
        if env.neighbors[i]:
            assert not np.all(neighbor_obs[i, 0] == 0), f"Agent {i} should have non-zero neighbor"

    print(f"  Neighbor Obs: shape={neighbor_obs.shape}, max_neighbors={neighbor_obs.shape[1]}")


def test_shared_gaussian_actor():
    """Test shared Gaussian actor with parameter sharing."""
    rng = np.random.default_rng(42)
    from marl.marl_agents import SharedGaussianActor

    actor = SharedGaussianActor(obs_dim=10, act_dim=2, hidden=32, seed=42)

    # Single observation
    obs = rng.standard_normal(10)
    action, log_prob = actor.sample(obs, np.random.default_rng(42))
    assert action.shape == (2,)
    assert np.all(np.abs(action) <= 1.0)
    assert log_prob < 0  # Log prob of continuous distribution

    # Batch observation
    obs_batch = rng.standard_normal((5, 10))
    means = actor.get_mean(obs_batch)
    assert means.shape == (5, 2)

    # Parameter sharing check
    params = actor.get_all_params()
    actor2 = SharedGaussianActor(obs_dim=10, act_dim=2, hidden=32, seed=0)
    actor2.set_all_params(params)
    action2, _ = actor2.sample(obs, np.random.default_rng(42))
    assert np.allclose(action, action2), "Parameter sharing failed"

    print(f"  Shared Actor: action_shape={action.shape}, params={len(params)}")


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

    # Parameter get/set
    params = critic.get_params()
    critic2 = MAPPOCentralCritic(n_relays, hidden=64, seed=0)
    critic2.set_params(params)
    value2 = critic2.value(TMS, PS, pv_level=0.5, topology_flag=1.0,
                             fault_type_idx=0, neighbor_stats=neighbor_stats)
    assert np.isclose(value, value2), "Critic parameter set/get failed"

    print(f"  MAPPO Critic: value={value:.4f}, params={critic.n_params()}")


def test_fault_type_idx_maps_env_fault_type_not_hardcoded_zero():
    """Regression: every real call site in MAPPOMARLTrainer used to pass
    the literal `fault_type_idx=0` to the critic regardless of the env's
    actual fault_type -- MAPPOCentralCritic's global state reserves 4
    dims for a one-hot over FAULT_TYPES, but that conditioning was
    functionally inert whenever training used a non-default fault type
    (e.g. via a real --fault-type CLI flag). _fault_type_idx() must map
    each real fault type string to its correct index, and a trainer
    built on a non-"3ph" env must actually use that index.
    """
    from marl.marl_agents import _fault_type_idx, MAPPOMARLTrainer
    from network.relay_physics import FAULT_TYPES
    from network.network_data import get_8bus
    from marl.marl_env import DynamicDOCREnv

    for expected_idx, ft in enumerate(FAULT_TYPES):
        assert _fault_type_idx(ft) == expected_idx
    assert _fault_type_idx("not-a-real-fault-type") == 0

    net = get_8bus()
    env = DynamicDOCREnv(net, seed=1, use_pandapower=False, fault_type="LL", curriculum_stage=3)
    trainer = MAPPOMARLTrainer(env, use_comm=False, seed=1)
    assert trainer.env.fault_type == "LL"
    assert _fault_type_idx(trainer.env.fault_type) == FAULT_TYPES.index("LL") != 0

    print("  fault_type_idx: maps real env.fault_type, not hardcoded to 0")


def test_mappo_buffer():
    """Test MAPPO rollout buffer with communication."""
    rng = np.random.default_rng(42)
    from marl.marl_agents import MAPPOBuffer

    n_agents, obs_dim, act_dim = 8, 10, 2
    capacity = 64

    # Without communication
    buffer = MAPPOBuffer(n_agents, obs_dim, act_dim, capacity, use_comm=False)
    obs = rng.standard_normal((n_agents, obs_dim))
    actions = rng.standard_normal((n_agents, act_dim))
    rewards = rng.standard_normal(n_agents)
    log_probs = rng.standard_normal(n_agents)
    value = 0.5
    done = False

    buffer.add(obs, actions, rewards, log_probs, value, done)
    assert buffer.size == 1

    # With communication
    buffer_comm = MAPPOBuffer(n_agents, obs_dim, act_dim, capacity, use_comm=True, msg_dim=8)
    messages = rng.standard_normal((n_agents, 8))
    attention = rng.random((n_agents, n_agents))
    buffer_comm.add(obs, actions, rewards, log_probs, value, done,
                    messages=messages, attention=attention)
    assert buffer_comm.size == 1
    assert buffer_comm.messages is not None

    # GAE computation
    for _ in range(10):
        buffer.add(obs, actions, rewards, log_probs, value, done)
    adv, ret = buffer.compute_gae(gamma=0.99, lam=0.95, last_value=0.0)
    assert adv.shape == (11, n_agents)
    assert ret.shape == (11, n_agents)

    print(f"  MAPPO Buffer: size={buffer.size}, adv_shape={adv.shape}")


def test_actor_sample_batch():
    """
    Phase V5 (GPU-vectorized-environment redesign): SharedGaussianActor.sample_batch()
    must (a) reproduce sample()'s exact math given matched noise, and (b) be
    statistically equivalent to sample() over many independent draws (RNG
    source differs -- torch.Generator vs. np.random.Generator -- so this is a
    distributional check, not bit-identical, same bar as this project's other
    RNG-order-changing vectorizations).
    """
    import torch
    import unittest.mock
    from marl.marl_agents import SharedGaussianActor

    obs_dim, act_dim = 14, 2
    actor = SharedGaussianActor(obs_dim, act_dim, seed=5, device="cpu")
    rng = np.random.default_rng(3)
    N = 10
    obs_batch = rng.standard_normal((N, obs_dim)).astype(np.float32)
    noise_arr = rng.standard_normal((N, act_dim)).astype(np.float32)

    mean_np = actor.get_mean(obs_batch)
    log_std_np = actor.log_std.detach().numpy()
    std_np = np.exp(np.clip(log_std_np, -3, 0))
    action_ref = np.clip(mean_np + std_np * noise_arr, -1.0, 1.0)
    log_prob_ref = -0.5 * np.sum(noise_arr**2 + 2 * log_std_np + np.log(2 * np.pi), axis=1)

    with unittest.mock.patch("torch.randn", return_value=torch.as_tensor(noise_arr)):
        action_new, log_prob_new = actor.sample_batch(obs_batch)

    assert np.max(np.abs(action_ref - action_new)) < 1e-5
    assert np.max(np.abs(log_prob_ref - log_prob_new)) < 1e-5

    # Statistical equivalence over many draws at a fixed observation.
    N2 = 3000
    obs_fixed = rng.standard_normal(obs_dim).astype(np.float32)
    obs_tile = np.tile(obs_fixed, (N2, 1))

    rng2 = np.random.default_rng(99)
    old_actions = np.array([actor.sample(obs_fixed, rng2)[0] for _ in range(N2)])
    old_lps = np.array([actor.sample(obs_fixed, rng2)[1] for _ in range(N2)])

    gen = torch.Generator().manual_seed(123)
    new_actions, new_lps = actor.sample_batch(obs_tile, generator=gen)

    assert abs(old_actions.mean() - new_actions.mean()) < 0.05
    assert abs(old_actions.std() - new_actions.std()) < 0.05
    assert abs(old_lps.mean() - new_lps.mean()) < 0.3

    print(f"  sample_batch: action mean old={old_actions.mean():.4f} new={new_actions.mean():.4f}")


def test_vectorized_mappo_buffer():
    """
    Phase V5: VectorizedMAPPOBuffer (new, parallel class -- MAPPOBuffer stays
    untouched) must reproduce MAPPOBuffer.compute_gae() exactly at n_envs=1,
    and must show zero cross-environment leakage at n_envs>1 (each env's GAE
    output must match an independently-run MAPPOBuffer fed the same per-env
    trajectory).
    """
    from marl.marl_agents import MAPPOBuffer, VectorizedMAPPOBuffer

    rng = np.random.default_rng(7)
    n_agents, obs_dim, act_dim, T = 4, 10, 2, 20

    buf_old = MAPPOBuffer(n_agents, obs_dim, act_dim, capacity=T)
    buf_new = VectorizedMAPPOBuffer(1, n_agents, obs_dim, act_dim, capacity=T, device="cpu")
    dones_seq = (rng.random(T) < 0.15)
    dones_seq[-1] = False
    for t in range(T):
        obs = rng.standard_normal((n_agents, obs_dim))
        actions = rng.standard_normal((n_agents, act_dim))
        rewards = rng.standard_normal(n_agents)
        log_probs = rng.standard_normal(n_agents)
        value = float(rng.standard_normal())
        done = bool(dones_seq[t])
        tms = rng.standard_normal(n_agents)
        ps = rng.standard_normal(n_agents)
        buf_old.add(obs, actions, rewards, log_probs, value, done, tms=tms, ps=ps)
        buf_new.add(obs[None], actions[None], rewards[None], log_probs[None],
                    np.array([value]), np.array([done]), tms=tms[None], ps=ps[None])

    adv_old, ret_old = buf_old.compute_gae(last_value=0.3)
    adv_new, ret_new = buf_new.compute_gae(last_value=0.3)
    assert np.max(np.abs(adv_old - adv_new[:, 0, :].numpy())) < 1e-9
    assert np.max(np.abs(ret_old - ret_new[:, 0, :].numpy())) < 1e-9

    # n_envs=3: no cross-environment leakage.
    n_envs = 3
    buf_multi = VectorizedMAPPOBuffer(n_envs, n_agents, obs_dim, act_dim, capacity=T, device="cpu")
    per_env_bufs = [MAPPOBuffer(n_agents, obs_dim, act_dim, capacity=T) for _ in range(n_envs)]
    per_env_dones = [(rng.random(T) < 0.2) for _ in range(n_envs)]
    for e in range(n_envs):
        per_env_dones[e][-1] = False
    for t in range(T):
        obs_t = rng.standard_normal((n_envs, n_agents, obs_dim))
        act_t = rng.standard_normal((n_envs, n_agents, act_dim))
        rew_t = rng.standard_normal((n_envs, n_agents))
        lp_t = rng.standard_normal((n_envs, n_agents))
        val_t = rng.standard_normal(n_envs)
        tms_t = rng.standard_normal((n_envs, n_agents))
        ps_t = rng.standard_normal((n_envs, n_agents))
        done_t = np.array([per_env_dones[e][t] for e in range(n_envs)])
        buf_multi.add(obs_t, act_t, rew_t, lp_t, val_t, done_t, tms=tms_t, ps=ps_t)
        for e in range(n_envs):
            per_env_bufs[e].add(obs_t[e], act_t[e], rew_t[e], lp_t[e], val_t[e], bool(done_t[e]),
                                 tms=tms_t[e], ps=ps_t[e])

    adv_multi, ret_multi = buf_multi.compute_gae(last_value=0.0)
    for e in range(n_envs):
        adv_e, ret_e = per_env_bufs[e].compute_gae(last_value=0.0)
        assert np.max(np.abs(adv_e - adv_multi[:, e, :].numpy())) < 1e-9
        assert np.max(np.abs(ret_e - ret_multi[:, e, :].numpy())) < 1e-9

    print(f"  VectorizedMAPPOBuffer: n_envs=1 exact match, n_envs={n_envs} no cross-env leakage")


def test_single_ppo():
    """Test Single-Agent PPO (centralized baseline)."""
    from marl.single_agent_ppo import CentralizedPPOEnvironment, SinglePPOTrainer

    net = get_3bus()
    env = CentralizedPPOEnvironment(net, use_pandapower=False, seed=42)

    obs = env.reset()
    assert obs.shape == (env.OBS_DIM,)
    assert len(env.TMS) == net.n_relays
    assert len(env.PS) == net.n_relays

    # Test action decoding
    trainer = SinglePPOTrainer(env, hidden_actor=32, hidden_critic=32, seed=42)
    action_norm, _ = trainer.actor.sample(obs, np.random.default_rng(42))
    TMS, PS = trainer.actor.decode_action(action_norm, env.ps_min, env.ps_max)
    assert len(TMS) == net.n_relays
    assert len(PS) == net.n_relays
    assert np.all(TMS >= TMS_MIN) and np.all(TMS <= TMS_MAX)
    assert np.all(PS >= env.ps_min) and np.all(PS <= env.ps_max)

    print(f"  Single PPO: obs={obs.shape}, action={action_norm.shape}, TMS_range=[{TMS.min():.3f}, {TMS.max():.3f}]")


def test_mappo_integration():
    """Test MAPPO integration with environment."""
    from marl.marl_env import DynamicDOCREnv
    from marl.marl_agents import MAPPOMARLTrainer

    net = get_3bus()
    env = DynamicDOCREnv(net, use_pandapower=False, seed=42)

    # Without communication
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

    # Quick training run
    result = trainer.train(total_steps=64, verbose=False, log_every=1000)
    assert "TMS" in result
    assert "PS" in result
    assert "OF" in result
    assert result["method"] == "MAPPO"

    print(f"  MAPPO Integration: OF={result['OF']:.4f}, method={result['method']}")


def test_bandwidth_tracking():
    """Test communication bandwidth tracking."""
    rng = np.random.default_rng(42)
    from marl.comm_graph import CommGraph, CommConfig, build_relay_adjacency

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

    # Simulate multiple steps
    for _ in range(10):
        obs = rng.standard_normal((net.n_relays, 10))
        comm.forward(obs)

    bw = comm.get_bandwidth_usage(steps_per_second=1.0)
    assert bw["within_limit"], f"Bandwidth {bw['kbps']:.2f}kbps exceeds limit {bw['limit_kbps']:.2f}kbps"
    assert bw["bytes_per_step"] == net.n_relays * 8 * 4  # n_agents * msg_dim * 4 bytes

    print(f"  Bandwidth: {bw['kbps']:.3f}kbps (limit: {bw['limit_kbps']:.0f}kbps), within_limit={bw['within_limit']}")


def test_backward_compatibility():
    """Test Phase 2 is backward compatible with Phase 1."""
    from marl.marl_env import DynamicDOCREnv
    from network.network_data import get_3bus

    net = get_3bus()
    env = DynamicDOCREnv(net, use_pandapower=False, seed=42)

    obs = env.reset(seed=42)
    assert obs.shape == (net.n_relays, env.OBS_DIM)
    assert env.OBS_DIM == 12  # Objective 2: 10 -> 12 (added real V, P_dir)

    actions = np.zeros((net.n_relays, env.ACT_DIM))
    next_obs, rewards, done, info = env.step(actions)
    assert next_obs.shape == obs.shape
    assert rewards.shape == (net.n_relays,)
    assert "pv_level" in info
    assert "fault_type" in info

    # Check neighbor functionality is available
    neighbor_obs = env.get_neighbor_observations(obs)
    assert neighbor_obs.ndim == 3

    print(f"  Backward Compatible: obs={obs.shape}, reward_mean={rewards.mean():.4f}")


def run_all_tests():
    """Run all Phase 2 unit tests."""
    print("\n" + "=" * 60)
    print("  Phase 2 Unit Tests: Decentralized Agent Architecture")
    print("=" * 60)

    tests = [
        test_gnn_encoder,
        test_gnn_attention,
        test_gnn_decoder,
        test_comm_graph,
        test_comm_graph_stress_mechanics,
        test_comm_graph_top_k_sparsify,
        test_comm_graph_top_k_sparsify_uses_real_degree_not_n_agents,
        test_relay_adjacency,
        test_neighbor_observations,
        test_shared_gaussian_actor,
        test_mappo_critic,
        test_fault_type_idx_maps_env_fault_type_not_hardcoded_zero,
        test_mappo_buffer,
        test_single_ppo,
        test_mappo_integration,
        test_bandwidth_tracking,
        test_backward_compatibility,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
            print(f"  [PASS] {test.__name__}")
        except Exception as e:
            failed += 1
            print(f"  [FAIL] {test.__name__}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n  Results: {passed} passed, {failed} failed")
    print("=" * 60)
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
