"""
tests/test_vectorized_marl_env.py
----------------------------------
Phase V6 of the GPU-vectorized-environment redesign:
VectorizedDynamicDOCREnv, the central integration class composing Phase V1
(VectorizedTopologyManager), Phase V-PF (VectorizedPowerFlowSolver), and
Phase V3's compute_operating_times_batch/objective_function_batch into one
batched environment.

Verification bar (see marl/vectorized_marl_env.py's module docstring and
elegant-rolling-map.md's Phase V6 section):
- n_envs=1 must match DynamicDOCREnv's existing output bit-for-bit for every
  observation feature EXCEPT v_norm (intentionally upgraded to a real
  power-flow voltage via V-PF instead of the old heuristic).
- n_envs>1 must show zero cross-environment leakage: each environment's
  trajectory in a batched call must exactly match running that same
  environment independently (its own seed/pv_level/TMS/PS).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from network.network_data import get_8bus, get_39bus
from network.relay_physics import TMS_MIN, TMS_MAX
from marl.marl_env import DynamicDOCREnv
from marl.vectorized_marl_env import VectorizedDynamicDOCREnv

V_NORM_FEATURE_IDX = 10


def _compare_n_envs_1(net_fn, seed=42, n_steps=5, pv_level=0.4, fault_type="3ph"):
    net1 = net_fn()
    ref = DynamicDOCREnv(net1, use_pandapower=False, seed=seed, fault_type=fault_type)
    obs_ref = ref.reset(seed=seed, pv_level=pv_level)

    net2 = net_fn()
    vec = VectorizedDynamicDOCREnv(net2, n_envs=1, seed=seed, fault_type=fault_type)
    obs_vec = vec.reset(seed=seed, pv_level=pv_level)

    def max_diff_excl_vnorm(o_ref, o_vec0):
        d = np.abs(o_ref - o_vec0)
        d[:, V_NORM_FEATURE_IDX] = 0.0
        return np.max(d)

    max_diff = max_diff_excl_vnorm(obs_ref, obs_vec[0])

    rng = np.random.default_rng(777)
    for _ in range(n_steps):
        actions = rng.uniform(-1, 1, (ref.n_agents, 2))
        o_ref, r_ref, done_ref, info_ref = ref.step(actions)
        o_vec, r_vec, done_vec, info_vec = vec.step(actions[None, :, :])
        max_diff = max(max_diff, max_diff_excl_vnorm(o_ref, o_vec[0]))
        assert done_ref == done_vec
        assert np.max(np.abs(r_ref - r_vec[0])) < 1e-6
        assert abs(info_ref["OF"] - info_vec["OF"][0]) < 1e-4

    return max_diff, obs_vec[0][:, V_NORM_FEATURE_IDX]


def test_vectorized_env_matches_single_env_8bus():
    max_diff, v_norm = _compare_n_envs_1(get_8bus)
    assert max_diff < 1e-9
    assert np.all((v_norm >= 0.0) & (v_norm <= 1.0))


def test_vectorized_env_matches_single_env_39bus():
    max_diff, v_norm = _compare_n_envs_1(get_39bus, pv_level=0.6)
    assert max_diff < 1e-9
    assert np.all((v_norm >= 0.0) & (v_norm <= 1.0))


def test_vectorized_env_no_cross_env_leakage():
    """n_envs=3, three different pv_levels -- each env slot's trajectory
    must exactly match an independently-run single-env instance fed that
    same slot's TMS/PS/pv_level."""
    n_envs = 3
    pv_levels = [0.1, 0.5, 0.9]
    fault_type = "3ph"

    net_multi = get_8bus()
    vec = VectorizedDynamicDOCREnv(net_multi, n_envs=n_envs, seed=0, fault_type=fault_type)
    vec.reset(seed=99, pv_level=np.array(pv_levels))
    n_agents = net_multi.n_relays

    # Confirm the batch's own TMS/PS draw matches a direct (n_envs, n_agents)
    # draw from a freshly-seeded RNG (same seed, same call order) -- this is
    # what "row e of the batch" means, not an independent (1, n_agents) draw.
    rng_ref = np.random.default_rng(99)
    TMS_expected = rng_ref.uniform(TMS_MIN + 0.02, TMS_MAX * 0.5, (n_envs, n_agents))
    PS_raw = rng_ref.uniform(0, 1, (n_envs, n_agents))
    ps_min, ps_max = net_multi.ps_bounds()
    ps_range = np.maximum(ps_max - ps_min, 1e-3)
    PS_expected = np.clip(PS_raw * ps_range[None, :] + ps_min[None, :],
                           ps_min[None, :], ps_max[None, :])
    assert np.max(np.abs(vec.TMS - TMS_expected)) < 1e-12
    assert np.max(np.abs(vec.PS - PS_expected)) < 1e-12

    single_envs = []
    for e in range(n_envs):
        s = VectorizedDynamicDOCREnv(get_8bus(), n_envs=1, seed=0, fault_type=fault_type)
        s.reset(pv_level=pv_levels[e])
        s.TMS = vec.TMS[e:e + 1].copy()
        s.PS = vec.PS[e:e + 1].copy()
        s._recompute_episode_cache()
        single_envs.append(s)

    obs_multi = vec._get_obs()
    max_obs_diff = 0.0
    for e in range(n_envs):
        obs_single = single_envs[e]._get_obs()
        max_obs_diff = max(max_obs_diff, np.max(np.abs(obs_multi[e] - obs_single[0])))

    actions_rng = np.random.default_rng(555)
    max_reward_diff = 0.0
    for _ in range(4):
        actions = actions_rng.uniform(-1, 1, (n_envs, n_agents, 2))
        o_multi, r_multi, done_multi, _ = vec.step(actions)
        for e in range(n_envs):
            o_single, r_single, done_single, _ = single_envs[e].step(actions[e:e + 1])
            max_obs_diff = max(max_obs_diff, np.max(np.abs(o_multi[e] - o_single[0])))
            max_reward_diff = max(max_reward_diff, np.max(np.abs(r_multi[e] - r_single[0])))
            assert done_multi == done_single

    assert max_obs_diff < 1e-9
    assert max_reward_diff < 1e-9


def test_vectorized_env_unsupported_system_raises():
    """Only systems with real per-branch impedance data (8/14/39-bus) are
    supported -- same restriction as VectorizedPowerFlowSolver, since V-PF
    is a mandatory component here."""
    from network.network_data import get_3bus
    try:
        VectorizedDynamicDOCREnv(get_3bus(), n_envs=2, seed=0)
        assert False, "expected ValueError for unsupported system"
    except ValueError:
        pass
