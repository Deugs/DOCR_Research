"""
tests/test_vectorized_marl_agents.py
--------------------------------------
Phase V9 of the GPU-vectorized-environment redesign: VectorizedMAPPOTrainer,
wiring Phases V1-V8 (VectorizedDynamicDOCREnv, SharedGaussianActor.sample_batch,
MAPPOCentralCritic.value_batch/update, VectorizedMAPPOBuffer/compute_gae)
together into one trainer stepping n_envs environments per rollout step.

Verification bar (see elegant-rolling-map.md's Phase V9 section and
marl/vectorized_marl_agents.py's module docstring): not bit-identical to
MAPPOMARLTrainer (Phase V7's RNG strategy makes that impossible by
construction, same precedent as the metaheuristics vectorization) -- the bar
here is a correctness/crash gate: training runs end-to-end without error at
n_envs=1 and n_envs>1, producing finite losses/rewards and the expected
shapes throughout, on CPU (this dev machine's torch build is CPU-only; real
CUDA execution is Phase V10's job on the target remote server).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from network.network_data import get_8bus
from marl.vectorized_marl_env import VectorizedDynamicDOCREnv
from marl.vectorized_marl_agents import VectorizedMAPPOTrainer


def _run_trainer(n_envs, total_steps=96, rollout_len=16):
    net = get_8bus()
    env = VectorizedDynamicDOCREnv(net, n_envs=n_envs, max_steps=50, seed=1)
    trainer = VectorizedMAPPOTrainer(env, hidden_actor=16, hidden_critic=32,
                                      rollout_len=rollout_len, n_epochs=2, seed=1, device="cpu")
    history = trainer.train(total_steps=total_steps, verbose=False)
    return trainer, history


def test_vectorized_trainer_runs_n_envs_1():
    trainer, history = _run_trainer(n_envs=1)
    assert len(history) > 0
    for h in history:
        assert np.isfinite(h["mean_reward"])
        assert np.isfinite(h["actor_loss"])
        assert np.isfinite(h["critic_loss"])


def test_vectorized_trainer_runs_n_envs_4_no_crash():
    trainer, history = _run_trainer(n_envs=4)
    assert len(history) > 0
    for h in history:
        assert np.isfinite(h["mean_reward"])
        assert np.isfinite(h["actor_loss"])
        assert np.isfinite(h["critic_loss"])

    # Shape sanity on the actor/critic after a real rollout+update cycle.
    assert trainer.buffer.n_envs == 4
    assert trainer.buffer.obs.shape[1:] == (4, trainer.n_agents, trainer.env.OBS_DIM)


def test_vectorized_trainer_neighbor_stats_batch_matches_per_env_loop():
    """_get_neighbor_stats_batch_envs (reduced per-environment) must match
    MAPPOMARLTrainer._get_neighbor_stats() run independently per env slot."""
    from marl.marl_agents import MAPPOMARLTrainer
    from marl.marl_env import DynamicDOCREnv

    rng = np.random.default_rng(3)
    n_envs, n_agents, obs_dim = 3, 5, 12
    obs = rng.standard_normal((n_envs, n_agents, obs_dim))

    batch_stats = VectorizedMAPPOTrainer._get_neighbor_stats_batch_envs(obs)

    net = get_8bus()
    ref_env = DynamicDOCREnv(net, use_pandapower=False, seed=0)
    ref_trainer = MAPPOMARLTrainer(ref_env, hidden_actor=8, hidden_critic=8, rollout_len=4, seed=0)
    for e in range(n_envs):
        single_stats = ref_trainer._get_neighbor_stats(obs[e])
        assert np.max(np.abs(batch_stats[e] - single_stats)) < 1e-12
