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
    trainer, result = _run_trainer(n_envs=1)
    n_rollouts = len(result["history"])
    assert n_rollouts > 0
    assert len(result["cti_satisfaction_history"]) == n_rollouts
    assert len(result["actor_loss_history"]) == n_rollouts
    assert len(result["critic_loss_history"]) == n_rollouts
    assert len(result["mean_reward_history"]) == n_rollouts
    assert np.isfinite(result["OF"])
    for of_val, cti_rate, actor_l, critic_l, rew in zip(
        result["history"], result["cti_satisfaction_history"],
        result["actor_loss_history"], result["critic_loss_history"],
        result["mean_reward_history"],
    ):
        assert np.isfinite(of_val)
        assert 0.0 <= cti_rate <= 1.0
        assert np.isfinite(actor_l)
        assert np.isfinite(critic_l)
        assert np.isfinite(rew)


def test_vectorized_trainer_runs_n_envs_4_no_crash():
    trainer, result = _run_trainer(n_envs=4)
    n_rollouts = len(result["history"])
    assert n_rollouts > 0
    for of_val, cti_rate, actor_l, critic_l, rew in zip(
        result["history"], result["cti_satisfaction_history"],
        result["actor_loss_history"], result["critic_loss_history"],
        result["mean_reward_history"],
    ):
        assert np.isfinite(of_val)
        assert 0.0 <= cti_rate <= 1.0
        assert np.isfinite(actor_l)
        assert np.isfinite(critic_l)
        assert np.isfinite(rew)
    assert result["TMS"].shape == (trainer.n_agents,)
    assert result["PS"].shape == (trainer.n_agents,)

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


def _run_trainer_comm(n_envs, total_steps=128, rollout_len=16):
    net = get_8bus()
    env = VectorizedDynamicDOCREnv(net, n_envs=n_envs, max_steps=50, seed=1)
    trainer = VectorizedMAPPOTrainer(env, use_comm=True, hidden_actor=16, hidden_critic=32,
                                      rollout_len=rollout_len, n_epochs=2, seed=1, device="cpu")
    history = trainer.train(total_steps=total_steps, verbose=False)
    return trainer, history


def test_vectorized_trainer_use_comm_runs_n_envs_1():
    """Phase V11: use_comm=True wired into VectorizedMAPPOTrainer -- same
    correctness/crash gate test_vectorized_trainer_runs_n_envs_1 applies to
    the no-comm path, now with CommGraph.forward_batched()/update_batched()
    in the loop."""
    trainer, result = _run_trainer_comm(n_envs=1)
    assert trainer.comm_graph is not None
    n_rollouts = len(result["history"])
    assert n_rollouts > 0
    for of_val, cti_rate, actor_l, critic_l, rew in zip(
        result["history"], result["cti_satisfaction_history"],
        result["actor_loss_history"], result["critic_loss_history"],
        result["mean_reward_history"],
    ):
        assert np.isfinite(of_val)
        assert 0.0 <= cti_rate <= 1.0
        assert np.isfinite(actor_l)
        assert np.isfinite(critic_l)
        assert np.isfinite(rew)
    assert result["use_comm"] is True
    assert len(result["comm_bandwidth_history"]) > 0


def test_vectorized_trainer_use_comm_runs_n_envs_4_and_gnn_trains():
    """n_envs=4 crash gate, plus a real correctness check update_batched()
    is actually training the GNN rather than leaving it at its random
    initialization (the same risk MAPPOMARLTrainer's own module docstring
    calls out for update())."""
    net = get_8bus()
    env = VectorizedDynamicDOCREnv(net, n_envs=4, max_steps=50, seed=1)
    trainer = VectorizedMAPPOTrainer(env, use_comm=True, hidden_actor=16, hidden_critic=32,
                                      rollout_len=16, n_epochs=2, seed=1, device="cpu")
    initial_comm_params = trainer.comm_graph.get_all_params().copy()

    result = trainer.train(total_steps=128, verbose=False)

    for of_val, cti_rate, actor_l, critic_l, rew in zip(
        result["history"], result["cti_satisfaction_history"],
        result["actor_loss_history"], result["critic_loss_history"],
        result["mean_reward_history"],
    ):
        assert np.isfinite(of_val)
        assert 0.0 <= cti_rate <= 1.0
        assert np.isfinite(actor_l)
        assert np.isfinite(critic_l)
        assert np.isfinite(rew)

    final_comm_params = trainer.comm_graph.get_all_params()
    assert not np.allclose(initial_comm_params, final_comm_params), \
        "GNN comm parameters never moved from initialization -- update_batched() isn't training it"
    assert trainer.buffer.use_comm
    assert trainer.buffer.obs.shape[-1] == trainer.comm_graph.decoder.enhanced_dim
    assert len(trainer.attention_snapshots) >= 0  # populated opportunistically, never required


def test_vectorized_trainer_safe_rl_integration():
    """Phase V12: VectorizedSafeRLWrapper duck-types the same surface
    VectorizedMAPPOTrainer expects from `env` (step/reset/get_state/OBS_DIM/
    ACT_DIM/net/n_envs/current_pv_level/topology/fault_type/TMS/_If_pri_dyn/
    _If_bkp_dyn, all forwarded via __getattr__), so
    VectorizedMAPPOTrainer(VectorizedSafeRLWrapper(vec_env), ...) must train
    end-to-end with zero trainer-side changes, mirroring
    test_safe_rl_mappo_integration()'s single-env composition pattern. Also
    confirms the shared Lagrangian's lambdas actually move during training
    (not frozen at init -- the same real risk this project's own code
    comments flag for the single-env wrapper)."""
    from marl.safe_rl import create_vectorized_safe_rl_env, SafeRLConfig

    net = get_8bus()
    # max_steps=10 (short relative to rollout_len*n_rollouts=32 real env.step()
    # calls below) so at least one episode boundary (done=True) is actually
    # reached -- otherwise update_lagrangian() never fires and this test
    # can't tell a working dual-ascent trigger from a silently-broken one.
    env = VectorizedDynamicDOCREnv(net, n_envs=4, max_steps=10, seed=2)
    safe_env = create_vectorized_safe_rl_env(env, SafeRLConfig(enforce_cti=True, project_actions=True))

    trainer = VectorizedMAPPOTrainer(safe_env, hidden_actor=16, hidden_critic=32,
                                      rollout_len=16, n_epochs=2, seed=2, device="cpu")
    initial_lambdas = safe_env.lagrangian.lambdas.copy()

    result = trainer.train(total_steps=128, verbose=False)

    assert "TMS" in result and "PS" in result and "OF" in result
    for of_val, cti_rate in zip(result["history"], result["cti_satisfaction_history"]):
        assert np.isfinite(of_val)
        assert 0.0 <= cti_rate <= 1.0

    stats = safe_env.get_safe_rl_stats()
    assert "lambda_mean" in stats and "rho" in stats
    assert safe_env.n_updates > 0, "Lagrangian should have been dual-ascent-updated at least once"
    assert not np.allclose(initial_lambdas, safe_env.lagrangian.lambdas), \
        "Lagrangian lambdas never moved from initialization during training"


def test_vectorized_trainer_curriculum_advances_stage():
    """Phase V13: curriculum stage auto-advance wiring, mirroring
    MAPPOMARLTrainer.train()'s identical trigger formula (recent
    CTI-satisfaction rate over the last 100 rollouts clearing
    stage_success_threshold, past a step checkpoint) and wiring
    VectorizedDynamicDOCREnv.set_curriculum_stage() (previously a dead
    stub with zero callers in this trainer). stage_success_threshold is
    overridden to -1.0 (always satisfied) to isolate the step-threshold/
    advancement-count mechanics from whether the policy genuinely improves
    CTI satisfaction within this short a budget -- that's not what this
    test checks."""
    net = get_8bus()
    env = VectorizedDynamicDOCREnv(net, n_envs=2, max_steps=50, seed=4)
    trainer = VectorizedMAPPOTrainer(env, hidden_actor=8, hidden_critic=16,
                                      rollout_len=1, n_epochs=1, seed=4, device="cpu",
                                      curriculum=True, curriculum_stages=3)
    trainer.stage_success_threshold = -1.0

    assert trainer.current_stage == 0
    result = trainer.train(total_steps=210, verbose=False)

    assert trainer.current_stage == 2, f"expected final stage 2, got {trainer.current_stage}"
    assert trainer.env.curriculum_stage == 2, \
        "env.set_curriculum_stage() should have been called -- the dead stub this phase wires up"
    assert result["curriculum_stage"] == 2


def test_vectorized_trainer_ewc_integration():
    """Phase V13: ContinualMAPPOTrainer (marl/continual_trainer.py) only
    ever touches base_trainer.actor.get_all_params()/.set_all_params() --
    already satisfied by VectorizedMAPPOTrainer's actor (the same
    SharedGaussianActor class MAPPOMARLTrainer uses) with zero changes to
    continual_trainer.py. Fisher estimation uses the real per-sample
    estimator fed from this trainer's own rollout buffer
    (get_fisher_diagonal_from_buffer()), not run_core_training.py's own,
    separate, older finite-difference approximation."""
    from marl.continual_trainer import ContinualMAPPOTrainer, EWCConfig

    net = get_8bus()
    env = VectorizedDynamicDOCREnv(net, n_envs=2, max_steps=50, seed=6)
    trainer = VectorizedMAPPOTrainer(env, hidden_actor=8, hidden_critic=16,
                                      rollout_len=8, n_epochs=1, seed=6, device="cpu")
    trainer.train(total_steps=32, verbose=False)  # populates the rollout buffer

    ewc = ContinualMAPPOTrainer(trainer, ewc_config=EWCConfig(fisher_samples=50))

    conditions_a = {"pv_level": 0.0, "topology_flag": 0.0, "fault_type": "3ph"}
    is_new, task_a = ewc.detect_task_change(conditions_a)
    assert is_new

    params_a = trainer.actor.get_all_params()
    fisher_a = trainer.get_fisher_diagonal_from_buffer(n_samples=20)
    assert fisher_a.shape == params_a.shape
    assert np.all(np.isfinite(fisher_a))
    assert np.all(fisher_a >= 0.0)

    conditions_b = {"pv_level": 0.6, "topology_flag": 0.0, "fault_type": "3ph"}
    is_new_b, task_b = ewc.detect_task_change(conditions_b)
    assert is_new_b
    ewc.on_task_transition(task_a, task_b, params_a)
    ewc.ewc_regularizer.update_fisher(task_a, fisher_a)

    penalty, ewc_grad = ewc.compute_ewc_penalty(trainer.actor.get_all_params())
    assert np.isfinite(penalty)
    assert ewc_grad.shape == params_a.shape

    fake_gradient = np.zeros_like(params_a)
    regularized = ewc.apply_ewc_to_gradient(trainer.actor.get_all_params(), fake_gradient)
    assert regularized.shape == params_a.shape
    assert np.all(np.isfinite(regularized))
