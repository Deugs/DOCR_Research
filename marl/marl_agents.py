"""
marl_agents.py
──────────────
Multi-Agent PPO (MAPPO) with GNN-Based Learned Communication.

Phase 2: Enhanced decentralized agent architecture with:
- MAPPO (Multi-Agent PPO) with parameter sharing
- GNN communication graph for coordination
- Centralized training, decentralized execution (CTDE)
- Support for neighbor observations and message passing

Architecture
────────────
- Shared actor network (parameter sharing across agents)
- Shared critic network conditioning on global state
- GNN communication module (comm_graph.py)
- Each agent: o_i -> [encode -> communicate -> decode] -> enhanced_o_i -> policy

Design Decisions
────────────────
- MAPPO over IPPO: centralized value function reduces variance
- GNN over fixed topology: learned attention adapts to network conditions
- 8-float messages (~32 bytes): bandwidth-efficient
- Parameter sharing: reduces sample complexity, enables scaling

References
──────────
- Yu et al., "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games", NeurIPS 2022
- Jiang & Lu, "Learning Attentional Communication for Multi-Agent Cooperation", NeurIPS 2018
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
import time

from network.network_data import NetworkData
from environment.dynamic_network import DistributionNetwork
from network.relay_physics import (
    objective_function, check_constraints,
    compute_operating_times,
    TMS_MIN, TMS_MAX, CTI, TOP_MIN, TOP_MAX,
    FAULT_TYPES,
)


def _fault_type_idx(fault_type: str) -> int:
    """Map an env's fault_type string to the critic's one-hot index.

    Every real call site in this file used to pass the literal
    `fault_type_idx=0` regardless of the env's actual fault_type --
    MAPPOCentralCritic's global state reserves 4 dims for a one-hot over
    FAULT_TYPES (see _critic_global_state), but that conditioning was
    functionally inert: the critic always "saw" 3ph even when training
    used a different fault type (e.g. via a real --fault-type CLI flag).
    Falls back to index 0 for an unrecognized string rather than raising,
    since a hardcoded 0 for every call was already the (broken) status quo.
    """
    try:
        return FAULT_TYPES.index(fault_type)
    except ValueError:
        return 0

# torch is now a REQUIRED dependency of this module, not optional.
#
# History: through 2026-10-21, training here was hand-rolled NumPy +
# OpenAI-ES (evolution strategies) everywhere, with torch only an optional
# accelerant for one narrow piece (batching the ES perturbation population
# via torch.einsum, real but modest 1.3x-1.6x on real CUDA hardware tested
# that day -- see git history / EXPERIMENT_GUIDE.md section 2.2.4 for that
# era's numbers, which no longer apply to the current code).
#
# 2026-10-22: ES was replaced throughout this file with real PyTorch
# autodiff (backprop + Adam) -- the actual motivation for using GPUs at
# all here, since batching a gradient-free population search is a much
# narrower win than real gradient-based training. There is no more NumPy
# training fallback: SharedGaussianActor/MAPPOCentralCritic are
# torch.nn.Module-backed unconditionally. `device` ("cpu" or "cuda") is
# chosen once per instance at construction (see _default_device below) --
# CPU still works fine (just without GPU acceleration), but torch itself
# must be installed either way.
#
# IMPORT ORDER WARNING: importing torch before pandapower in the same
# process segfaults immediately on at least one real development machine
# used on this project — zero Python traceback, zero diagnostic output, a
# native OpenMP/MKL runtime conflict between torch's and pandapower's (via
# numba) bundled runtimes. Confirmed order-dependent: pandapower-then-torch
# is safe, torch-then-pandapower crashes. Every real entry point in this
# codebase imports network/marl_env (which transitively imports pandapower
# via environment/dynamic_network.py) before this module, so this is not
# triggered today — but a segfault can't be caught with try/except, so if
# you ever write a new script, notebook, or REPL session that touches this
# project, import something from network/environment/marl BEFORE import
# torch, not after.
import os
import torch
_TORCH_AVAILABLE = True
_CUDA_PRESENT = torch.cuda.is_available()  # pure hardware fact, always accurate

# GPU does NOT auto-activate just because CUDA is present -- requires
# explicit opt-in (DOCR_USE_GPU=1). Re-benchmarked after the ES-to-backprop
# rewrite above (2026-10-22, same GTX 1050 as the ES-era numbers in
# EXPERIMENT_GUIDE.md section 2.2.4): real backprop's one lean forward+
# backward pass per call is so much cheaper than ES's old 2*n_pert forward
# passes that GPU's fixed per-call overhead (kernel launch, sync, host-
# device transfer) now DOMINATES even more than before, flipping the
# verdict entirely -- actor update measured 0.45x-0.67x (i.e. SLOWER) on
# GPU at both 8-bus and 39-bus, not the 1.3x-1.6x the old ES-population-
# batching path achieved (that code no longer exists). Auto-enabling GPU
# whenever present would make training slower by default on any machine
# with a GPU -- the opposite of this rewrite's goal. If you're on
# different hardware (a much more powerful GPU, or far larger batch
# sizes) where you have reason to believe this verdict flips, opt in
# explicitly: `DOCR_USE_GPU=1 python ...`. Re-measure before trusting it;
# this exact mistake (assuming a port helps without checking) is why this
# comment exists.
_GPU_AVAILABLE = _CUDA_PRESENT and os.environ.get("DOCR_USE_GPU") == "1"
_GPU_DEVICE = torch.device("cuda") if _GPU_AVAILABLE else None


def relu(x): return np.maximum(0, x)
def tanh(x): return np.tanh(x)


def _default_device():
    """Resolves to the module-level GPU flags at call time (not import
    time), so run_core_training.py's _disable_gpu_in_worker() monkeypatch
    (which flips _GPU_AVAILABLE False in pool workers) still works exactly
    as before, even though the dispatch mechanism downstream of this
    changed from a per-call `if _GPU_AVAILABLE: ... _gpu variant` branch
    (ES-era) to a device baked into the actor/critic at construction time
    (2026-10-22 GPU-port rewrite, see module docstring)."""
    return _GPU_DEVICE if _GPU_AVAILABLE else torch.device("cpu")


class _TorchMLP(torch.nn.Module):
    """2-hidden-layer MLP: in -> h1 -> h2 -> out. Replaces the hand-rolled
    NumPy MLP so actor/critic updates can use real backprop + Adam instead
    of OpenAI-ES (see module docstring for why ES was replaced). Weight
    init matches the old NumPy version's N(0, sqrt(2/fan_in)), zero bias —
    not load-bearing for correctness, just keeps early-training behavior
    close to what this codebase's hyperparameters were tuned against."""

    def __init__(self, in_dim, hidden, out_dim, seed=0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.fc1 = torch.nn.Linear(in_dim, hidden)
        self.fc2 = torch.nn.Linear(hidden, hidden)
        self.fc3 = torch.nn.Linear(hidden, out_dim)
        for layer, fan_in in ((self.fc1, in_dim), (self.fc2, hidden), (self.fc3, hidden)):
            torch.nn.init.normal_(layer.weight, mean=0.0, std=(2.0 / fan_in) ** 0.5, generator=gen)
            torch.nn.init.zeros_(layer.bias)

    def forward(self, x):
        h1 = torch.relu(self.fc1(x))
        h2 = torch.relu(self.fc2(h1))
        return self.fc3(h2)


class SharedGaussianActor:
    """
    Shared Gaussian policy for all agents (parameter sharing).

    All agents use the same network weights but receive different observations.
    This reduces sample complexity and enables scaling to larger networks.

    2026-10-22: rewritten from a hand-rolled NumPy MLP + OpenAI-ES surrogate
    gradient to a real torch.nn.Module trained via backprop + Adam (see
    module docstring and EXPERIMENT_GUIDE.md section 2.2.4 for why). The
    public interface (get_mean/sample/log_prob/get_all_params/
    set_all_params) is unchanged so callers elsewhere in this codebase
    don't need to change; internally these now wrap torch tensors instead
    of raw NumPy arrays. update() replaces _update_actor_es.
    """

    def __init__(self, obs_dim, act_dim, hidden=64, seed=0, device=None, lr=3e-4):
        self.act_dim = act_dim
        self.device = torch.device(device) if device is not None else _default_device()
        self.net = _TorchMLP(obs_dim, hidden, act_dim, seed).to(self.device)
        self.log_std = torch.nn.Parameter(torch.full((act_dim,), -1.0, device=self.device))
        self.optimizer = torch.optim.Adam(list(self.net.parameters()) + [self.log_std], lr=lr)

    def get_mean(self, obs):
        """Get mean action for single (obs_dim,) or batch (T, obs_dim)
        observations. NumPy in, NumPy out, no-grad -- this is the hot path
        for rollout collection (acting in the environment), which has no
        business building an autograd graph."""
        obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            mean = torch.tanh(self.net(obs_t))
        return mean.cpu().numpy()

    def sample(self, obs, rng) -> Tuple[np.ndarray, float]:
        """Sample action and compute log probability."""
        mean = self.get_mean(obs)
        log_std_np = self.log_std.detach().cpu().numpy()
        std = np.exp(np.clip(log_std_np, -3, 0))
        noise = rng.normal(0, 1, self.act_dim)
        action = np.clip(mean + std * noise, -1.0, 1.0)
        log_prob = float(np.sum(
            -0.5 * (noise**2 + 2 * log_std_np + np.log(2 * np.pi))))
        return action, log_prob

    def _log_prob_t(self, obs_t, act_t):
        """Differentiable log-prob, used by both log_prob() (under
        no_grad) and update() (with grad tracking)."""
        mean = torch.tanh(self.net(obs_t))
        std = torch.exp(torch.clamp(self.log_std, -3, 0))
        z = (act_t - mean) / (std + 1e-8)
        terms = -0.5 * (z**2 + 2 * self.log_std + np.log(2 * np.pi))
        return terms.sum(dim=-1)

    def log_prob(self, obs, action):
        """Compute log probability of action(s). NumPy in, NumPy/float out,
        no-grad (see _log_prob_t for the differentiable version used by
        update()).

        obs/action of shape (obs_dim,)/(act_dim,) -> returns a scalar float.
        obs/action of shape (T, obs_dim)/(T, act_dim) -> returns a (T,) array.
        """
        obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(np.asarray(action), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            lp = self._log_prob_t(obs_t, act_t)
        return float(lp.item()) if obs_t.ndim == 1 else lp.cpu().numpy()

    def update(self, obs_batch, act_batch, adv_batch, old_lp_batch, clip_eps=0.2):
        """One real PPO-clip gradient step via backprop + Adam. Replaces
        _update_actor_es's one ES generation; the caller (MAPPOMARLTrainer
        ._update_actor) still loops this n_epochs times per rollout,
        exactly as it looped _update_actor_es before."""
        obs_t = torch.as_tensor(np.asarray(obs_batch), dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(np.asarray(act_batch), dtype=torch.float32, device=self.device)
        adv_t = torch.as_tensor(np.asarray(adv_batch), dtype=torch.float32, device=self.device)
        old_lp_t = torch.as_tensor(np.asarray(old_lp_batch), dtype=torch.float32, device=self.device)

        self.optimizer.zero_grad()
        new_lp = self._log_prob_t(obs_t, act_t)
        ratio = torch.exp(torch.clamp(new_lp - old_lp_t, -5, 5))
        clip_r = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
        loss = -torch.minimum(ratio * adv_t, clip_r * adv_t).mean()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def get_all_params(self):
        """Flat parameter vector. Kept for backward compatibility with
        tests/fixtures written against the ES-era API (see
        tests/test_phase2.py, test_phase5.py, test_figure_regressions.py)
        -- no longer used internally for perturbation, just an adapter
        over the real torch parameters."""
        with torch.no_grad():
            flat = torch.cat([p.reshape(-1) for p in self.net.parameters()] + [self.log_std.reshape(-1)])
        return flat.cpu().numpy()

    def set_all_params(self, p):
        p_t = torch.as_tensor(np.asarray(p), dtype=torch.float32, device=self.device)
        torch.nn.utils.vector_to_parameters(p_t[:-self.act_dim], self.net.parameters())
        with torch.no_grad():
            self.log_std.copy_(p_t[-self.act_dim:])

    def estimate_fisher_diagonal(self, obs_batch, act_batch) -> np.ndarray:
        """Real per-sample Fisher Information diagonal via torch.autograd.grad:
        F_i = mean_samples[(d log pi(a|o)/d theta_i)^2] (Kirkpatrick et al.,
        2017's actual definition). 2026-10-22: replaces the finite-difference
        approximation marl/experiments/exp_fig39_ablation.py previously used
        (perturb each parameter by h=1e-3, measure log-prob change) because
        no real gradients existed before this rewrite -- that was both
        approximate (finite-difference truncation error) and expensive (two
        full forward passes per (sample, parameter) pair, so only a random
        subset of parameters was ever actually estimated). This is exact and
        faster (one backward pass per sample, all parameters at once).

        obs_batch/act_batch: (n_samples, obs_dim)/(n_samples, act_dim).
        Returns a flat (n_params,) array in the same layout as
        get_all_params()/set_all_params()."""
        obs_t = torch.as_tensor(np.asarray(obs_batch), dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(np.asarray(act_batch), dtype=torch.float32, device=self.device)
        params = list(self.net.parameters()) + [self.log_std]
        n_samples = obs_t.shape[0]

        fisher_sum = torch.zeros(sum(p.numel() for p in params), device=self.device)
        for i in range(n_samples):
            lp = self._log_prob_t(obs_t[i:i + 1], act_t[i:i + 1]).squeeze()
            grads = torch.autograd.grad(lp, params, retain_graph=False, create_graph=False)
            flat_grad = torch.cat([g.reshape(-1) for g in grads])
            fisher_sum += flat_grad.detach() ** 2

        fisher = (fisher_sum / max(n_samples, 1)).cpu().numpy()
        return fisher


class MAPPOCentralCritic:
    """
    MAPPO Centralized Critic with global state.

    Global state: [TMS_all || PS_all || PV_level || topology_flag || fault_type_enc || neighbor_stats]

    Unlike IPPO where each agent has its own critic, MAPPO uses a single
    shared critic that conditions on the global state, reducing variance.
    """

    def __init__(self, n_relays, hidden=128, seed=0, use_global_state=True, device=None, lr=1e-3):
        self.n_relays = n_relays
        self.use_global_state = use_global_state

        # Global state dimension
        if use_global_state:
            # TMS (n_relays) + PS (n_relays) + PV_level (1) + topology_flag (1) +
            # fault_type_onehot (4) + neighbor_stats (6)
            global_dim = 2 * n_relays + 12
        else:
            global_dim = 2 * n_relays + 6

        self.device = torch.device(device) if device is not None else _default_device()
        self.net = _TorchMLP(global_dim, hidden, 1, seed).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

    def _global_state_np(self, TMS, PS, pv_level, topology_flag, fault_type_idx, neighbor_stats):
        global_state = np.concatenate([TMS, PS])
        fault_onehot = np.zeros(4)
        if 0 <= fault_type_idx < 4:
            fault_onehot[fault_type_idx] = 1.0
        if self.use_global_state and neighbor_stats is not None:
            return np.concatenate([global_state, [pv_level, topology_flag], fault_onehot, neighbor_stats])
        return np.concatenate([global_state, [pv_level, topology_flag], fault_onehot])

    def value(self, TMS, PS, pv_level=0.0, topology_flag=0.0,
              fault_type_idx=0, neighbor_stats=None) -> float:
        """Compute value given global state. NumPy in, float out, no-grad
        -- the hot path used during rollout collection and GAE."""
        global_state = self._global_state_np(TMS, PS, pv_level, topology_flag, fault_type_idx, neighbor_stats)
        gs_t = torch.as_tensor(global_state, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return float(self.net(gs_t)[0].item())

    def _global_state_batch_np(self, TMS, PS, pv_level, topology_flag, neighbor_stats, fault_type_idx):
        T = TMS.shape[0]
        global_state = np.concatenate([TMS, PS], axis=1)
        fault_onehot = np.zeros((T, 4))
        if 0 <= fault_type_idx < 4:
            fault_onehot[:, fault_type_idx] = 1.0
        extra = np.stack([np.asarray(pv_level), np.asarray(topology_flag)], axis=1)
        if self.use_global_state and neighbor_stats is not None:
            return np.concatenate([global_state, extra, fault_onehot, neighbor_stats], axis=1)
        return np.concatenate([global_state, extra, fault_onehot], axis=1)

    def value_batch(self, TMS, PS, pv_level, topology_flag, neighbor_stats, fault_type_idx=0) -> np.ndarray:
        """Batched form of value() for T timesteps at once. NumPy in,
        NumPy out, no-grad (see update() for the differentiable path)."""
        global_state = self._global_state_batch_np(TMS, PS, pv_level, topology_flag, neighbor_stats, fault_type_idx)
        gs_t = torch.as_tensor(global_state, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return self.net(gs_t).cpu().numpy().ravel()

    def update(self, TMS, PS, pv_level, topology_flag, neighbor_stats, ret_batch, fault_type_idx=0):
        """One real MSE gradient step via backprop + Adam, replacing
        _update_critic_es's one ES generation. Caller (MAPPOMARLTrainer
        ._update_critic) still loops this n_epochs times per rollout."""
        global_state = self._global_state_batch_np(TMS, PS, pv_level, topology_flag, neighbor_stats, fault_type_idx)
        gs_t = torch.as_tensor(global_state, dtype=torch.float32, device=self.device)
        ret_t = torch.as_tensor(np.asarray(ret_batch), dtype=torch.float32, device=self.device)

        self.optimizer.zero_grad()
        pred = self.net(gs_t)[:, 0]  # (T,) -- one centralized prediction per timestep
        # ret_t is (T, n_agents) -- each agent has its own GAE-computed
        # return at a timestep, compared against that timestep's single
        # shared critic prediction (broadcast, not elementwise-matched).
        loss = torch.mean((ret_t - pred[:, None]) ** 2)
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def get_params(self):
        """Flat parameter vector -- kept for backward compatibility with
        ES-era tests/fixtures, see SharedGaussianActor.get_all_params."""
        with torch.no_grad():
            flat = torch.cat([p.reshape(-1) for p in self.net.parameters()])
        return flat.cpu().numpy()

    def set_params(self, p):
        p_t = torch.as_tensor(np.asarray(p), dtype=torch.float32, device=self.device)
        torch.nn.utils.vector_to_parameters(p_t, self.net.parameters())

    def n_params(self):
        return sum(p.numel() for p in self.net.parameters())


class MAPPOBuffer:
    """
    Rollout buffer for MAPPO with communication tracking.

    Stores: observations, enhanced_obs (with comm), actions, rewards,
            log_probs, values, communication info, done flags
    """

    def __init__(self, n_agents, obs_dim, act_dim, capacity, use_comm=False, msg_dim=8,
                 raw_obs_dim=None):
        self.obs = np.zeros((capacity, n_agents, obs_dim))
        self.actions = np.zeros((capacity, n_agents, act_dim))
        self.rewards = np.zeros((capacity, n_agents))
        self.log_probs = np.zeros((capacity, n_agents))
        self.values = np.zeros(capacity)
        self.dones = np.zeros(capacity, dtype=bool)
        self.pv_levels = np.zeros(capacity)
        self.top_flags = np.zeros(capacity)
        # Per-timestep global relay state, needed to recompute the critic's
        # prediction during the update step against the *historical* state
        # that produced ret_batch[t] rather than the environment's current state.
        self.tms = np.zeros((capacity, n_agents))
        self.ps = np.zeros((capacity, n_agents))

        # Communication tracking
        self.use_comm = use_comm
        if use_comm:
            self.messages = np.zeros((capacity, n_agents, msg_dim))
            self.attention = np.zeros((capacity, n_agents, n_agents))
            # Pre-communication observations, needed to train the GNN comm
            # module itself (obs -> forward() -> enhanced_obs); self.obs above
            # already holds the *enhanced* observations when use_comm is on.
            self.raw_obs = np.zeros((capacity, n_agents, raw_obs_dim or obs_dim))

        self.ptr = self.size = 0
        self.capacity = capacity
        self.n_agents = n_agents

    def add(self, obs, actions, rewards, log_probs, value, done,
            pv_level=0.0, topology_flag=0.0, messages=None, attention=None,
            tms=None, ps=None, raw_obs=None):
        i = self.ptr % self.capacity
        self.obs[i] = obs
        self.actions[i] = actions
        self.rewards[i] = rewards
        self.log_probs[i] = log_probs
        self.values[i] = value
        self.dones[i] = done
        self.pv_levels[i] = pv_level
        if tms is not None:
            self.tms[i] = tms
        if ps is not None:
            self.ps[i] = ps
        self.top_flags[i] = topology_flag
        if self.use_comm and raw_obs is not None:
            self.raw_obs[i] = raw_obs

        if self.use_comm and messages is not None:
            self.messages[i] = messages
        if self.use_comm and attention is not None:
            self.attention[i] = attention

        self.ptr += 1
        self.size = min(self.ptr, self.capacity)

    def compute_gae(self, gamma=0.99, lam=0.95, last_value=0.0) -> Tuple[np.ndarray, np.ndarray]:
        """Generalised Advantage Estimation (single shared value)."""
        T = self.size
        adv = np.zeros((T, self.n_agents))
        ret = np.zeros((T, self.n_agents))

        for ag in range(self.n_agents):
            gae = 0.0
            for t in reversed(range(T)):
                Vn = (last_value if (t == T - 1 or self.dones[t])
                      else self.values[t + 1])
                delta = (self.rewards[t, ag]
                         + gamma * Vn * (1 - self.dones[t])
                         - self.values[t])
                gae = delta + gamma * lam * (1 - self.dones[t]) * gae
                adv[t, ag] = gae
                ret[t, ag] = gae + self.values[t]

            adv[:, ag] = ((adv[:, ag] - adv[:, ag].mean())
                          / (adv[:, ag].std() + 1e-8))
        return adv, ret

    def clear(self):
        self.ptr = self.size = 0


class MAPPOMARLTrainer:
    """
    MAPPO Trainer with optional GNN Communication.

    Parameters
    ──────────
    env           DynamicDOCREnv instance
    use_comm      Enable GNN communication
    hidden_actor  hidden units per shared actor
    hidden_critic hidden units for centralized critic
    lr_actor      Adam learning rate for actor update (was an ES step size
                  before the 2026-10-22 backprop rewrite -- 0.02 is far too
                  large for Adam, hence the new default)
    lr_critic     Adam learning rate for critic update (same note)
    clip_eps      PPO clip ratio epsilon
    gamma / lam   discount and GAE parameters
    rollout_len   steps collected before each update
    n_epochs      gradient steps per rollout (was ES perturbation rounds
                  before the backprop rewrite -- same call pattern, now
                  each one is a real Adam step instead of one ES generation)
    curriculum    Enable curriculum learning
    """

    def __init__(
        self,
        env,
        use_comm=False,
        hidden_actor=64,
        hidden_critic=128,
        lr_actor=3e-4,
        lr_critic=1e-3,
        lr_comm=1e-3,
        clip_eps=0.2,
        gamma=0.99,
        lam=0.95,
        rollout_len=64,
        n_epochs=4,
        seed=0,
        curriculum=False,
        curriculum_stages=6,
    ):
        self.env = env
        self.n_agents = env.n_agents
        self.use_comm = use_comm
        self.clip_eps = clip_eps
        self.gamma = gamma
        self.lam = lam
        self.rollout_len = rollout_len
        self.n_epochs = n_epochs
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.rng = np.random.default_rng(seed)

        # Curriculum learning
        self.curriculum = curriculum
        self.curriculum_stages = curriculum_stages
        self.current_stage = 0
        self.stage_episodes = 0
        self.stage_success_threshold = 0.95

        # Communication module (must be created before actor to know enhanced_dim).
        # 2026-10-22: CommGraph now trains itself via update() (real backprop
        # through the actual downstream policy objective) -- there's no more
        # separate GNNCommTrainer object; without calling comm_graph.update()
        # each rollout (see train() below) the GNN's weights would never move
        # from their random initialization, same risk as before this rewrite.
        self.comm_graph = None
        if use_comm:
            try:
                from marl.comm_graph import CommGraph, build_relay_adjacency
                adj = build_relay_adjacency(env.net)
                self.comm_graph = CommGraph(
                    n_agents=self.n_agents,
                    obs_dim=env.OBS_DIM,
                    enhanced_dim=env.OBS_DIM + 8,  # Original + message features
                    adjacency=adj,
                    seed=seed + 100,
                    device=_default_device(),
                    lr=lr_comm,
                )
            except ImportError:
                print("  Warning: comm_graph not available, running without communication")
                self.use_comm = False

        # Shared actor (parameter sharing)
        # When using communication, actor receives enhanced observations
        actor_obs_dim = env.OBS_DIM
        if self.use_comm and self.comm_graph is not None:
            actor_obs_dim = self.comm_graph.decoder.enhanced_dim
        self.actor = SharedGaussianActor(actor_obs_dim, env.ACT_DIM, hidden_actor, seed, lr=lr_actor)

        # Centralized critic
        self.critic = MAPPOCentralCritic(
            self.n_agents, hidden_critic, seed + 999,
            use_global_state=True, lr=lr_critic,
        )

        # Buffer - use enhanced observation dimension if using communication
        buffer_obs_dim = env.OBS_DIM
        if self.use_comm and self.comm_graph is not None:
            buffer_obs_dim = self.comm_graph.decoder.enhanced_dim
        self.buffer = MAPPOBuffer(
            self.n_agents, buffer_obs_dim, env.ACT_DIM, rollout_len,
            use_comm=use_comm,
            msg_dim=8 if use_comm else 0,
            raw_obs_dim=env.OBS_DIM,
        )

        # Statistics
        self.comm_bandwidth_history = []
        self.comm_attention_history = []

        # ── Tier-2: Attention snapshots (Figs 8, 38) ──────────────────────────
        # Saved every `attention_checkpoint_interval` rollout steps when use_comm.
        # Each entry: {"step": int, "attn_matrix": ndarray(n,n), "entropy": float}
        self.attention_checkpoint_interval = 500
        self.attention_snapshots: List[Dict] = []
        self._global_step: int = 0

        # ── Tier-2: Per-agent return log (Fig 37) ─────────────────────────────
        # per_agent_returns[i] = list of episode mean returns for agent i
        self.per_agent_returns: List[List[float]] = [[] for _ in range(self.n_agents)]

    def _get_neighbor_stats(self, obs):
        """Compute neighbor statistics for global critic input."""
        # Mean and std of neighbor TMS/PS from observations
        # obs[:, 0] = tms_norm, obs[:, 1] = ps_norm
        mean_tms = float(np.mean(obs[:, 0]))
        mean_ps = float(np.mean(obs[:, 1]))
        std_tms = float(np.std(obs[:, 0]))
        std_ps = float(np.std(obs[:, 1]))
        max_viol = float(np.max(obs[:, 4]))  # viol_flag
        mean_slack = float(np.mean(obs[:, 3]))  # cti_slack_norm
        return np.array([mean_tms, mean_ps, std_tms, std_ps, max_viol, mean_slack])

    def _get_neighbor_stats_batch(self, obs_batch):
        """Batched form of _get_neighbor_stats: obs_batch shape
        (T, n_agents, obs_dim) -> (T, 6), same formulas reduced over the
        agent axis (axis=1) instead of a Python loop calling
        _get_neighbor_stats(obs_batch[t]) once per timestep."""
        mean_tms = obs_batch[:, :, 0].mean(axis=1)
        mean_ps = obs_batch[:, :, 1].mean(axis=1)
        std_tms = obs_batch[:, :, 0].std(axis=1)
        std_ps = obs_batch[:, :, 1].std(axis=1)
        max_viol = obs_batch[:, :, 4].max(axis=1)
        mean_slack = obs_batch[:, :, 3].mean(axis=1)
        return np.stack([mean_tms, mean_ps, std_tms, std_ps, max_viol, mean_slack], axis=1)

    def _update_actor(self, agent_idx, obs_batch, act_batch, adv_batch, old_lp):
        """One real PPO-clip gradient step for the shared actor, via
        SharedGaussianActor.update() (backprop + Adam). Replaces
        _update_actor_es's one ES generation -- train()'s call site still
        loops this n_epochs times per rollout, same as before. `agent_idx`
        is unused (the actor is shared across agents; kept in the
        signature to match the pre-rewrite call site in train() without
        forcing a call-site change)."""
        self.actor.update(obs_batch, act_batch, adv_batch, old_lp, clip_eps=self.clip_eps)

    def _update_critic(self, obs_batch, ret_batch):
        """One real MSE gradient step for the centralized critic, via
        MAPPOCentralCritic.update() (backprop + Adam). Replaces
        _update_critic_es's one ES generation."""
        T = len(obs_batch)
        # Use the historical per-timestep relay/PV/topology state stored in
        # the buffer (not self.env's *current* state) — ret_batch[t] was
        # computed from the state at timestep t, not the state after rollout.
        neighbor_stats_batch = self._get_neighbor_stats_batch(obs_batch)
        tms_batch = self.buffer.tms[:T]
        ps_batch = self.buffer.ps[:T]
        pv_batch = self.buffer.pv_levels[:T]
        topo_batch = self.buffer.top_flags[:T]
        self.critic.update(
            tms_batch, ps_batch, pv_batch, topo_batch, neighbor_stats_batch,
            ret_batch, fault_type_idx=_fault_type_idx(self.env.fault_type),
        )

    def _collect_rollout(self, obs):
        """Collect one rollout with optional communication."""
        self.buffer.clear()

        for _ in range(self.rollout_len):
            # Apply communication if enabled
            if self.use_comm and self.comm_graph is not None:
                enhanced_obs, messages, comm_info = self.comm_graph.forward(obs, self.rng)
                current_obs = enhanced_obs
            else:
                current_obs = obs
                messages = None
                comm_info = {}

            # Sample actions from shared actor
            actions = np.zeros((self.n_agents, self.env.ACT_DIM))
            log_probs = np.zeros(self.n_agents)
            for i in range(self.n_agents):
                actions[i], log_probs[i] = self.actor.sample(current_obs[i], self.rng)

            TMS, PS = self.env.get_state()

            # Compute value with global state
            pv_level = getattr(self.env, 'current_pv_level', 0.0)
            topology_flag = 0.0
            if hasattr(self.env, 'current_state') and self.env.current_state:
                topology_flag = 1.0 if any(
                    not v for v in self.env.current_state.topology_state.line_status.values()
                ) else 0.0

            # _get_neighbor_stats indexes raw feature columns (tms_norm,
            # ps_norm, cti_slack_norm, viol_flag) — current_obs is the
            # GNN-decoded enhanced observation when use_comm is on, whose
            # columns are an arbitrary nonlinear mix of raw+message
            # features (see GNNMessageDecoder.decode), not those raw
            # columns. Always use the true raw obs here so this matches
            # the GAE bootstrap value below, which already (correctly)
            # computes neighbor_stats from raw obs.
            neighbor_stats = self._get_neighbor_stats(obs)
            value = self.critic.value(
                TMS, PS, pv_level, topology_flag,
                fault_type_idx=_fault_type_idx(self.env.fault_type), neighbor_stats=neighbor_stats
            )

            next_obs, rewards, done, info = self.env.step(actions)

            # Store experience
            self.buffer.add(
                current_obs, actions, rewards, log_probs, value, done,
                pv_level=pv_level, topology_flag=topology_flag,
                messages=messages,
                attention=comm_info.get("attention") if comm_info else None,
                tms=TMS, ps=PS,
                raw_obs=obs if self.use_comm else None,
            )

            # Track communication stats
            if self.use_comm and comm_info:
                bw = self.comm_graph.get_bandwidth_usage(steps_per_second=1.0)
                self.comm_bandwidth_history.append(bw["kbps"])

            # ── Tier-2: save attention snapshot every N steps (Figs 8, 38) ──
            self._global_step += 1
            if (self.use_comm and comm_info and
                    self._global_step % self.attention_checkpoint_interval == 0):
                attn_raw = comm_info.get("attention")   # (n_agents, n_agents) or None
                if attn_raw is not None:
                    attn_mat = np.array(attn_raw)
                    # Compute mean entropy across agents (measure of selectivity)
                    eps_e = 1e-9
                    p = np.clip(attn_mat, eps_e, 1.0)
                    p /= p.sum(axis=1, keepdims=True) + eps_e
                    entropy = float(-np.mean(np.sum(p * np.log(p + eps_e), axis=1)))
                    self.attention_snapshots.append({
                        "step":        self._global_step,
                        "attn_matrix": attn_mat.copy(),
                        "entropy":     entropy,
                    })

            obs = next_obs
            if done:
                obs = self.env.reset()

        return obs

    def train(self, total_steps=30000, verbose=True, log_every=2000,
              early_stop_patience=None, early_stop_min_delta=1e-4):
        """Train MAPPO with optional GNN communication.

        early_stop_patience: if set, stop once `early_stop_patience`
        consecutive rollout evaluations pass without the best objective
        function improving by more than early_stop_min_delta. Disabled by
        default (None) — total_steps is otherwise a fixed budget regardless
        of whether the policy has already converged, which wastes compute on
        easy configurations without helping hard ones. When curriculum
        learning is enabled, early stopping only takes effect once the final
        curriculum stage has been reached, so it can't cut a run off before
        every stage has been visited.
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

        stage_history = {i: [] for i in range(self.curriculum_stages)}
        cti_satisfaction_history = []

        if verbose:
            print(f"\n{'─' * 55}")
            print(f"  MAPPO  ·  {self.env.net.name}  ·  "
                  f"{self.n_agents} agents  ·  {self.env.net.n_pairs} pairs")
            if self.use_comm:
                print(f"  GNN Communication: ON  (msg_dim={self.comm_graph.msg_dim if self.comm_graph else 8})")
            else:
                print(f"  GNN Communication: OFF")
            if self.curriculum:
                print(f"  Curriculum Learning: {self.curriculum_stages} stages")
            print(f"{'─' * 55}")

        while step < total_steps:
            obs = self._collect_rollout(obs)
            T = self.buffer.size
            step += T

            TMS, PS = self.env.get_state()

            # Compute value for GAE
            pv_level = getattr(self.env, 'current_pv_level', 0.0)
            topology_flag = 0.0
            if hasattr(self.env, 'current_state') and self.env.current_state:
                topology_flag = 1.0 if any(
                    not v for v in self.env.current_state.topology_state.line_status.values()
                ) else 0.0

            neighbor_stats = self._get_neighbor_stats(obs)
            last_val = self.critic.value(
                TMS, PS, pv_level, topology_flag,
                fault_type_idx=_fault_type_idx(self.env.fault_type), neighbor_stats=neighbor_stats
            )
            advs, rets = self.buffer.compute_gae(self.gamma, self.lam, last_val)

            # Update shared actor
            for ag in range(self.n_agents):
                for _ in range(self.n_epochs):
                    self._update_actor(
                        ag,
                        self.buffer.obs[:T, ag],
                        self.buffer.actions[:T, ag],
                        advs[:T, ag],
                        self.buffer.log_probs[:T, ag],
                    )

            # Update centralized critic. Pass raw_obs (not buffer.obs, the
            # GNN-enhanced observation used to update the actor above) —
            # _update_critic only uses this to compute neighbor_stats,
            # which indexes raw feature columns that don't exist in the
            # decoded enhanced representation. buffer.raw_obs is None when
            # use_comm is off, in which case buffer.obs already *is* the
            # raw obs (see _collect_rollout).
            for _ in range(self.n_epochs):
                self._update_critic(
                    self.buffer.raw_obs[:T] if self.use_comm else self.buffer.obs[:T],
                    rets[:T],
                )

            # Update GNN communication weights (encoder/attention/decoder) --
            # real PPO-clip backprop through the actual actor objective (see
            # CommGraph.update()'s docstring), not the old ES proxy. Without
            # this call the module's params never move from their random
            # initialization regardless of how long training runs.
            if self.use_comm and self.comm_graph is not None:
                self.comm_graph.update(
                    self.buffer.raw_obs[:T], self.buffer.actions[:T],
                    advs[:T], self.buffer.log_probs[:T], self.actor,
                    clip_eps=self.clip_eps,
                )

            # Evaluate
            from network.relay_physics import objective_function as OF_fn
            if hasattr(self.env, 'dist_net'):
                eval_result = self.env.evaluate_current_settings()
                OF_now = eval_result["OF"]
                cti_ok = eval_result["constraints"]["cti_satisfied"]
            else:
                OF_now, _, _ = OF_fn(self.env.net, TMS, PS)
                cti_ok = False

            improved = OF_now < best_OF - early_stop_min_delta
            if OF_now < best_OF:
                best_OF = OF_now
                best_TMS = TMS.copy()
                best_PS = PS.copy()
            no_improve_count = 0 if improved else no_improve_count + 1

            history.append(OF_now)
            cti_satisfaction_history.append(1.0 if cti_ok else 0.0)

            # ── Tier-2: per-agent return logging (Fig 37) ─────────────────────
            # buffer.rewards shape: (T, n_agents); mean over time dimension
            if self.buffer.size > 0:
                agent_means = self.buffer.rewards[:self.buffer.size].mean(axis=0)
                for i in range(self.n_agents):
                    self.per_agent_returns[i].append(float(agent_means[i]))

            # Curriculum stage advancement
            if self.curriculum:
                stage_history[self.current_stage].append(OF_now)
                recent_cti_rate = np.mean(cti_satisfaction_history[-100:]) if len(cti_satisfaction_history) >= 100 else 0.0

                if (recent_cti_rate > self.stage_success_threshold and
                    self.current_stage < self.curriculum_stages - 1 and
                    step > (self.current_stage + 1) * total_steps // self.curriculum_stages):
                    self.current_stage += 1
                    if hasattr(self.env, 'set_curriculum_stage'):
                        self.env.set_curriculum_stage(self.current_stage)
                    if verbose:
                        print(f"  ↑ Advanced to curriculum stage {self.current_stage}")

            if verbose and step % log_every < T:
                stage_info = f" stage={self.current_stage}" if self.curriculum else ""
                comm_info = ""
                if self.use_comm and self.comm_bandwidth_history:
                    avg_bw = np.mean(self.comm_bandwidth_history[-100:])
                    comm_info = f" bw={avg_bw:.2f}kbps"
                print(f"  step {step:6d}/{total_steps}  "
                      f"OF={OF_now:.4f}  best={best_OF:.4f}  "
                      f"CTI_ok={cti_ok}{stage_info}{comm_info}  "
                      f"t={time.perf_counter() - t0:.1f}s")

            if early_stop_patience is not None:
                curriculum_done = (not self.curriculum) or (self.current_stage >= self.curriculum_stages - 1)
                if curriculum_done and no_improve_count >= early_stop_patience:
                    early_stopped = True
                    if verbose:
                        print(f"  Early stop at step {step}/{total_steps}: "
                              f"no improvement > {early_stop_min_delta} for {no_improve_count} rollouts")
                    break

        elapsed = time.perf_counter() - t0
        if verbose:
            print(f"  Done. Best OF = {best_OF:.4f}  [{elapsed:.1f}s]")
            if self.curriculum:
                print(f"  Final curriculum stage: {self.current_stage}")
            if self.use_comm and self.comm_bandwidth_history:
                avg_bw = np.mean(self.comm_bandwidth_history)
                print(f"  Avg comm bandwidth: {avg_bw:.2f} kbps")

        return {
            "method":           "MAPPO" + ("-Comm" if self.use_comm else ""),
            "network":          self.env.net.name,
            "TMS":              best_TMS,
            "PS":               best_PS,
            "OF":               best_OF,
            "history":          history,
            "elapsed":          elapsed,
            "curriculum_stage": self.current_stage if self.curriculum else 0,
            "stage_history":    stage_history if self.curriculum else {},
            "use_comm":         self.use_comm,
            "comm_bandwidth":   np.mean(self.comm_bandwidth_history) if self.comm_bandwidth_history else 0.0,
            "early_stopped":    early_stopped,
            "stopped_at_step":  step,
            # ── Tier-2 additions ──────────────────────────────────────────────
            "attention_snapshots": self.attention_snapshots,        # Figs 8, 38
            "per_agent_returns":   self.per_agent_returns,          # Fig 37
            "cti_satisfaction_history": cti_satisfaction_history,   # Figs 9, 21
        }


def run_mappo(
    net,
    total_steps=30000,
    seed=0,
    verbose=True,
    use_comm=False,
    curriculum=False,
    curriculum_stage=0,
    fault_type="3ph",
    use_pandapower=False,
    hidden_actor=64,
    hidden_critic=128,
):
    """Convenience function to run MAPPO."""
    from marl.marl_env import DynamicDOCREnv

    env = DynamicDOCREnv(
        net,
        action_scale=0.05,
        max_steps=300,
        seed=seed,
        use_pandapower=use_pandapower,
        curriculum_stage=curriculum_stage,
        fault_type=fault_type,
    )

    trainer = MAPPOMARLTrainer(
        env,
        use_comm=use_comm,
        hidden_actor=hidden_actor,
        hidden_critic=hidden_critic,
        seed=seed,
        curriculum=curriculum,
    )

    return trainer.train(total_steps=total_steps, verbose=verbose)
