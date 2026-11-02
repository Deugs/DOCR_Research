"""
vectorized_scenario_generator.py
──────────────────────────────────
Batched (n_envs,)-wide counterpart of `pv_model/scenario_generator.py`'s
`MarkovCloudModel.generate_daily_profile()`, for the GPU-vectorized-
environment redesign (Phase V2 of the vectorized-env plan).

Scope: ports only `generate_daily_profile()` (and the `generate_sequence()`
Markov-chain step it depends on) -- confirmed via grep across the whole
repo that `environment/dynamic_network.py`'s real training hot path
(`reset()`/`step()`) calls only `self.scenario_gen.cloud_model.
generate_daily_profile()`, cached once per episode and indexed by
time-of-day every step. `ScenarioGenerator`'s other methods
(`generate_scenarios`, `_generate_single_scenario`, `get_scenario_batch`,
`get_curriculum_batch`-adjacent helpers) have no callers in the real
training pipeline (`run_core_training.py`) -- only tests reference them --
so they are out of scope here, same "port only what's load-bearing"
discipline as Phase V1.

Of the three outputs this produces (`irradiance`, `temperature`,
`load_pu`), only `load_pu` is ever actually read downstream
(`DistributionNetwork.get_sensor_measurements()`'s `load_current` term) --
`irradiance`/`temperature` are stored on `DynamicState` but never consumed
by anything reward/observation-relevant (confirmed via grep). All three
are still implemented faithfully regardless, matching the single-env
version's exact formulas, since the Markov chain and the hour-based
profile math are not meaningfully more work together than `load_pu` alone
would be, and partial fidelity would make future consumers of
`irradiance`/`temperature` a silent trap.

Statistical, not bit-identical, equivalence to the single-env version is
the correctness bar here (same precedent as the metaheuristics
vectorization in the completed torch-port rewrite): the Markov chain's RNG
draw order necessarily changes once draws happen across all `n_envs` at
once rather than one environment's full sequence at a time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


def _default_device():
    """Reuses the project's DOCR_USE_GPU=1 opt-in policy."""
    try:
        from marl.marl_agents import _default_device as _marl_default_device
        return _marl_default_device()
    except ImportError:
        import os
        cuda_present = torch.cuda.is_available()
        gpu_available = cuda_present and os.environ.get("DOCR_USE_GPU") == "1"
        return torch.device("cuda") if gpu_available else torch.device("cpu")


@dataclass
class VectorizedDailyProfile:
    """Batched counterpart of DailyProfile. hours: (n_steps,) -- shared
    across all environments (the time grid itself doesn't vary).
    irradiance/temperature/load_pu: (n_envs, n_steps)."""
    hours: torch.Tensor
    irradiance: torch.Tensor
    temperature: torch.Tensor
    load_pu: torch.Tensor


class VectorizedMarkovCloudModel:
    """Batched Markov chain cloud-cover model -- same transition matrix,
    same states (Clear=0, Partly Cloudy=1, Overcast=2), same irradiance
    scaling factors as the single-env MarkovCloudModel, running n_envs
    independent chains in lockstep (sequential in time -- each timestep's
    state genuinely depends on the previous one, so that axis cannot be
    parallelized away -- but batched across all environments at each of
    those sequential steps, which is where the real per-call cost was)."""

    def __init__(self, n_envs: int, device=None, seed: Optional[int] = None):
        self.n_envs = n_envs
        self.device = device if device is not None else _default_device()
        self.P = torch.tensor([
            [0.85, 0.12, 0.03],   # From Clear
            [0.20, 0.70, 0.10],   # From Partly Cloudy
            [0.10, 0.25, 0.65],   # From Overcast
        ], dtype=torch.float64, device=self.device)
        # index order matches MarkovCloudModel.states = ["clear", "partly_cloudy", "overcast"]
        self.irradiance_factor = torch.tensor([1.0, 0.6, 0.2], dtype=torch.float64, device=self.device)

        self.generator = torch.Generator(device=self.device)
        if seed is not None:
            self.generator.manual_seed(seed)

    def generate_sequence(self, n_steps: int, initial_state: int = 0) -> torch.Tensor:
        """Returns (n_envs, n_steps) long tensor of cloud states."""
        states = torch.full((self.n_envs,), initial_state, dtype=torch.long, device=self.device)
        seq = [states]
        for _ in range(n_steps - 1):
            probs = self.P[states]  # (n_envs, 3) -- this env's current-state transition row
            states = torch.multinomial(probs, 1, generator=self.generator).squeeze(1)
            seq.append(states)
        return torch.stack(seq, dim=1)  # (n_envs, n_steps)

    def generate_daily_profile(self, resolution_min: int = 15) -> VectorizedDailyProfile:
        """Batched counterpart of MarkovCloudModel.generate_daily_profile(),
        same formulas throughout."""
        n_steps = 24 * 60 // resolution_min
        hours = torch.arange(0, 24, resolution_min / 60, dtype=torch.float64, device=self.device)

        states = self.generate_sequence(n_steps)  # (n_envs, n_steps)

        hour_angle = 2 * np.pi * (hours - 12) / 24
        cos_factor = torch.clamp(torch.cos(hour_angle), min=0.0)  # (n_steps,)
        g_clear = 1000.0 * cos_factor  # (n_steps,)

        irr_factor_per_step = self.irradiance_factor[states]  # (n_envs, n_steps)
        irradiance = g_clear.unsqueeze(0) * irr_factor_per_step  # (n_envs, n_steps)

        temp_noise = torch.randn((self.n_envs, n_steps), generator=self.generator,
                                  device=self.device, dtype=torch.float64)
        temperature = 20.0 + 10.0 * cos_factor.unsqueeze(0) + 2.0 * temp_noise

        load_base = (
            0.4
            + 0.35 * torch.exp(-((hours - 8) ** 2) / 8)
            + 0.25 * torch.exp(-((hours - 19) ** 2) / 12)
        )  # (n_steps,)
        load_noise = torch.randn((self.n_envs, n_steps), generator=self.generator,
                                  device=self.device, dtype=torch.float64)
        load_pu = torch.clamp(load_base.unsqueeze(0) + 0.05 * load_noise, 0.2, 1.5)

        return VectorizedDailyProfile(hours=hours, irradiance=irradiance,
                                       temperature=temperature, load_pu=load_pu)
