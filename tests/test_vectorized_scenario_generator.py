"""
tests/test_vectorized_scenario_generator.py
---------------------------------------------
Verification for pv_model/vectorized_scenario_generator.py (Phase V2 of the
GPU-vectorized-environment plan).

Statistical, not bit-identical, equivalence is the correctness bar here
(same precedent as the metaheuristics vectorization in the completed
torch-port rewrite) -- the Markov chain's RNG draw order necessarily
changes once draws happen across all n_envs at once instead of one
environment's full sequence at a time. Verified via: (1) the cloud-state
chain's empirical marginal distribution matching the transition matrix's
analytical stationary distribution, (2) irradiance/temperature/load_pu
summary statistics (mean/std at fixed hours) matching many independent
single-env realizations, (3) basic physical sanity (irradiance zero at
night, load_pu within its clipped bounds).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from pv_model.scenario_generator import MarkovCloudModel
from pv_model.vectorized_scenario_generator import VectorizedMarkovCloudModel


def _stationary_distribution(P: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(eigvals - 1))
    stat = np.real(eigvecs[:, idx])
    return stat / stat.sum()


def test_cloud_state_marginal_matches_stationary_distribution():
    """The batched Markov chain's empirical state-visitation frequency over
    many steps must match the transition matrix's true stationary
    distribution -- a chain-vs-math check independent of any reference
    implementation."""
    vm = VectorizedMarkovCloudModel(n_envs=50, device="cpu", seed=7)
    states = vm.generate_sequence(20000, initial_state=0)
    unique, counts = np.unique(states.numpy(), return_counts=True)
    empirical = np.zeros(3)
    empirical[unique] = counts / counts.sum()

    expected = _stationary_distribution(vm.P.numpy())
    assert np.max(np.abs(empirical - expected)) < 0.02, (
        f"empirical {empirical} vs analytical stationary {expected}"
    )


def test_daily_profile_statistics_match_single_env_reference():
    """Mean/std of irradiance, temperature, and load_pu at fixed hours,
    aggregated over many independent single-env realizations, must match
    one large batched run's per-environment distribution."""
    n_samples = 1500
    old_irr_noon, old_load_8am, old_temp_noon = [], [], []
    for seed in range(n_samples):
        m = MarkovCloudModel(seed=seed)
        p = m.generate_daily_profile()
        noon_idx = len(p.hours) // 2
        morn_idx = int(8 / (p.hours[1] - p.hours[0]))
        old_irr_noon.append(p.irradiance[noon_idx])
        old_load_8am.append(p.load_pu[morn_idx])
        old_temp_noon.append(p.temperature[noon_idx])
    old_irr_noon = np.array(old_irr_noon)
    old_load_8am = np.array(old_load_8am)
    old_temp_noon = np.array(old_temp_noon)

    vm = VectorizedMarkovCloudModel(n_envs=n_samples, device="cpu", seed=123)
    vp = vm.generate_daily_profile()
    noon_idx = len(vp.hours) // 2
    morn_idx = int(8 / (vp.hours[1] - vp.hours[0]).item())
    new_irr_noon = vp.irradiance[:, noon_idx].numpy()
    new_load_8am = vp.load_pu[:, morn_idx].numpy()
    new_temp_noon = vp.temperature[:, noon_idx].numpy()

    # Loose tolerances -- comparing sample statistics of two independent
    # random processes, not a deterministic equivalence.
    assert abs(old_irr_noon.mean() - new_irr_noon.mean()) < 25, "irradiance mean diverged"
    assert abs(old_irr_noon.std() - new_irr_noon.std()) < 25, "irradiance std diverged"
    assert abs(old_load_8am.mean() - new_load_8am.mean()) < 0.01, "load_pu mean diverged"
    assert abs(old_load_8am.std() - new_load_8am.std()) < 0.01, "load_pu std diverged"
    assert abs(old_temp_noon.mean() - new_temp_noon.mean()) < 0.5, "temperature mean diverged"
    assert abs(old_temp_noon.std() - new_temp_noon.std()) < 0.3, "temperature std diverged"


def test_profile_physical_sanity():
    vm = VectorizedMarkovCloudModel(n_envs=20, device="cpu", seed=5)
    vp = vm.generate_daily_profile()

    assert torch.all(vp.irradiance[:, 0] == 0.0), "irradiance should be exactly 0 at midnight"
    noon_idx = len(vp.hours) // 2
    assert torch.all(vp.irradiance[:, noon_idx] > 0.0), "irradiance should be positive at noon"
    assert torch.all(vp.load_pu >= 0.2) and torch.all(vp.load_pu <= 1.5), "load_pu outside its clipped bounds"
    assert vp.hours.shape[0] == vp.irradiance.shape[1] == vp.load_pu.shape[1] == vp.temperature.shape[1]
    assert vp.irradiance.shape[0] == vp.load_pu.shape[0] == vp.temperature.shape[0] == 20


if __name__ == "__main__":
    test_cloud_state_marginal_matches_stationary_distribution()
    test_daily_profile_statistics_match_single_env_reference()
    test_profile_physical_sanity()
    print("All vectorized scenario generator tests passed.")
