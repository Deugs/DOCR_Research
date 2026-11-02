"""
vectorized_topology_manager.py
───────────────────────────────
Batched (n_envs,)-wide counterpart of `topology_manager.py`'s `TopologyManager`,
for the GPU-vectorized-environment redesign (Phase V1 of the vectorized-env
plan -- see plan file for full context).

Scope: ports only the methods actually called from the real training hot
path (`environment/dynamic_network.py`'s `step()`/`reset()`), confirmed via
grep across the whole repo: `apply_multiple_outages()` and
`get_outage_factor()`. `TopologyManager`'s other methods
(`reconfigure_for_radial`, `get_affected_buses`, `get_fault_current_modifiers`,
`random_n1_contingency`, `get_topology_description`, the precomputed
`n1_contingencies` list) have zero real callers anywhere in this repo
outside the class itself -- a deliberate scope decision, not an oversight,
matching this project's established practice of not spending effort
vectorizing dead code (see e.g. `network/relay_physics.py`'s module
docstring for the same reasoning applied elsewhere in this rewrite).

State is a genuine torch boolean tensor (not a NumPy-in/NumPy-out wrapper
like the single-environment-path classes elsewhere in this project) --
unlike those classes, this one's whole purpose is to stay GPU-resident
across a batch of environments, so there is no "cheap, stays NumPy"
argument here the way there was for e.g. `ActionProjector`'s tiny
single-environment arrays.
"""

from __future__ import annotations

from typing import Optional


def _default_device():
    """Reuses the project's DOCR_USE_GPU=1 opt-in policy (see
    marl.marl_agents's module docstring) rather than inventing a second
    device-selection mechanism. Falls back to an inline equivalent if the
    marl package isn't importable from this context."""
    import torch
    try:
        from marl.marl_agents import _default_device as _marl_default_device
        return _marl_default_device()
    except ImportError:
        import os
        cuda_present = torch.cuda.is_available()
        gpu_available = cuda_present and os.environ.get("DOCR_USE_GPU") == "1"
        return torch.device("cuda") if gpu_available else torch.device("cpu")


class VectorizedTopologyManager:
    """Batched topology state for `n_envs` parallel environments sharing one
    network's line/switch count (not necessarily sharing the same active
    outages -- confirmed during planning that N-1 contingencies are
    randomized per-episode in the current design, so different environments
    in a batch will typically have different active outages).

    line_status/switch_status: (n_envs, n_lines)/(n_envs, n_switches) bool
    tensors, True = in service -- same convention as TopologyState's
    dict values, just batched.
    """

    def __init__(self, n_envs: int, n_buses: int, n_lines: int,
                 n_switches: int = 0, device=None):
        self.n_envs = n_envs
        self.n_buses = n_buses
        self.n_lines = n_lines
        self.n_switches = n_switches
        self.device = device if device is not None else _default_device()
        self.reset()

    def reset(self, env_mask: Optional["torch.Tensor"] = None) -> None:
        """Reset to base topology (all lines/switches in service).
        env_mask: optional (n_envs,) boolean tensor selecting which
        environments to reset (None = all) -- for the common case where
        only a subset of a batch finishes an episode and needs a fresh
        topology draw while others continue (see plan: per-env termination
        is NOT expected given fixed max_steps and synchronized batch
        resets, but this is cheap to support regardless and keeps the API
        symmetric with reset()'s single-env counterpart, which every env
        calls independently today)."""
        import torch
        if env_mask is None:
            self.line_status = torch.ones((self.n_envs, self.n_lines), dtype=torch.bool, device=self.device)
            self.switch_status = torch.ones((self.n_envs, self.n_switches), dtype=torch.bool, device=self.device)
        else:
            self.line_status[env_mask] = True
            self.switch_status[env_mask] = True

    def apply_multiple_outages(
        self,
        line_outage_mask: Optional["torch.Tensor"] = None,
        switch_outage_mask: Optional["torch.Tensor"] = None,
    ):
        """Accumulate outages on top of whatever's already active for each
        environment -- same semantic as the single-env
        `TopologyManager.apply_multiple_outages()` ("accumulating on top of
        whatever outages are already active in current_state (call reset()
        first if a clean base topology is wanted)"), just batched.

        line_outage_mask/switch_outage_mask: (n_envs, n_lines)/
        (n_envs, n_switches) boolean tensors, True = outage this element
        for this environment as part of this call. None = no change to
        that dimension (matches the original's None-means-no-outages-this-
        call default).

        Returns (line_status, switch_status) after the update, matching the
        single-env version's "returns the new TopologyState" convention.
        """
        if line_outage_mask is not None:
            self.line_status = self.line_status & ~line_outage_mask
        if switch_outage_mask is not None:
            self.switch_status = self.switch_status & ~switch_outage_mask
        return self.line_status, self.switch_status

    def get_outage_factor(
        self,
        line_status: Optional["torch.Tensor"] = None,
        switch_status: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        """Per-environment outage factor for fault-current scaling, exactly
        the single-env version's formula (`exp(-0.1 * total_outages)`,
        clamped to [0.5, 1.0]) evaluated for every environment at once.
        Returns an (n_envs,) tensor. line_status/switch_status default to
        self.line_status/self.switch_status (current state) if not given,
        matching the single-env version's `topology_state or
        self.current_state` default.
        """
        import torch
        ls = line_status if line_status is not None else self.line_status
        ss = switch_status if switch_status is not None else self.switch_status

        n_line_outages = (~ls).sum(dim=1).to(torch.float64)
        n_switch_outages = (~ss).sum(dim=1).to(torch.float64)
        total_outages = n_line_outages + n_switch_outages

        max_outages = self.n_lines + self.n_switches
        if max_outages == 0:
            return torch.ones(self.n_envs, dtype=torch.float64, device=self.device)

        factor = torch.exp(-0.1 * total_outages)
        return torch.clamp(factor, min=0.5)
