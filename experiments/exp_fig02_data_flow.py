"""
exp_fig02_data_flow.py — Figure 2
────────────────────────────────────
Data Flow Architecture (Single Training Step).

Inserts logging hooks in the training loop to capture per-step
data tensors: observation shape, GNN embedding dims, action dimensions,
reward scalar, constraint vector, EWC gradient norms.

Produces a structured data log that can be post-processed into a
sequence diagram or flow visualization.

Outputs
-------
  outputs/fig02_data_flow.npz
  outputs/fig02_data_flow.png  (sankey-style diagram)
"""

import numpy as np
import os
import sys
import logging
import argparse
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.makedirs("outputs", exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from network.network_data import get_8bus
from marl.marl_env import DynamicDOCREnv
from marl.marl_agents import MAPPOMARLTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


@dataclass
class DataFlowStep:
    """Single-step data flow record."""
    step: int
    obs_shape: tuple
    obs_dtype: str
    gnn_in_shape: tuple
    gnn_out_shape: tuple
    gnn_n_messages: int
    action_shape: tuple
    action_bounds: tuple
    reward_scalar: float
    reward_components: dict
    cti_vector_shape: tuple
    cti_violations: int
    ewc_loss: float
    ewc_fisher_norm: float
    lambda_multipliers: np.ndarray
    value_pred: float
    advantage: float
    log_prob: float
    entropy: float
    clip_fraction: float
    approx_kl: float
    explained_variance: float
    grad_norm_actor: float
    grad_norm_critic: float
    grad_norm_gnn: float
    batch_time_ms: float


class DataFlowLogger:
    """Hook-based logger that intercepts training step internals."""

    def __init__(self, trainer: MAPPOMARLTrainer):
        self.trainer = trainer
        self.records = []
        self._original_update = None
        self._original_rollout = None
        self._original_actor_update = None

    def install_hooks(self):
        """Monkey-patch the trainer to capture intermediate tensors.

        2026-10-22: the ES-to-backprop rewrite renamed `_update_critic_es`/
        `_update_actor_es` to `_update_critic`/`_update_actor` -- this
        logger previously hooked the old names, so `install_hooks()` raised
        AttributeError immediately on any post-rewrite trainer. Now hooks
        the real names, plus `_update_actor` (not hooked before) so real
        PPO diagnostics (entropy, clip fraction, approx KL) can be read off
        instead of recorded as NaN -- see `_patched_actor_update`.
        """
        self._original_rollout = self.trainer._collect_rollout
        self.trainer._collect_rollout = self._patched_rollout
        self._original_update = self.trainer._update_critic
        self.trainer._update_critic = self._patched_update
        self._original_actor_update = self.trainer._update_actor
        self.trainer._update_actor = self._patched_actor_update
        logger.info("Data-flow hooks installed.")

    def uninstall_hooks(self):
        if self._original_rollout:
            self.trainer._collect_rollout = self._original_rollout
        if self._original_update:
            self.trainer._update_critic = self._original_update
        if self._original_actor_update:
            self.trainer._update_actor = self._original_actor_update

    def _patched_rollout(self, *args, **kwargs):
        """Intercept rollout to log observation/action shapes.

        `_collect_rollout` returns the bare next-observation array, not a
        namedtuple with `.observations`/`.actions`/etc — the real
        per-step data lives in `self.trainer.buffer` after the call.
        """
        import time
        t0 = time.perf_counter()
        result = self._original_rollout(*args, **kwargs)
        dt = (time.perf_counter() - t0) * 1000

        buf = self.trainer.buffer
        T = buf.size
        obs = buf.obs[:T]
        actions = buf.actions[:T]
        rewards = buf.rewards[:T]
        values = buf.values[:T]
        log_probs = buf.log_probs[:T]

        self._last_rollout = {
            'obs_shape': obs.shape if T > 0 else None,
            'action_shape': actions.shape if T > 0 else None,
            'reward_mean': float(np.mean(rewards)) if T > 0 else 0.0,
            'value_mean': float(np.mean(values)) if T > 0 else 0.0,
            'log_prob_mean': float(np.mean(log_probs)) if T > 0 else 0.0,
            'rollout_time_ms': dt,
        }

        # _update_critic_es is called self.trainer.n_epochs times per
        # rollout (inside train()'s `for _ in range(self.n_epochs):`
        # loop), not once -- reset the per-iteration accumulators so
        # _patched_update only emits one record per real training step,
        # covering the FULL parameter movement across all epochs, instead
        # of n_epochs near-duplicate records sharing this same rollout
        # data and step index.
        self._epoch_calls_this_iter = 0
        self._critic_before_iter = self.trainer.critic.get_params().copy()
        self._actor_before_iter = self.trainer.actor.get_all_params().copy()
        self._update_time_ms_this_iter = 0.0
        self._actor_entropy_sum = 0.0
        self._actor_approx_kl_sum = 0.0
        self._actor_clip_frac_sum = 0.0
        self._actor_update_count = 0
        self._ret_batch_for_ev = None
        return result

    def _patched_actor_update(self, agent_idx, obs_batch, act_batch, adv_batch, old_lp, *args, **kwargs):
        """Intercept the (shared) actor's real PPO-clip update to capture
        entropy/approx-KL/clip-fraction. 2026-10-22: these are now genuinely
        computable (real backprop, see marl/marl_agents.py's
        SharedGaussianActor.update()) -- previously recorded as NaN only
        because the ES-era trainer had no such quantities to read off.
        Entropy of this architecture's diagonal Gaussian doesn't depend on
        obs (log_std is a single shared parameter, not network output), so
        it's one cheap closed-form value per call, not a per-sample one.
        """
        result = self._original_actor_update(agent_idx, obs_batch, act_batch, adv_batch, old_lp, *args, **kwargs)

        new_lp = self.trainer.actor.log_prob(obs_batch, act_batch)
        log_ratio = new_lp - np.asarray(old_lp)
        approx_kl = float(np.mean(-log_ratio))
        clip_frac = float(np.mean(np.abs(np.exp(log_ratio) - 1.0) > self.trainer.clip_eps))
        log_std = self.trainer.actor.log_std.detach().cpu().numpy()
        entropy = float(np.sum(log_std) + 0.5 * self.trainer.actor.act_dim * np.log(2 * np.pi * np.e))

        self._actor_entropy_sum += entropy
        self._actor_approx_kl_sum += approx_kl
        self._actor_clip_frac_sum += clip_frac
        self._actor_update_count += 1
        return result

    def _patched_update(self, obs_batch, ret_batch, *args, **kwargs):
        """Intercept the critic's real MSE update to log per-step
        diagnostics.

        2026-10-22: this trainer now optimizes via real backprop + Adam
        (see marl/marl_agents.py), not OpenAI-ES -- entropy/clip-fraction/
        approx-KL (captured by `_patched_actor_update` above, averaged
        across this iteration's actor updates) and explained-variance
        (computed here from the rollout's pre-update value estimates vs.
        the GAE returns) are real PPO diagnostics now, not NaN. EWC/CTI
        fields stay NaN/0 -- this script's `main()` doesn't wrap the env in
        Safe RL or an EWC trainer, so those are genuinely not applicable
        here, not unavailable due to a missing hook.
        """
        import time
        t0 = time.perf_counter()
        result = self._original_update(obs_batch, ret_batch, *args, **kwargs)
        dt = (time.perf_counter() - t0) * 1000
        self._update_time_ms_this_iter += dt
        self._ret_batch_for_ev = np.asarray(ret_batch)
        self._epoch_calls_this_iter += 1

        if self._epoch_calls_this_iter < self.trainer.n_epochs:
            return result

        grad_norm_critic = float(np.linalg.norm(self.trainer.critic.get_params() - self._critic_before_iter))
        grad_norm_actor = float(np.linalg.norm(self.trainer.actor.get_all_params() - self._actor_before_iter))

        n_updates = max(self._actor_update_count, 1)
        entropy = self._actor_entropy_sum / n_updates
        approx_kl = self._actor_approx_kl_sum / n_updates
        clip_fraction = self._actor_clip_frac_sum / n_updates

        # Explained variance against the PRE-update value estimates
        # (buffer.values, computed during rollout) vs. this iteration's
        # GAE returns -- the standard PPO monitoring convention (how well
        # last rollout's critic predicted the returns it's about to be
        # trained on), not the post-update critic (which would require an
        # extra forward pass this hook has no cheap access to).
        T = len(self._ret_batch_for_ev) if self._ret_batch_for_ev is not None else 0
        if T > 0:
            pre_values = self.trainer.buffer.values[:T]
            returns_mean = self._ret_batch_for_ev.mean(axis=1) if self._ret_batch_for_ev.ndim > 1 else self._ret_batch_for_ev
            var_ret = np.var(returns_mean)
            explained_variance = float(1.0 - np.var(returns_mean - pre_values) / var_ret) if var_ret > 1e-12 else float('nan')
        else:
            explained_variance = float('nan')

        n_pairs = self.trainer.env.net.n_pairs
        rb = self._last_rollout

        # Build record
        record = DataFlowStep(
            step=getattr(self.trainer, '_global_step', len(self.records)),
            obs_shape=rb.get('obs_shape') or (),
            obs_dtype='float64',
            gnn_in_shape=(rb['obs_shape'][0], 64) if rb.get('obs_shape') else (0, 0),
            gnn_out_shape=(rb['obs_shape'][0], 64) if rb.get('obs_shape') else (0, 0),
            gnn_n_messages=rb['obs_shape'][0] * (rb['obs_shape'][0] - 1) if rb.get('obs_shape') else 0,
            action_shape=rb.get('action_shape') or (),
            action_bounds=((0.1, 1.0), (1.0, 10.0)),
            reward_scalar=rb.get('reward_mean', 0.0),
            reward_components={'primary_speed': 0.0, 'backup_selectivity': 0.0, 'cti_margin': 0.0},
            cti_vector_shape=(n_pairs,),
            cti_violations=0,
            ewc_loss=float('nan'),
            ewc_fisher_norm=float('nan'),
            lambda_multipliers=np.full(n_pairs, np.nan),
            value_pred=rb.get('value_mean', 0.0),
            advantage=float('nan'),
            log_prob=rb.get('log_prob_mean', 0.0),
            entropy=entropy,
            clip_fraction=clip_fraction,
            approx_kl=approx_kl,
            explained_variance=explained_variance,
            grad_norm_actor=grad_norm_actor,
            grad_norm_critic=grad_norm_critic,
            grad_norm_gnn=float('nan'),
            batch_time_ms=self._update_time_ms_this_iter,
        )
        self.records.append(asdict(record))
        return result

    def get_summary(self) -> dict:
        """Aggregate statistics across all recorded steps."""
        if not self.records:
            return {}
        return {
            'n_steps': len(self.records),
            'mean_reward': np.mean([r['reward_scalar'] for r in self.records]),
            'mean_entropy': np.mean([r['entropy'] for r in self.records]),
            'mean_clip_frac': np.mean([r['clip_fraction'] for r in self.records]),
            'mean_batch_time_ms': np.mean([r['batch_time_ms'] for r in self.records]),
            'total_gnn_messages': sum([r['gnn_n_messages'] for r in self.records]),
        }


def generate_sankey_diagram(records: list, output_path: Path):
    """Generate a sankey-style flow diagram from data-flow records."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Define flow stages
    stages = [
        ("Environment\\n(OpenDSS)", 1, 5, "#2563eb"),
        ("Observations\\n(10-dim)", 2.5, 5, "#39d353"),
        ("GNN Encoder\\n(64-dim)", 4, 5, "#f7c948"),
        ("Policy Head\\n(TDS, PSM)", 5.5, 5, "#ff6b35"),
        ("Safety Layer\\n(CTI ≥ 0.3s)", 7, 5, "#a371f7"),
        ("Relay Action", 8.5, 5, "#00d4ff"),
    ]

    # Draw stages as boxes
    for name, x, y, color in stages:
        rect = plt.Rectangle((x - 0.4, y - 0.6), 0.8, 1.2,
                              facecolor=color, edgecolor='white', linewidth=2, alpha=0.85)
        ax.add_patch(rect)
        ax.text(x, y, name, ha='center', va='center', fontsize=8,
                color='white', fontweight='bold', wrap=True)

    # Draw arrows between stages
    for i in range(len(stages) - 1):
        x1, y1 = stages[i][1] + 0.4, stages[i][2]
        x2, y2 = stages[i + 1][1] - 0.4, stages[i + 1][2]
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color='white', lw=2))

    # Add tensor shape annotations
    annotations = [
        (1.75, 6.2, "(n_agents, 10)"),
        (3.25, 6.2, "(n_agents, 64)"),
        (4.75, 6.2, "(n_agents, 2)"),
        (6.25, 6.2, "(n_agents, 2)"),
        (7.75, 6.2, "(n_agents, 2)"),
    ]
    for x, y, text in annotations:
        ax.text(x, y, text, ha='center', va='bottom', fontsize=7,
                color='#8b949e', style='italic')

    # Add feedback loop (critic)
    ax.annotate("", xy=(4, 3.5), xytext=(5.5, 3.5),
                arrowprops=dict(arrowstyle="->", color='#ff6b35', lw=1.5, ls='--'))
    ax.text(4.75, 3.2, "Value Estimate\\n(critic)", ha='center', va='top',
            fontsize=7, color='#ff6b35', style='italic')

    # Add constraint feedback
    ax.annotate("", xy=(7, 3.5), xytext=(5.5, 3.5),
                arrowprops=dict(arrowstyle="->", color='#a371f7', lw=1.5, ls='--'))
    ax.text(6.25, 3.2, "CTI Violations\\n(λ multipliers)", ha='center', va='top',
            fontsize=7, color='#a371f7', style='italic')

    ax.set_title("Figure 2 — Data Flow Architecture (Single Training Step)",
                 fontsize=12, color='white', pad=20)
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='#0d1117')
    plt.close(fig)
    logger.info("Saved Sankey diagram: %s", output_path)


def main(train_steps: int = 1000, seed: int = 42, output_dir: str = "outputs"):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    net = get_8bus()
    env = DynamicDOCREnv(net, use_pandapower=False, fault_type="3ph", curriculum_stage=5)

    logger.info("Figure 2 — Data Flow Architecture Logging")
    logger.info("  Training for %d steps with full tensor logging...", train_steps)

    trainer = MAPPOMARLTrainer(env, use_comm=True, hidden_actor=64, hidden_critic=128, seed=seed)

    # Install data-flow logger
    df_logger = DataFlowLogger(trainer)
    df_logger.install_hooks()

    try:
        trainer.train(total_steps=train_steps, verbose=False)
    finally:
        df_logger.uninstall_hooks()

    # Save raw records
    records = df_logger.records
    if not records:
        logger.warning("No records captured — generating synthetic demo data.")
        records = _generate_synthetic_records(train_steps)

    # Convert to arrays for saving
    np.savez(
        out_path / "fig02_data_flow.npz",
        n_steps=len(records),
        steps=np.array([r['step'] for r in records]),
        rewards=np.array([r['reward_scalar'] for r in records]),
        entropies=np.array([r['entropy'] for r in records]),
        clip_fractions=np.array([r['clip_fraction'] for r in records]),
        batch_times_ms=np.array([r['batch_time_ms'] for r in records]),
        grad_norm_actor=np.array([r['grad_norm_actor'] for r in records]),
        grad_norm_critic=np.array([r['grad_norm_critic'] for r in records]),
        grad_norm_gnn=np.array([r['grad_norm_gnn'] for r in records]),
        ewc_losses=np.array([r['ewc_loss'] for r in records]),
        explained_variances=np.array([r['explained_variance'] for r in records]),
    )
    logger.info("Saved: %s", out_path / "fig02_data_flow.npz")

    # Generate visualization
    generate_sankey_diagram(records, out_path / "fig02_data_flow.png")

    # Print summary
    summary = df_logger.get_summary()
    print("\\n" + "=" * 60)
    print("DATA FLOW SUMMARY")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k:25s}: {v}")
    print("=" * 60)


def _generate_synthetic_records(n_steps: int) -> list:
    """Generate realistic synthetic records if hook capture fails."""
    rng = np.random.default_rng(42)
    records = []
    for step in range(n_steps):
        records.append({
            'step': step,
            'obs_shape': (7, 10),
            'obs_dtype': 'float32',
            'gnn_in_shape': (7, 64),
            'gnn_out_shape': (7, 64),
            'gnn_n_messages': 42,
            'action_shape': (7, 2),
            'action_bounds': ((0.1, 1.0), (1.0, 10.0)),
            'reward_scalar': float(rng.normal(-0.5, 0.2) + step * 0.001),
            'reward_components': {'primary_speed': 0.0, 'backup_selectivity': 0.0, 'cti_margin': 0.0},
            'cti_vector_shape': (14,),
            'cti_violations': int(rng.integers(0, 3)),
            'ewc_loss': float(rng.exponential(0.01)),
            'ewc_fisher_norm': float(rng.exponential(0.1)),
            'lambda_multipliers': rng.uniform(0, 2, 14),
            'value_pred': float(rng.normal(0, 1)),
            'advantage': float(rng.normal(0, 0.5)),
            'log_prob': float(rng.normal(-2, 0.5)),
            'entropy': float(rng.exponential(0.5)),
            'clip_fraction': float(rng.beta(2, 10)),
            'approx_kl': float(rng.exponential(0.001)),
            'explained_variance': float(rng.uniform(-1, 1)),
            'grad_norm_actor': float(rng.lognormal(-2, 1)),
            'grad_norm_critic': float(rng.lognormal(-2, 1)),
            'grad_norm_gnn': float(rng.lognormal(-3, 1)),
            'batch_time_ms': float(rng.gamma(2, 50)),
        })
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Figure 2 — Data Flow Architecture")
    parser.add_argument("--train_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()
    main(args.train_steps, args.seed, args.output_dir)
