
"""
exp_decision_boundary.py  —  Figure 49
──────────────────────────────────────
Decision Boundary Visualization (2D Projection).

Collects observations from test episodes, projects to 2D via t-SNE
(perplexity=30) or UMAP (n_neighbors=15), and colors points by
discrete action type:
  • Increase TDS  : ΔTDS > +0.05
  • Decrease TDS  : ΔTDS < -0.05
  • Maintain TDS  : otherwise
  (and similarly for PSM)

Outputs
-------
  outputs/fig49_decision_boundary.npz
  outputs/fig49_decision_boundary.png
"""

import numpy as np
import os
import sys
import logging
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from network.network_data import get_8bus
from marl.marl_env import DynamicDOCREnv
from marl.marl_agents import MAPPOMARLTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def collect_observations(env, trainer, n_episodes: int = 200, seed: int = 0) -> tuple:
    """Collect (obs, action) pairs from test episodes."""
    rng = np.random.default_rng(seed)
    obs_list = []
    action_list = []

    for ep in range(n_episodes):
        obs = env.reset(seed=seed + ep)
        prev_actions = None
        done = False
        step = 0
        while not done:
            actions = np.zeros((env.n_agents, env.ACT_DIM))
            for i in range(env.n_agents):
                a, _ = trainer.actor.sample(obs[i], rng)
                actions[i] = a

            # Store obs and action for each agent
            for i in range(env.n_agents):
                obs_list.append(obs[i].copy())
                action_list.append(actions[i].copy())

            obs, _, done, info = env.step(actions)
            step += 1
            if step > 1000:   # safety break
                break

    return np.array(obs_list), np.array(action_list)


def classify_actions(actions: np.ndarray, threshold: float = 0.05) -> dict:
    """
    Classify actions into discrete categories.
    Returns dict of label arrays for TDS and PSM.
    """
    # For single-step classification, we need action deltas
    # Since we don't have previous actions, classify by magnitude sign
    # TDS is typically action[:, 0], PSM is action[:, 1]
    tds = actions[:, 0]
    psm = actions[:, 1]

    tds_labels = np.where(tds > threshold, 2,
                          np.where(tds < -threshold, 0, 1))   # 0=dec, 1=maint, 2=inc
    psm_labels = np.where(psm > threshold, 2,
                          np.where(psm < -threshold, 0, 1))

    return {"TDS": tds_labels, "PSM": psm_labels}


def main(train_steps: int = 8000, n_episodes: int = 200,
         method: str = "tsne", seed: int = 42, output_dir: str = "outputs"):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    net = get_8bus()
    env = DynamicDOCREnv(net, use_pandapower=False, fault_type="3ph", curriculum_stage=5)

    logger.info("Training policy (%d steps)...", train_steps)
    trainer = MAPPOMARLTrainer(env, use_comm=False, seed=seed)
    trainer.train(total_steps=train_steps, verbose=False)

    logger.info("Collecting observations from %d test episodes...", n_episodes)
    obs_all, actions_all = collect_observations(env, trainer, n_episodes, seed)
    logger.info("Collected %d (obs, action) pairs", len(obs_all))

    # Subsample if too large for t-SNE
    max_samples = 5000
    if len(obs_all) > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(obs_all), max_samples, replace=False)
        obs_all = obs_all[idx]
        actions_all = actions_all[idx]
        logger.info("Subsampled to %d points", max_samples)

    # Validate
    if not np.all(np.isfinite(obs_all)):
        logger.warning("Non-finite observations found — replacing with zeros")
        obs_all = np.nan_to_num(obs_all, nan=0.0, posinf=5.0, neginf=-5.0)

    # Dimensionality reduction
    logger.info("Running %s projection...", method.upper())
    if method.lower() == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, perplexity=min(30, len(obs_all) - 1),
                       random_state=seed, n_iter=1000, learning_rate="auto",
                       init="pca")
    elif method.lower() == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_components=2, n_neighbors=min(15, len(obs_all) - 1),
                                random_state=seed, min_dist=0.1)
        except ImportError:
            logger.warning("UMAP not installed, falling back to t-SNE")
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, perplexity=min(30, len(obs_all) - 1),
                           random_state=seed)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'tsne' or 'umap'.")

    emb = reducer.fit_transform(obs_all)
    logger.info("Projection complete. Shape: %s", emb.shape)

    # Classify actions
    labels = classify_actions(actions_all)

    # Save
    np.savez(
        out_path / "fig49_decision_boundary.npz",
        embeddings=emb,
        obs_samples=obs_all,
        actions=actions_all,
        tds_labels=labels["TDS"],
        psm_labels=labels["PSM"],
        method=np.array([method]),
    )
    logger.info("Saved: %s", out_path / "fig49_decision_boundary.npz")

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    label_names = ["Decrease", "Maintain", "Increase"]
    colors = ["#ff6b35", "#8b949e", "#39d353"]
    markers = ["v", "o", "^"]

    for ax_idx, (ax, action_name, label_arr) in enumerate(
        zip(axes, ["TDS", "PSM"], [labels["TDS"], labels["PSM"]])):

        for cls in range(3):
            mask = label_arr == cls
            ax.scatter(emb[mask, 0], emb[mask, 1],
                       c=colors[cls], marker=markers[cls], s=15,
                       alpha=0.6, edgecolors="none", label=label_names[cls])

        ax.set_title(f"{action_name} Decision Boundary", fontsize=10)
        ax.set_xlabel("Component 1", fontsize=9)
        ax.set_ylabel("Component 2", fontsize=9)
        ax.legend(fontsize=8, title="Action", title_fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Figure 49 — Decision Boundary Visualization ({method.upper()})",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path / "fig49_decision_boundary.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path / "fig49_decision_boundary.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Figure 49 — Decision Boundary Visualization")
    p.add_argument("--train_steps", type=int, default=8000)
    p.add_argument("--n_episodes", type=int, default=200)
    p.add_argument("--method", type=str, default="tsne", choices=["tsne", "umap"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="outputs")
    args = p.parse_args()
    main(args.train_steps, args.n_episodes, args.method, args.seed, args.output_dir)
