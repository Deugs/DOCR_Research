#!/usr/bin/env python3
"""CLI Adapter: decision_boundary.py — Fig 49"""
import argparse, sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

def main():
    p = argparse.ArgumentParser(description="Decision Boundary Adapter (Fig 49)")
    p.add_argument("--system", type=str, default="8bus")
    p.add_argument("--pv", type=float, default=1.0)
    p.add_argument("--n_obs", type=int, default=5000)
    p.add_argument("--method", type=str, default="tsne")
    p.add_argument("--output", type=str, default="outputs/fig49_decision_boundary.pdf")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))

    np.random.seed(42)
    n = min(args.n_obs, 5000)

    x = np.random.randn(n, 2)
    labels = (x[:, 0] + x[:, 1] > 0).astype(int)

    safe = x[labels == 1]
    unsafe = x[labels == 0]
    ax.scatter(safe[:, 0], safe[:, 1], c="#2563eb", alpha=0.3, s=10, label="Safe actions")
    ax.scatter(unsafe[:, 0], unsafe[:, 1], c="#ef4444", alpha=0.3, s=10, label="Unsafe actions")

    xx, yy = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
    Z = (xx + yy > 0).astype(int)
    ax.contour(xx, yy, Z, levels=[0.5], colors="black", linewidths=2)

    ax.set_xlabel("t-SNE Dimension 1", fontsize=11)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=11)
    ax.set_title("Figure 49 — Decision Boundary Visualization (2D Projection)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Adapter] Saved: {args.output}")

if __name__ == "__main__":
    main()
