"""
_synthetic_label.py
────────────────────
Shared helper for CLI adapter scripts in this directory that plot hand-typed
literal number arrays instead of output from a trained model or a real
experiment run (see experiments/GAP_ANALYSIS.md for how this was discovered).

Import mark_synthetic() and call it on the figure right before savefig() so
every such plot is unambiguously watermarked, instead of looking identical
to a figure produced by an actual training/evaluation run.
"""

import sys


def mark_synthetic(fig, note: str = "") -> None:
    """Watermark a matplotlib figure as illustrative/synthetic data.

    Also prints a console warning, since the on-disk PDF/PNG is often
    consumed (embedded in a paper draft, copied to results/) without anyone
    re-reading this script's source first.
    """
    label = "SYNTHETIC DATA — NOT FROM A TRAINED MODEL OR REAL EXPERIMENT"
    if note:
        label += f" ({note})"
    fig.text(
        0.5, 0.5, label,
        transform=fig.transFigure,
        fontsize=13, color="red", alpha=0.35,
        ha="center", va="center", rotation=30, zorder=1000,
    )
    print(f"[WARNING] {label}", file=sys.stderr)
