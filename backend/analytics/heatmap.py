"""
heatmap.py
----------
Renders a player's court-space position samples into a heatmap image drawn
over a schematic court outline. Kept separate from stats_engine.py so the
dashboard can request re-renders (e.g. different colormap) without
recomputing statistics.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from backend.court.court_detector import COURT_LENGTH_M, COURT_WIDTH_M


def render_court_outline(ax, color: str = "white") -> None:
    """Draw a schematic badminton court (both singles+doubles lines) on a matplotlib Axes."""
    ax.set_xlim(-0.5, COURT_WIDTH_M + 0.5)
    ax.set_ylim(-0.5, COURT_LENGTH_M + 0.5)
    ax.set_facecolor("#1b6b3a")

    # Outer doubles boundary
    ax.plot([0, COURT_WIDTH_M, COURT_WIDTH_M, 0, 0], [0, 0, COURT_LENGTH_M, COURT_LENGTH_M, 0], color=color)
    # Net (mid-court)
    ax.plot([0, COURT_WIDTH_M], [COURT_LENGTH_M / 2, COURT_LENGTH_M / 2], color=color, linewidth=2)
    # Singles side lines (0.46m in from each doubles sideline)
    inset = 0.46
    ax.plot([inset, inset], [0, COURT_LENGTH_M], color=color, linewidth=0.7)
    ax.plot([COURT_WIDTH_M - inset, COURT_WIDTH_M - inset], [0, COURT_LENGTH_M], color=color, linewidth=0.7)
    # Short service lines (1.98m from net on each side)
    for y in (COURT_LENGTH_M / 2 - 1.98, COURT_LENGTH_M / 2 + 1.98):
        ax.plot([0, COURT_WIDTH_M], [y, y], color=color, linewidth=0.7)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")


def points_to_grid(points: List[Tuple[float, float]], grid_size: int = 40) -> np.ndarray:
    """Bin court-space points into a 2D histogram for heatmap shading."""
    if not points:
        return np.zeros((grid_size, grid_size))
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    hist, _, _ = np.histogram2d(
        ys, xs, bins=grid_size, range=[[0, COURT_LENGTH_M], [0, COURT_WIDTH_M]]
    )
    return hist


def figure_for_player(player_name: str, points: List[Tuple[float, float]]):
    """Returns a matplotlib Figure — imported lazily so headless/report code
    that never renders heatmaps doesn't pay the matplotlib import cost."""
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter

    fig, ax = plt.subplots(figsize=(4, 8))
    render_court_outline(ax)

    grid = points_to_grid(points)
    grid = gaussian_filter(grid, sigma=1.2)
    if grid.max() > 0:
        ax.imshow(
            grid,
            extent=[0, COURT_WIDTH_M, 0, COURT_LENGTH_M],
            origin="lower",
            cmap="hot",
            alpha=0.65,
        )
    ax.set_title(f"{player_name} — Court Heatmap", color="white")
    fig.patch.set_facecolor("#0e1117")
    return fig
