"""
Percept visualization.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
from matplotlib.figure import Figure
import logging

logger = logging.getLogger(__name__)

# Consistent colour palette across all percept plots
PERCEPT_COLORS = {
    "face": "#4C72B0",    # blue
    "house": "#DD8452",   # orange
    "mixed": "#AAAAAA",   # grey
}
TRANSITION_COLORS = {
    "jump_induced": "#D62728",   # red
    "spontaneous": "#2CA02C",    # green
}


def plot_percept_timeline(
    df: pd.DataFrame,
    trial: int = None,
    show_reported: bool = True,
    show_inferred: bool = True,
) -> Figure:
    """Plot smoothed percept states over time with fixpoint jumps and transitions.

    Requires that :func:`~bret.features.temporal.label_percept_transitions` has
    been run, adding ``InferredPerceptSmoothed``, ``PerceptTransition``, and
    ``TransitionType`` columns.  Falls back gracefully to ``InferredPercept`` if
    the smoothed column is absent.

    Args:
        df: Percept DataFrame (one row per sample), optionally filtered to one
            trial.  Must contain ``Timestamp``, ``Trial``, ``InferredPercept``,
            ``FixpointSection``.
        trial: If provided, plot only this trial.  If None and the DataFrame
            contains multiple trials, each trial is plotted on a separate
            sub-panel stacked vertically.
        show_reported: Overlay ``ReportedPercept`` as a strip at the top of
            each panel (only when the column is present).
        show_inferred: Show the smoothed inferred percept band (default True).

    Returns:
        Matplotlib Figure.
    """
    percept_col = "InferredPerceptSmoothed" if "InferredPerceptSmoothed" in df.columns else "InferredPercept"

    if trial is not None:
        data = df[df["Trial"] == trial].copy()
        trials = [trial]
    else:
        data = df.copy()
        trials = sorted(data["Trial"].dropna().unique())

    n_panels = len(trials)
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 2.5 * n_panels), squeeze=False)

    for ax, t in zip(axes[:, 0], trials):
        grp = data[data["Trial"] == t].sort_values("Timestamp")
        if grp.empty:
            ax.set_visible(False)
            continue

        ts = grp["Timestamp"].to_numpy(dtype=float)

        # --- inferred percept coloured bands ---
        if show_inferred and percept_col in grp.columns:
            percept_vals = grp[percept_col].to_numpy(dtype=str)
            _draw_percept_band(ax, ts, percept_vals, y0=0.1, y1=0.9, alpha=0.55)

        # --- faint raw InferredPercept background (only when smoothed is shown) ---
        if percept_col == "InferredPerceptSmoothed" and "InferredPercept" in grp.columns:
            raw_vals = grp["InferredPercept"].to_numpy(dtype=str)
            _draw_percept_band(ax, ts, raw_vals, y0=0.0, y1=1.0, alpha=0.12)

        # --- reported percept strip at top ---
        if show_reported and "ReportedPercept" in grp.columns:
            rep_vals = grp["ReportedPercept"].to_numpy(dtype=str)
            _draw_percept_band(ax, ts, rep_vals, y0=0.9, y1=1.0, alpha=0.85)

        # --- fixpoint jump lines ---
        fp = grp["FixpointSection"]
        jump_ts = grp.loc[fp.diff().fillna(0) != 0, "Timestamp"]
        for jt in jump_ts:
            ax.axvline(jt, color="black", linewidth=0.8, linestyle="--", alpha=0.5, zorder=3)

        # --- transition markers ---
        if "PerceptTransition" in grp.columns and "TransitionType" in grp.columns:
            trans = grp[grp["PerceptTransition"] == True]
            for _, row in trans.iterrows():
                c = TRANSITION_COLORS.get(row["TransitionType"], "grey")
                ax.plot(row["Timestamp"], 0.5, marker="v", color=c,
                        markersize=7, zorder=5, clip_on=False)

        ax.set_xlim(ts[0], ts[-1])
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel("Time within trial (s)")
        ax.set_title(f"Trial {int(t)}", fontsize=9, loc="left")
        ax.spines[["top", "right", "left"]].set_visible(False)

    # --- legend ---
    legend_handles = [
        mpatches.Patch(color=c, label=p.capitalize())
        for p, c in PERCEPT_COLORS.items()
    ] + [
        plt.Line2D([0], [0], marker="v", color="w", markerfacecolor=c,
                   markersize=8, label=lbl.replace("_", "-").capitalize())
        for lbl, c in TRANSITION_COLORS.items()
    ] + [
        plt.Line2D([0], [0], color="black", linewidth=0.8,
                   linestyle="--", label="Fixpoint jump")
    ]
    fig.legend(handles=legend_handles, loc="upper right",
               ncol=3, fontsize=8, framealpha=0.7)
    fig.suptitle(f"Percept timeline — trial{'s' if trial is None else f' {trial}'}",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    return fig


def plot_transition_heatmap(
    df: pd.DataFrame,
    bin_ms: float = 100.0,
) -> Figure:
    """Heatmap of percept state over time across all trials, with jump/transition overlays.

    X-axis: time within trial binned into ``bin_ms`` bins.
    Y-axis: one row per trial.
    Cell colour: modal smoothed percept in that bin.
    Overlaid markers: short tick at fixpoint jumps (black), dot at percept
    transitions coloured by type (red = jump-induced, green = spontaneous).

    Requires :func:`~bret.features.temporal.label_percept_transitions` output
    columns (``InferredPerceptSmoothed``, ``PerceptTransition``,
    ``TransitionType``).

    Args:
        df: Percept DataFrame with transition labels.
        bin_ms: Width of each time bin in ms (default 100).

    Returns:
        Matplotlib Figure.
    """
    from scipy.stats import mode as scipy_mode

    percept_col = "InferredPerceptSmoothed" if "InferredPerceptSmoothed" in df.columns else "InferredPercept"
    percept_to_num = {"face": 0.0, "mixed": 0.5, "house": 1.0}
    trials = sorted(df["Trial"].dropna().unique())

    # Timestamps are in seconds; convert bin_ms to seconds for array operations
    bin_s = bin_ms / 1000.0
    max_t = df.groupby("Trial")["Timestamp"].max().max()
    bin_edges = np.arange(0, max_t + bin_s, bin_s)
    n_bins = len(bin_edges) - 1

    # Grid: rows = trials, cols = bins
    grid = np.full((len(trials), n_bins), np.nan)

    for i, t in enumerate(trials):
        grp = df[df["Trial"] == t]
        ts = grp["Timestamp"].to_numpy(dtype=float)
        pv = grp[percept_col].to_numpy(dtype=str)
        for j in range(n_bins):
            mask = (ts >= bin_edges[j]) & (ts < bin_edges[j + 1])  # seconds
            vals = pv[mask]
            valid = vals[vals != "nan"]
            if len(valid) == 0:
                continue
            counts = {p: np.sum(valid == p) for p in percept_to_num}
            modal = max(counts, key=counts.get)
            grid[i, j] = percept_to_num[modal]

    # Colour map: face=blue, mixed=grey, house=orange
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap([PERCEPT_COLORS["face"], PERCEPT_COLORS["mixed"], PERCEPT_COLORS["house"]])
    norm = BoundaryNorm([-0.25, 0.25, 0.75, 1.25], cmap.N)

    fig, ax = plt.subplots(figsize=(16, max(3, len(trials) * 0.55)))
    im = ax.imshow(
        grid,
        aspect="auto",
        origin="upper",
        extent=[0, max_t, len(trials), 0],
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )

    # --- overlay fixpoint jumps and transitions ---
    for i, t in enumerate(trials):
        grp = df[df["Trial"] == t].sort_values("Timestamp")
        fp = grp["FixpointSection"]
        jump_ts = grp.loc[fp.diff().fillna(0) != 0, "Timestamp"]
        for jt in jump_ts:
            ax.plot(jt, i + 0.5, "|", color="black", markersize=9,
                    markeredgewidth=1.2, zorder=4)

        if "PerceptTransition" in grp.columns and "TransitionType" in grp.columns:
            trans = grp[grp["PerceptTransition"] == True]
            for _, row in trans.iterrows():
                c = TRANSITION_COLORS.get(row["TransitionType"], "grey")
                ax.plot(row["Timestamp"], i + 0.5, "o", color=c,
                        markersize=5, zorder=5, markeredgewidth=0.4,
                        markeredgecolor="white")

    ax.set_xlabel("Time within trial (s)")
    ax.set_ylabel("Trial")
    ax.set_yticks(np.arange(len(trials)) + 0.5)
    ax.set_yticklabels([f"T{int(t)}" for t in trials], fontsize=8)
    ax.set_title("Percept state heatmap across trials", fontsize=11)

    legend_handles = [
        mpatches.Patch(color=PERCEPT_COLORS["face"], label="Face"),
        mpatches.Patch(color=PERCEPT_COLORS["mixed"], label="Mixed"),
        mpatches.Patch(color=PERCEPT_COLORS["house"], label="House"),
        plt.Line2D([0], [0], marker="|", color="black", linewidth=0,
                   markersize=9, label="Fixpoint jump"),
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=TRANSITION_COLORS["jump_induced"],
                   markersize=7, label="Jump-induced switch"),
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=TRANSITION_COLORS["spontaneous"],
                   markersize=7, label="Spontaneous switch"),
    ]
    ax.legend(handles=legend_handles, loc="upper right",
              ncol=2, fontsize=8, framealpha=0.8, bbox_to_anchor=(1.0, 1.15))
    fig.tight_layout()
    return fig


# ── internal helper ──────────────────────────────────────────────────────────

def _draw_percept_band(
    ax: plt.Axes,
    timestamps: np.ndarray,
    percept_values: np.ndarray,
    y0: float,
    y1: float,
    alpha: float = 0.6,
) -> None:
    """Fill horizontal spans with percept-coded colours.

    Iterates over contiguous runs of the same percept label and draws one
    filled rectangle per epoch using ``fill_between``.
    """
    if len(timestamps) == 0:
        return
    prev_p = percept_values[0]
    t_start = timestamps[0]
    for i in range(1, len(timestamps)):
        p = percept_values[i]
        if p != prev_p:
            color = PERCEPT_COLORS.get(prev_p, "#CCCCCC")
            ax.fill_between([t_start, timestamps[i]], y0, y1,
                            color=color, alpha=alpha, zorder=1)
            t_start = timestamps[i]
            prev_p = p
    # final epoch
    color = PERCEPT_COLORS.get(prev_p, "#CCCCCC")
    ax.fill_between([t_start, timestamps[-1]], y0, y1,
                    color=color, alpha=alpha, zorder=1)

