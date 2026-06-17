"""Generate the two branded Monolith Research validation charts.

    py -3.14 examples/make_plots.py

Outputs (Monolith palette: Ink #0E0C14 / Plum #A78AA3 / Paper #F3EFE8):

  assets/dsr_deflation_curve.png
      The Deflated Sharpe Ratio as a function of the number of trials searched.
      A backtest that looks great after one trial can fall below the gate once
      you account for how many configurations you actually tried. This is the
      single most important chart in the library: it shows why an in-sample
      Sharpe is not evidence on its own.

  assets/pbo_histogram.png
      The CSCV logit distribution behind a Probability of Backtest Overfitting
      number, for a noise book (no edge) vs a book with one genuine signal.

Everything here is synthetic. No proprietary returns, no licensed data.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations

from monolith_validation import (
    INK,
    PAPER,
    PLUM,
    PLUM_DARK,
    deflated_sharpe_ratio,
)

ASSETS = Path(__file__).resolve().parents[1] / "assets"
ASSETS.mkdir(exist_ok=True)


def _brand(ax):
    ax.set_facecolor(PAPER)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(INK)
        ax.spines[spine].set_alpha(0.6)
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(True, color=INK, alpha=0.08, linewidth=0.8)


def _sigma(fig):
    # The Σ mark, top-left, in plum — the Monolith signature.
    fig.text(
        0.013, 0.965, "Σ", color=PLUM, fontsize=30, fontstyle="italic",
        fontfamily="serif", va="top", ha="left",
    )
    fig.text(
        0.052, 0.952, "MONOLITH RESEARCH", color=INK, fontsize=8.5,
        fontfamily="monospace", va="top", ha="left", alpha=0.85,
    )
    fig.text(
        0.052, 0.918, "EVIDENCE OVER NARRATIVE", color=PLUM_DARK, fontsize=6.5,
        fontfamily="monospace", va="top", ha="left", alpha=0.9,
    )


# --------------------------------------------------------------------------- #
# Chart 1 — DSR vs number of trials (the deflation curve)
# --------------------------------------------------------------------------- #
def deflation_curve():
    # A fixed, honest-looking in-sample result: per-period Sharpe 0.20 over a
    # 4-year daily track, with per-trial Sharpe dispersion typical of a real
    # hyperparameter search. The gate is DSR >= 0.50.
    sr_pp = 0.115
    n_obs = 252 * 4
    trial_sr_std = 0.040
    gate = 0.50

    trials = np.unique(np.round(np.logspace(0, math.log10(90000), 240)).astype(int))
    dsr = np.array([deflated_sharpe_ratio(sr_pp, n_obs, 0.0, 3.0, int(t), trial_sr_std) for t in trials])

    fig, ax = plt.subplots(figsize=(8.2, 5.0), dpi=150)
    fig.patch.set_facecolor(PAPER)
    _brand(ax)

    ax.semilogx(trials, dsr, color=INK, linewidth=2.2, solid_capstyle="round")
    ax.axhline(gate, color=PLUM, linewidth=1.6, linestyle="--")
    ax.fill_between(trials, dsr, gate, where=(dsr >= gate), color=PLUM, alpha=0.14, interpolate=True)
    ax.fill_between(trials, dsr, gate, where=(dsr < gate), color=INK, alpha=0.06, interpolate=True)

    # crossing point where the deflated edge dies
    below = np.where(dsr < gate)[0]
    if below.size:
        cross = trials[below[0]]
        ax.axvline(cross, color=PLUM_DARK, linewidth=1.0, alpha=0.7, linestyle=":")
        ax.annotate(
            f"edge dies at ~{cross:,} trials",
            xy=(cross, gate), xytext=(cross * 1.4, gate + 0.16),
            color=PLUM_DARK, fontsize=8.5,
            arrowprops=dict(arrowstyle="->", color=PLUM_DARK, lw=1.0),
        )

    ax.text(1.3, gate + 0.012, "GATE  DSR ≥ 0.50", color=PLUM_DARK, fontsize=8.5, va="bottom")
    ax.set_xlabel("number of trials / configurations searched (N)", color=INK, fontsize=10)
    ax.set_ylabel("Deflated Sharpe Ratio", color=INK, fontsize=10)
    ax.set_title(
        "Deflation curve: the same in-sample Sharpe, priced for the search behind it",
        color=INK, fontsize=11.5, pad=26, loc="left",
    )
    ax.set_ylim(0.0, 1.02)
    ax.set_xlim(1, trials.max())
    _sigma(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out = ASSETS / "dsr_deflation_curve.png"
    fig.savefig(out, facecolor=PAPER)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# Chart 2 — CSCV PBO logit histogram (noise book vs genuine edge)
# --------------------------------------------------------------------------- #
def _cscv_logits(M: np.ndarray, n_splits: int = 12) -> np.ndarray:
    """Per-split CSCV relative-rank logits (the distribution behind PBO)."""
    T, N = M.shape
    block_len = T // n_splits
    M = M[: block_len * n_splits]
    blocks = [np.arange(i * block_len, (i + 1) * block_len) for i in range(n_splits)]

    def col_sharpe(rows):
        mean = rows.mean(axis=0)
        std = rows.std(axis=0, ddof=0)
        out = np.zeros_like(mean)
        nz = std > 0
        out[nz] = mean[nz] / std[nz]
        return out

    logits = []
    for is_ids in combinations(range(n_splits), n_splits // 2):
        is_set = set(is_ids)
        is_rows = np.concatenate([blocks[i] for i in is_ids])
        oos_rows = np.concatenate([blocks[i] for i in range(n_splits) if i not in is_set])
        is_perf = col_sharpe(M[is_rows])
        oos_perf = col_sharpe(M[oos_rows])
        n_star = int(np.argmax(is_perf))
        order = np.argsort(oos_perf, kind="mergesort")
        ranks = np.empty(N)
        ranks[order] = np.arange(1, N + 1)
        omega = min(max(ranks[n_star] / (N + 1.0), 1e-12), 1.0 - 1e-12)
        logits.append(math.log(omega / (1.0 - omega)))
    return np.asarray(logits)


def pbo_histogram():
    rng = np.random.default_rng(11)
    T, M = 360, 40

    noise = rng.standard_normal((T, M)) * 0.01
    l_noise = _cscv_logits(noise, n_splits=12)
    pbo_noise = float(np.mean(l_noise < 0))

    edge_book = rng.standard_normal((T, M)) * 0.01
    edge_book[:, 0] = 0.025 + rng.standard_normal(T) * 0.004  # one persistent real edge
    l_edge = _cscv_logits(edge_book, n_splits=12)
    pbo_edge = float(np.mean(l_edge < 0))

    fig, ax = plt.subplots(figsize=(8.2, 5.0), dpi=150)
    fig.patch.set_facecolor(PAPER)
    _brand(ax)

    bins = np.linspace(min(l_noise.min(), l_edge.min()), max(l_noise.max(), l_edge.max()), 36)
    ax.hist(l_noise, bins=bins, color=INK, alpha=0.55, label=f"noise book — PBO {pbo_noise:.2f}")
    ax.hist(l_edge, bins=bins, color=PLUM, alpha=0.70, label=f"one genuine edge — PBO {pbo_edge:.2f}")
    ax.axvline(0.0, color=PLUM_DARK, linewidth=1.4, linestyle="--")
    ax.text(-3.9, ax.get_ylim()[1] * 0.92, "logit < 0  →  IS winner lost OOS",
            color=PLUM_DARK, fontsize=8, ha="left")

    ax.set_xlabel("CSCV relative-rank logit  (per IS/OOS split)", color=INK, fontsize=10)
    ax.set_ylabel("count of splits", color=INK, fontsize=10)
    ax.set_title(
        "Probability of Backtest Overfitting: the CSCV split distribution",
        color=INK, fontsize=11.5, pad=26, loc="left",
    )
    leg = ax.legend(frameon=False, fontsize=9, loc="upper right", bbox_to_anchor=(1.0, 0.86))
    for txt in leg.get_texts():
        txt.set_color(INK)
    _sigma(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out = ASSETS / "pbo_histogram.png"
    fig.savefig(out, facecolor=PAPER)
    plt.close(fig)
    return out


if __name__ == "__main__":
    p1 = deflation_curve()
    p2 = pbo_histogram()
    print(f"wrote {p1}")
    print(f"wrote {p2}")
