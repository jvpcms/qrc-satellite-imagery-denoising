"""Plotting helpers for the QRC denoising notebooks.

Everything visual lives here so the notebooks stay focused on the method.
Functions take data explicitly (no notebook globals) and either show the
figure or, for the two paper figures, also save pdf+png to a directory.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

LABELS = {"mse": "ΔMSE", "ssim": "ΔSSIM", "teng": "ΔTENG"}
BETTER = {"mse": "< 0", "ssim": "> 0", "teng": "> 0"}
BLUE, ORANGE, GREEN = "#4878a8", "#c05a2e", "#6a9a58"


def image_rows(rows, col_titles=None, suptitle=None, panel=1.5, title_fontsize=7):
    """Grid of image rows.

    rows       : list of (imgs, label) with imgs shaped (n, 64, 64), values in [0, 1]
    col_titles : optional list of n strings for the top row
    """
    n = len(rows[0][0])
    fig, axes = plt.subplots(len(rows), n, figsize=(n * panel, panel * len(rows) + 0.5),
                             squeeze=False)
    for r, (imgs, name) in enumerate(rows):
        for c in range(n):
            axes[r, c].imshow(imgs[c], cmap="gray", vmin=0, vmax=1)
            axes[r, c].axis("off")
        axes[r, 0].axis("on")
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        axes[r, 0].set_ylabel(name, fontsize=9)
    if col_titles is not None:
        for c, t in enumerate(col_titles):
            axes[0, c].set_title(t, fontsize=title_fontsize)
    if suptitle:
        plt.suptitle(suptitle, fontsize=11)
    plt.tight_layout()
    plt.show()


def explained_variance(evr):
    """Cumulative explained variance of a fitted PCA (evr = explained_variance_ratio_)."""
    cumulative = np.cumsum(evr)
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(evr) + 1), cumulative, marker=".", markersize=4)
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("EuroSAT — PCA Cumulative Explained Variance (clean train)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def pc1_extremes(X, scores_pc1, n=10):
    """Images sampled along the PC1 axis (low → high). Visual check that PC1 ≈ brightness."""
    sorted_idx = np.argsort(scores_pc1)
    sample_idx = [sorted_idx[int(i * len(sorted_idx) / n)] for i in range(n)]
    fig, axes = plt.subplots(1, n, figsize=(n * 1.5, 2))
    for col, idx in enumerate(sample_idx):
        axes[col].imshow(X[idx], cmap="gray", vmin=0, vmax=1)
        axes[col].set_title(f"{scores_pc1[idx]:.1f}", fontsize=7)
        axes[col].axis("off")
    plt.suptitle("Images sorted by PC1 score (low → high)", fontsize=9)
    plt.tight_layout()
    plt.show()


def delta_hist_grid(deltas, sigmas, metrics, subtitle=""):
    """Per-image paired-delta histograms, one row per metric, one column per σ.

    deltas : dict[(metric, sigma)] -> (n,) array. Freedman–Diaconis bins
    (robust to outliers) — the heavy right tail of ΔTENG is visible here.
    """
    fig, axes = plt.subplots(len(metrics), len(sigmas),
                             figsize=(3.2 * len(sigmas), 2.4 * len(metrics)),
                             squeeze=False)
    for r, m in enumerate(metrics):
        for c, s in enumerate(sigmas):
            d = deltas[(m, s)]
            ax = axes[r, c]
            ax.hist(d, bins="fd", color=BLUE, edgecolor="white", linewidth=0.3)
            ax.axvline(0, color="gray", linestyle="--", linewidth=1)
            ax.axvline(d.mean(), color="crimson", linewidth=1.2)
            ax.axvline(np.median(d), color=ORANGE, linewidth=1.2, linestyle=":")
            if r == 0:
                ax.set_title(f"σ = {s}", fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{LABELS[m]}\n[{BETTER[m]} → hybrid better]", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)
    fig.suptitle(f"Per-image paired differences (red = mean, orange dotted = median){subtitle}",
                 y=1.0)
    plt.tight_layout()
    plt.show()


def bootstrap_grid(boots, deltas, sigmas, metrics, subtitle=""):
    """Bootstrap distributions of the MEAN paired delta (10 000 resamples).

    boots  : dict[(metric, sigma)] -> (n_bootstrap,) array of resample means
    deltas : dict[(metric, sigma)] -> (n,) array (for the observed mean line)
    Much narrower than the per-image histograms — spread ~ std/√n.
    """
    fig, axes = plt.subplots(len(metrics), len(sigmas),
                             figsize=(3.2 * len(sigmas), 2.4 * len(metrics)),
                             squeeze=False)
    for r, m in enumerate(metrics):
        for c, s in enumerate(sigmas):
            boot = boots[(m, s)]
            lo, hi = np.percentile(boot, [2.5, 97.5])
            ax = axes[r, c]
            ax.hist(boot, bins="fd", color=BLUE, edgecolor="white", linewidth=0.3)
            ax.axvspan(lo, hi, color="orange", alpha=0.25)
            ax.axvline(0, color="gray", linestyle="--", linewidth=1)
            ax.axvline(deltas[(m, s)].mean(), color="crimson", linewidth=1.2)
            if r == 0:
                ax.set_title(f"σ = {s}", fontsize=10)
            if c == 0:
                ax.set_ylabel(f"mean {LABELS[m]}", fontsize=8)
            verdict = "contains 0" if lo <= 0.0 <= hi else "excludes 0"
            ax.text(0.02, 0.95, f"95% CI {verdict}", transform=ax.transAxes,
                    fontsize=7, va="top")
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)
    fig.suptitle("Bootstrap distributions of the mean paired difference "
                 f"(shaded = 95% CI){subtitle}", y=1.0)
    plt.tight_layout()
    plt.show()


def fig_delta_vs_sigma(stats, sigmas, metrics, fig_dir):
    """Paper Fig. 1 — mean Δ ± 95% bootstrap CI vs σ (IEEE two-column span, 7.16 in).

    stats : stats[metric][sigma] -> dict with keys "mean", "ci" (as built in 02_analysis)
    Saves delta_vs_sigma.pdf + .png to fig_dir.
    """
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 1.45), constrained_layout=True)
    for ax, m in zip(axes, metrics):
        sig  = [s for s in sigmas if s in stats[m]]
        mean = np.array([stats[m][s]["mean"] for s in sig])
        lo   = np.array([stats[m][s]["ci"][0] for s in sig])
        hi   = np.array([stats[m][s]["ci"][1] for s in sig])
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.errorbar(sig, mean, yerr=[mean - lo, hi - mean],
                    fmt="o", color=BLUE, markersize=4, capsize=3, elinewidth=1.2)
        ax.set_xlabel("noise level σ", fontsize=8)
        ax.set_ylabel(LABELS[m], fontsize=8)
        ax.set_title(f"{BETTER[m]} → hybrid better", loc="right",
                     fontsize=7, color="black", y=1.05, pad=1)
        ax.set_xticks(sig)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
        ax.yaxis.get_offset_text().set_fontsize(7)
    fig.savefig(fig_dir / "delta_vs_sigma.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "delta_vs_sigma.png", dpi=300, bbox_inches="tight")
    plt.show()
    print(f"saved {fig_dir}/delta_vs_sigma.pdf (+ .png preview)")


def mean_vs_median(stats, sigmas, metrics, d_qrc):
    """Mean and median paired difference per σ with their bootstrap CIs.

    Diverging mean/median (opposite signs for ΔTENG at σ = 0.3) = outlier-dominated mean.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, m in zip(axes, metrics):
        sig    = [s for s in sigmas if s in stats[m]]
        mean   = np.array([stats[m][s]["mean"] for s in sig])
        m_lo   = np.array([stats[m][s]["ci"][0] for s in sig])
        m_hi   = np.array([stats[m][s]["ci"][1] for s in sig])
        median = np.array([stats[m][s]["median"] for s in sig])
        d_lo   = np.array([stats[m][s]["ci_med"][0] for s in sig])
        d_hi   = np.array([stats[m][s]["ci_med"][1] for s in sig])
        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax.errorbar(sig, mean, yerr=[mean - m_lo, m_hi - mean],
                    fmt="o", color=BLUE, markersize=6, capsize=4, elinewidth=1.5,
                    label="mean")
        ax.errorbar(sig, median, yerr=[median - d_lo, d_hi - median],
                    fmt="s", color=ORANGE, markersize=6, markerfacecolor="white",
                    capsize=4, elinewidth=1.5, label="median")
        ax.set_xlabel("noise level σ")
        ax.set_ylabel(f"{LABELS[m]} (hybrid − baseline)")
        ax.set_title(LABELS[m])
        ax.set_xticks(sig)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle(f"Mean vs median paired difference — 95% bootstrap CIs, d = {d_qrc}", y=1.02)
    plt.tight_layout()
    plt.show()


def win_rate(stats, sigmas, metrics, d_qrc):
    """% of test images where the hybrid is better (direction-corrected per metric)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    markers = {"mse": "o", "ssim": "s", "teng": "^"}
    colors  = {"mse": BLUE, "ssim": ORANGE, "teng": GREEN}
    for m in metrics:
        sig = [s for s in sigmas if s in stats[m]]
        # win_pct stores %(delta > 0); for MSE hybrid wins when delta < 0
        win = [100 - stats[m][s]["win_pct"] if BETTER[m] == "< 0" else stats[m][s]["win_pct"]
               for s in sig]
        ax.plot(sig, win, marker=markers[m], color=colors[m], linewidth=2,
                markersize=6, label=LABELS[m].replace("Δ", ""))
    ax.axhline(50, color="gray", linestyle="--", linewidth=1, label="coin flip (50%)")
    ax.set_xlabel("noise level σ")
    ax.set_ylabel("hybrid wins (% of test images)")
    ax.set_title(f"Hybrid win rate vs noise — d = {d_qrc}")
    ax.set_xticks(sigmas)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


def fig_qualitative(runs, sigmas, fig_dir):
    """Paper Fig. 2 — one texture-rich test image across all σ.

    Clean shown once (left); rows: PCA(18) linear projection → baseline MLP → hybrid QRC.
    runs : dict[sigma] -> npz with showcase_* arrays. Saves qualitative_vs_sigma.pdf + .png.
    """
    rows = [("pca", "PCA proj"), ("baseline", "Baseline"), ("hybrid", "QRC")]
    sig = [s for s in sigmas if s in runs]
    fig = plt.figure(figsize=(7.16, 3.5))
    gs = GridSpec(3, len(sig) + 2, figure=fig,
                  width_ratios=[2.2, 0.45] + [1.0] * len(sig), wspace=0.05, hspace=0.0)
    ax_clean = fig.add_subplot(gs[:, 0])
    ax_clean.imshow(runs[sig[0]]["showcase_clean"].reshape(64, 64),
                    cmap="gray", vmin=0, vmax=1)
    ax_clean.set_title("Clean", fontsize=13)
    ax_clean.set_xticks([]); ax_clean.set_yticks([])
    ax_clean.set_frame_on(False)
    for r, (key, label) in enumerate(rows):
        for c, s in enumerate(sig):
            ax = fig.add_subplot(gs[r, c + 2])
            img = np.clip(runs[s][f"showcase_{key}"].reshape(64, 64), 0, 1)
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"σ = {s}", fontsize=13)
            if c == 0:
                ax.set_ylabel(label, fontsize=13)
    fig.savefig(fig_dir / "qualitative_vs_sigma.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "qualitative_vs_sigma.png", dpi=300, bbox_inches="tight")
    plt.show()
    print(f"saved {fig_dir}/qualitative_vs_sigma.pdf (+ .png preview)")
