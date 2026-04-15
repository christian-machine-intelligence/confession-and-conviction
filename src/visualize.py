"""Publication-ready figures for the confession-and-conviction study.

Generates 4 figures:
  1. Divergence profile (||delta|| by layer)
  2. Patching heatmap (layers x heads)
  3. Delta consistency (cosine similarity heatmap)
  4. MFT domain comparison
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns

from src.config import NUM_LAYERS, NUM_HEADS, RESULTS_DIR, FIGURES_DIR
from src.prompts import MFT_DOMAINS


def load_results():
    """Load all result files."""
    results = {}

    profile_path = RESULTS_DIR / "divergence_profile.json"
    if profile_path.exists():
        with open(profile_path) as f:
            results["profile"] = json.load(f)

    cos_path = RESULTS_DIR / "cosine_similarity.json"
    if cos_path.exists():
        with open(cos_path) as f:
            results["cosine"] = json.load(f)

    patch_path = RESULTS_DIR / "patching_scores.json"
    if patch_path.exists():
        with open(patch_path) as f:
            results["patching"] = json.load(f)

    perm_path = RESULTS_DIR / "permutation_test.json"
    if perm_path.exists():
        with open(perm_path) as f:
            results["permutation"] = json.load(f)

    return results


def fig1_divergence_profile(results: dict):
    """Figure 1: ||delta|| by layer with error bars."""
    profile = results["profile"]

    layers = list(range(NUM_LAYERS))
    means = [profile[str(l)]["mean"] for l in layers]
    stds = [profile[str(l)]["std"] for l in layers]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(layers, means, yerr=stds, fmt="o-", capsize=4,
                color="#2c3e50", linewidth=2, markersize=6)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("||delta|| (L2 norm)", fontsize=12)
    ax.set_title("Divergence Profile: Religious vs Baseline", fontsize=14)
    ax.set_xticks(layers)
    ax.grid(True, alpha=0.3)

    # Mark peak
    peak_layer = np.argmax(means)
    ax.annotate(f"Peak: L{peak_layer}\n({means[peak_layer]:.2f})",
                xy=(peak_layer, means[peak_layer]),
                xytext=(peak_layer + 1.5, means[peak_layer]),
                arrowprops=dict(arrowstyle="->", color="#e74c3c"),
                fontsize=10, color="#e74c3c")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig1_divergence_profile.png", dpi=200)
    plt.close()
    print("Figure 1: divergence profile saved")


def fig2_patching_heatmap(results: dict):
    """Figure 2: 12x12 patching score heatmap (layers x heads)."""
    mean_scores = np.array(results["patching"]["mean_scores"])

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(mean_scores.T, aspect="auto", cmap="RdBu_r",
                   vmin=-0.3, vmax=0.3, origin="lower")

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Head", fontsize=12)
    ax.set_title("Activation Patching: Head-Level Normalized Scores", fontsize=14)
    ax.set_xticks(range(NUM_LAYERS))
    ax.set_yticks(range(NUM_HEADS))

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Normalized Patching Score", fontsize=10)

    # Annotate top 5 heads
    flat = mean_scores.ravel()
    top5 = np.argsort(np.abs(flat))[::-1][:5]
    for flat_idx in top5:
        l, h = np.unravel_index(flat_idx, mean_scores.shape)
        ax.plot(l, h, "k*", markersize=12)
        ax.annotate(f"L{l}H{h}\n{mean_scores[l,h]:.3f}",
                    xy=(l, h), xytext=(l + 0.5, h + 0.5),
                    fontsize=7, color="black",
                    arrowprops=dict(arrowstyle="->", color="black", lw=0.5))

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_patching_heatmap.png", dpi=200)
    plt.close()
    print("Figure 2: patching heatmap saved")


def fig3_delta_consistency(results: dict):
    """Figure 3: Cosine similarity heatmap of delta vectors at the OUTPUT layer.

    Note: we deliberately use layer 11 (output layer), not the layer of
    maximum consistency. The maximum-consistency layer is layer 0, where the
    delta is dominated by the identical prefix-token embeddings — a trivial
    result. Layer 11 reflects the substantively interesting question: how
    consistently does the prefix's effect propagate through the network's
    contextual processing?
    """
    cos_data = results["cosine"]
    best_layer = NUM_LAYERS - 1  # layer 11 (output)
    sim_matrix = np.array(cos_data[str(best_layer)]["matrix"])

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(sim_matrix, cmap="viridis", vmin=-0.2, vmax=1.0)

    ax.set_xlabel("Prompt Pair", fontsize=12)
    ax.set_ylabel("Prompt Pair", fontsize=12)
    mean_sim = cos_data[str(best_layer)]["mean_cosine_sim"]
    ax.set_title(f"Delta Consistency (Layer {best_layer}, "
                 f"mean cos sim = {mean_sim:.3f})", fontsize=14)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Cosine Similarity", fontsize=10)

    # Add permutation test result if available
    if "permutation" in results:
        perm = results["permutation"]
        ax.text(0.02, 0.02,
                f"Permutation p = {perm['p_value']:.4f}",
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig3_delta_consistency.png", dpi=200)
    plt.close()
    print("Figure 3: delta consistency saved")


def fig4_summary(results: dict):
    """Figure 4: Summary panel combining key findings."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel A: Divergence profile (compact)
    profile = results["profile"]
    layers = list(range(NUM_LAYERS))
    means = [profile[str(l)]["mean"] for l in layers]
    axes[0].plot(layers, means, "o-", color="#2c3e50", linewidth=2)
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("||delta||")
    axes[0].set_title("A. Divergence Profile")
    axes[0].set_xticks(layers)
    axes[0].grid(True, alpha=0.3)

    # Panel B: Mean cosine sim by layer
    cos_data = results["cosine"]
    cos_means = [cos_data[str(l)]["mean_cosine_sim"] for l in layers]
    axes[1].plot(layers, cos_means, "s-", color="#8e44ad", linewidth=2)
    axes[1].axhline(y=0.3, color="#e74c3c", linestyle="--", alpha=0.5,
                    label="Threshold (0.3)")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Mean Cosine Similarity")
    axes[1].set_title("B. Delta Consistency")
    axes[1].set_xticks(layers)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Panel C: Patching heatmap (compact)
    if "patching" in results:
        mean_scores = np.array(results["patching"]["mean_scores"])
        im = axes[2].imshow(mean_scores.T, aspect="auto", cmap="RdBu_r",
                           vmin=-0.3, vmax=0.3, origin="lower")
        axes[2].set_xlabel("Layer")
        axes[2].set_ylabel("Head")
        axes[2].set_title("C. Head Patching Scores")
        plt.colorbar(im, ax=axes[2], shrink=0.8)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig4_summary.png", dpi=200)
    plt.close()
    print("Figure 4: summary panel saved")


def generate_all_figures():
    """Generate all figures from saved results."""
    results = load_results()

    if "profile" in results:
        fig1_divergence_profile(results)

    if "patching" in results:
        fig2_patching_heatmap(results)

    if "cosine" in results:
        fig3_delta_consistency(results)

    if "profile" in results and "cosine" in results:
        fig4_summary(results)

    print(f"\nAll figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    generate_all_figures()
