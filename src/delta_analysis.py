"""Delta vector analysis: divergence profiles and consistency.

Computes delta = religious - baseline activations at each layer,
then checks:
  H1: WHERE do conditions diverge? (divergence profile)
  H4: Is the delta direction consistent across 20 pairs? (cosine similarity)
"""

import json
import torch
import numpy as np
from scipy.spatial.distance import cosine
from tqdm import tqdm

from src.config import NUM_LAYERS, RESULTS_DIR, N_PERMUTATIONS, SEED
from src.extract import load_cached
from src.prompts import get_prompt_pairs, MFT_DOMAINS


def compute_deltas(pair_indices: list[int] | None = None) -> dict:
    """Compute delta vectors (religious - baseline) at each layer.

    For the last token position of the baseline prompt. Since the religious
    prompt has extra prefix tokens, we align on the shared suffix by using
    the last token of each.

    Returns dict with:
      - deltas: {layer: [n_pairs, d_model]}
      - norms: {layer: [n_pairs]}
      - actual_indices: list of pair indices used
    """
    pairs = get_prompt_pairs()
    if pair_indices is not None:
        pairs = [p for p in pairs if p["idx"] in pair_indices]

    deltas = {l: [] for l in range(NUM_LAYERS)}
    norms = {l: [] for l in range(NUM_LAYERS)}

    for pair in tqdm(pairs, desc="Computing deltas"):
        idx = pair["idx"]
        rel = load_cached(idx, "religious")
        base = load_cached(idx, "baseline")

        for layer in range(NUM_LAYERS):
            # Last token of each prompt's residual stream
            rel_vec = rel["resid_post"][layer][-1]   # [d_model]
            base_vec = base["resid_post"][layer][-1]  # [d_model]
            delta = rel_vec - base_vec
            deltas[layer].append(delta)
            norms[layer].append(delta.norm().item())

    # Stack into tensors
    for layer in range(NUM_LAYERS):
        deltas[layer] = torch.stack(deltas[layer])  # [n_pairs, d_model]
        norms[layer] = np.array(norms[layer])

    return {"deltas": deltas, "norms": norms, "actual_indices": [p["idx"] for p in pairs]}


def divergence_profile(norms: dict) -> dict:
    """Compute mean and std of ||delta|| at each layer."""
    profile = {}
    for layer in range(NUM_LAYERS):
        profile[layer] = {
            "mean": float(np.mean(norms[layer])),
            "std": float(np.std(norms[layer])),
            "per_pair": norms[layer].tolist(),
        }
    return profile


def cosine_similarity_matrix(deltas: dict, layer: int) -> np.ndarray:
    """Compute pairwise cosine similarity of delta vectors at a given layer."""
    vecs = deltas[layer].numpy()  # [n_pairs, d_model]
    n = vecs.shape[0]
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim_matrix[i, j] = 1.0 - cosine(vecs[i], vecs[j])
    return sim_matrix


def mean_cosine_sim(sim_matrix: np.ndarray) -> float:
    """Mean off-diagonal cosine similarity."""
    n = sim_matrix.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return float(np.mean(sim_matrix[mask]))


def permutation_test(deltas: dict, layer: int, actual_indices: list[int],
                     n_perms: int = N_PERMUTATIONS) -> dict:
    """Test whether delta consistency is above chance via label shuffling.

    Shuffles which prompt is 'religious' vs 'baseline' and recomputes
    mean cosine similarity. Returns observed value and p-value.
    """
    rng = np.random.default_rng(SEED)

    # Observed
    sim_matrix = cosine_similarity_matrix(deltas, layer)
    observed = mean_cosine_sim(sim_matrix)

    # Collect both conditions' last-token residuals
    rel_vecs = []
    base_vecs = []
    for pair_idx in actual_indices:
        rel = load_cached(pair_idx, "religious")
        base = load_cached(pair_idx, "baseline")
        rel_vecs.append(rel["resid_post"][layer][-1].numpy())
        base_vecs.append(base["resid_post"][layer][-1].numpy())

    rel_vecs = np.array(rel_vecs)
    base_vecs = np.array(base_vecs)

    null_sims = []
    for _ in range(n_perms):
        # For each pair, randomly swap religious/baseline labels
        shuffled_deltas = []
        for i in range(len(rel_vecs)):
            if rng.random() < 0.5:
                shuffled_deltas.append(rel_vecs[i] - base_vecs[i])
            else:
                shuffled_deltas.append(base_vecs[i] - rel_vecs[i])
        shuffled_deltas = np.array(shuffled_deltas)

        n = len(shuffled_deltas)
        sim = np.zeros((n, n))
        for a in range(n):
            for b in range(n):
                sim[a, b] = 1.0 - cosine(shuffled_deltas[a], shuffled_deltas[b])
        mask = ~np.eye(n, dtype=bool)
        null_sims.append(float(np.mean(sim[mask])))

    p_value = float(np.mean(np.array(null_sims) >= observed))

    return {
        "observed_mean_cosine_sim": observed,
        "p_value": p_value,
        "null_mean": float(np.mean(null_sims)),
        "null_std": float(np.std(null_sims)),
    }


def run_delta_analysis(pair_indices: list[int] | None = None):
    """Run full delta analysis and save results."""
    result = compute_deltas(pair_indices)

    # Divergence profile
    profile = divergence_profile(result["norms"])
    profile_path = RESULTS_DIR / "divergence_profile.json"
    with open(profile_path, "w") as f:
        json.dump(profile, f, indent=2)
    print(f"Divergence profile saved to {profile_path}")

    # Find peak divergence layer
    peak_layer = max(range(NUM_LAYERS), key=lambda l: profile[l]["mean"])
    print(f"Peak divergence at layer {peak_layer} "
          f"(||delta|| = {profile[peak_layer]['mean']:.4f})")

    # Cosine similarity at each layer
    cos_results = {}
    for layer in range(NUM_LAYERS):
        sim_matrix = cosine_similarity_matrix(result["deltas"], layer)
        mean_sim = mean_cosine_sim(sim_matrix)
        cos_results[layer] = {
            "mean_cosine_sim": mean_sim,
            "matrix": sim_matrix.tolist(),
        }
        print(f"  Layer {layer:2d}: mean cosine sim = {mean_sim:.4f}")

    cos_path = RESULTS_DIR / "cosine_similarity.json"
    with open(cos_path, "w") as f:
        json.dump(cos_results, f, indent=2)

    # Permutation test at peak layer
    print(f"\nRunning permutation test at layer {peak_layer} ...")
    perm_result = permutation_test(result["deltas"], peak_layer, result["actual_indices"])
    print(f"  Observed mean cosine sim: {perm_result['observed_mean_cosine_sim']:.4f}")
    print(f"  Null distribution: {perm_result['null_mean']:.4f} +/- {perm_result['null_std']:.4f}")
    print(f"  p-value: {perm_result['p_value']:.4f}")

    perm_path = RESULTS_DIR / "permutation_test.json"
    with open(perm_path, "w") as f:
        json.dump(perm_result, f, indent=2)

    # Save delta vectors for visualization
    delta_path = RESULTS_DIR / "delta_vectors.pt"
    torch.save(result["deltas"], delta_path)
    print(f"Delta vectors saved to {delta_path}")

    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=str, default=None)
    args = parser.parse_args()

    pair_indices = None
    if args.pairs:
        if "-" in args.pairs:
            start, end = args.pairs.split("-")
            pair_indices = list(range(int(start), int(end) + 1))
        else:
            pair_indices = [int(x) for x in args.pairs.split(",")]

    run_delta_analysis(pair_indices)
