"""Robustness experiments addressing methodological concerns.

Four analyses, all on GPT-2-small with TransformerLens:

  1. KL-divergence patching: replace the target-token mean log-prob metric with
     a full-distribution KL-divergence metric, recompute patching scores for
     all 144 heads, and verify the top heads are stable across metrics.

  2. Random-head ablation baseline: ablate each of 30 randomly selected heads
     (excluding our top 4) and report the recovery-rate distribution. Gives a
     proper chance baseline for the "fraction toward baseline" statistic.

  3. Stronger permutation null: compute deltas for 20 (Christian-A vs
     Christian-B) intra-condition pairs and check whether those deltas also
     cluster. If yes, the prefix is not the source of the alignment.

  4. Default-circuit characterization: for the 20 baseline (unprefixed)
     prompts, measure each head's L2 output norm at the last token. Reports
     which heads are most active under default processing, for comparison
     with the four heads we identify as Christian-circuit.
"""

import json
import random
import torch
import numpy as np
from itertools import combinations
from scipy.spatial.distance import cosine
from tqdm import tqdm
from transformer_lens import HookedTransformer

from src.config import MODEL_NAME, NUM_LAYERS, NUM_HEADS, RESULTS_DIR, SEED
from src.prompts import get_prompt_pairs

OUR_TOP_HEADS = {(9, 8), (11, 3), (5, 10), (10, 0)}
N_RANDOM_HEADS = 30


# ---------------------------------------------------------------------------
# Experiment 1: KL-divergence patching
# ---------------------------------------------------------------------------

def kl_div(p: torch.Tensor, q: torch.Tensor) -> float:
    """KL(P || Q) for distributions over the vocabulary. Both [vocab_size]."""
    p = p.clamp(min=1e-12)
    q = q.clamp(min=1e-12)
    return float((p * (p.log() - q.log())).sum().item())


def kl_patching(model: HookedTransformer):
    """Recompute patching scores using full-distribution KL divergence.

    Score: 1 - KL(P_patched || P_religious) / KL(P_baseline || P_religious)

    A score of 1 means the patched distribution is identical to the religious
    distribution; a score of 0 means it is unchanged from baseline. This
    metric considers the entire output distribution rather than 16 hand-picked
    target tokens.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: KL-divergence patching metric")
    print("=" * 60)

    pairs = get_prompt_pairs()
    all_scores = np.zeros((len(pairs), NUM_LAYERS, NUM_HEADS))

    for pair_idx, pair in enumerate(tqdm(pairs, desc="KL patching")):
        clean_tokens = model.to_tokens(pair["baseline"], prepend_bos=True)
        corrupted_tokens = model.to_tokens(pair["religious"], prepend_bos=True)

        with torch.no_grad():
            clean_logits, _ = model.run_with_cache(clean_tokens)
            corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_tokens)

        clean_probs = clean_logits[0, -1].softmax(dim=-1)
        corrupted_probs = corrupted_logits[0, -1].softmax(dim=-1)

        denom = kl_div(clean_probs, corrupted_probs)
        if denom < 1e-8:
            continue

        for layer in range(NUM_LAYERS):
            for head in range(NUM_HEADS):
                def make_hook(l, h):
                    def hook_fn(value, hook):
                        corrupted_z = corrupted_cache[f"blocks.{l}.attn.hook_z"]
                        value[:, -1, h, :] = corrupted_z[:, -1, h, :]
                        return value
                    return hook_fn

                with torch.no_grad():
                    patched_logits = model.run_with_hooks(
                        clean_tokens,
                        fwd_hooks=[(f"blocks.{layer}.attn.hook_z", make_hook(layer, head))],
                    )
                patched_probs = patched_logits[0, -1].softmax(dim=-1)
                kl_patched = kl_div(patched_probs, corrupted_probs)
                # 1 - KL(patched || corrupted) / KL(clean || corrupted)
                # = how much closer to corrupted (religious) we got
                all_scores[pair_idx, layer, head] = 1.0 - kl_patched / denom

    mean_scores = all_scores.mean(axis=0)
    flat = np.argsort(mean_scores.ravel())[::-1][:10]
    print("\nTop 10 heads by KL patching score:")
    top10 = []
    for fi in flat:
        l, h = np.unravel_index(fi, mean_scores.shape)
        score = mean_scores[l, h]
        in_orig_top4 = (l, h) in OUR_TOP_HEADS
        marker = " *" if in_orig_top4 else ""
        print(f"  L{l}H{h}: {score:.4f}{marker}")
        top10.append({"layer": int(l), "head": int(h),
                      "kl_score": float(score), "in_orig_top4": in_orig_top4})

    out = {
        "method": "1 - KL(patched || religious) / KL(baseline || religious)",
        "mean_scores": mean_scores.tolist(),
        "top10": top10,
    }
    with open(RESULTS_DIR / "kl_patching.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {RESULTS_DIR / 'kl_patching.json'}")
    return out


# ---------------------------------------------------------------------------
# Experiment 2: Random-head ablation baseline
# ---------------------------------------------------------------------------

def random_head_baseline(model: HookedTransformer):
    """Ablate 30 random heads (excluding our top 4) and report recovery
    rates. Gives a proper chance baseline for the 'fraction toward baseline'
    statistic.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Random-head ablation baseline")
    print("=" * 60)

    pairs = get_prompt_pairs()

    rng = random.Random(SEED)
    all_heads = [(l, h) for l in range(NUM_LAYERS) for h in range(NUM_HEADS)
                 if (l, h) not in OUR_TOP_HEADS]
    random_heads = rng.sample(all_heads, N_RANDOM_HEADS)

    TOP_K = 15
    results = []

    for layer, head in tqdm(random_heads, desc="Random heads"):
        recovery_per_pair = []
        for pair in pairs:
            base_tokens = model.to_tokens(pair["baseline"], prepend_bos=True)
            rel_tokens = model.to_tokens(pair["religious"], prepend_bos=True)

            with torch.no_grad():
                base_logits = model(base_tokens)
                rel_logits = model(rel_tokens)
            base_probs = base_logits[0, -1].softmax(dim=-1)
            rel_probs = rel_logits[0, -1].softmax(dim=-1)

            def hook_fn(value, hook):
                value[:, :, head, :] = 0.0
                return value
            model.add_hook(f"blocks.{layer}.attn.hook_z", hook_fn)
            with torch.no_grad():
                abl_logits = model(rel_tokens)
            model.reset_hooks()
            abl_probs = abl_logits[0, -1].softmax(dim=-1)

            abl_diff = (abl_probs - rel_probs).abs()
            top_indices = abl_diff.topk(TOP_K).indices

            n_toward = 0
            for tid in top_indices.tolist():
                rel_p = rel_probs[tid].item()
                abl_p = abl_probs[tid].item()
                base_p = base_probs[tid].item()
                if abs(abl_p - base_p) < abs(rel_p - base_p):
                    n_toward += 1
            recovery_per_pair.append(n_toward / TOP_K)

        mean_rec = sum(recovery_per_pair) / len(recovery_per_pair)
        results.append({"layer": int(layer), "head": int(head), "mean_recovery": mean_rec})

    rec_values = [r["mean_recovery"] for r in results]
    rec_arr = np.array(rec_values)
    print(f"\nRandom head ablation across {N_RANDOM_HEADS} heads:")
    print(f"  Mean recovery: {rec_arr.mean()*100:.1f}%")
    print(f"  Std:           {rec_arr.std()*100:.1f}%")
    print(f"  Min:           {rec_arr.min()*100:.1f}%")
    print(f"  Median:        {np.median(rec_arr)*100:.1f}%")
    print(f"  Max:           {rec_arr.max()*100:.1f}%")
    print(f"  95% percentile: {np.percentile(rec_arr, 95)*100:.1f}%")
    print(f"\n  For comparison: top-4 head recoveries were 56.3-60.0%")

    out = {
        "n_heads": N_RANDOM_HEADS,
        "per_head": results,
        "summary": {
            "mean": float(rec_arr.mean()),
            "std": float(rec_arr.std()),
            "min": float(rec_arr.min()),
            "median": float(np.median(rec_arr)),
            "max": float(rec_arr.max()),
            "p95": float(np.percentile(rec_arr, 95)),
        },
    }
    with open(RESULTS_DIR / "random_head_baseline.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {RESULTS_DIR / 'random_head_baseline.json'}")
    return out


# ---------------------------------------------------------------------------
# Experiment 3: Stronger permutation null
# ---------------------------------------------------------------------------

def stronger_null(model: HookedTransformer):
    """Compute deltas for (Christian-A vs Christian-B) intra-condition pairs.

    If these deltas also cluster at high cosine similarity, the prefix is not
    driving the alignment — it's just that any pair of activations from this
    model are correlated at the late layer.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Stronger permutation null")
    print("=" * 60)

    pairs = get_prompt_pairs()
    n_pairs = len(pairs)

    # First, recompute the original (Christian, baseline) deltas at layer 11
    print("Computing original (Christian - baseline) deltas at layer 11...")
    rel_resids = []
    base_resids = []
    for pair in tqdm(pairs):
        rel_tokens = model.to_tokens(pair["religious"], prepend_bos=True)
        base_tokens = model.to_tokens(pair["baseline"], prepend_bos=True)
        with torch.no_grad():
            _, rel_cache = model.run_with_cache(rel_tokens)
            _, base_cache = model.run_with_cache(base_tokens)
        rel_resids.append(rel_cache["blocks.11.hook_resid_post"][0, -1].cpu().numpy())
        base_resids.append(base_cache["blocks.11.hook_resid_post"][0, -1].cpu().numpy())

    rel_resids = np.array(rel_resids)
    base_resids = np.array(base_resids)

    orig_deltas = rel_resids - base_resids

    def mean_off_diag_cos(vecs):
        n = len(vecs)
        sims = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                sims.append(1.0 - cosine(vecs[i], vecs[j]))
        return float(np.mean(sims))

    orig_cos = mean_off_diag_cos(orig_deltas)
    print(f"  Original (Christian - baseline) mean cos sim: {orig_cos:.4f}")

    # NULL 1: 20 (Christian_i - Christian_j) deltas (intra-condition,
    # different content)
    print("\nNull 1: (Christian-A - Christian-B) intra-condition pairs...")
    rng = np.random.default_rng(SEED)
    indices = list(range(n_pairs))
    intra_christian_deltas = []
    for k in range(n_pairs):
        i, j = rng.choice(indices, size=2, replace=False)
        intra_christian_deltas.append(rel_resids[i] - rel_resids[j])
    null1_cos = mean_off_diag_cos(np.array(intra_christian_deltas))
    print(f"  (Christian-A - Christian-B) mean cos sim: {null1_cos:.4f}")

    # NULL 2: 20 (baseline_i - baseline_j) deltas (no prefix at all)
    print("\nNull 2: (baseline-A - baseline-B) intra-condition pairs...")
    intra_base_deltas = []
    for k in range(n_pairs):
        i, j = rng.choice(indices, size=2, replace=False)
        intra_base_deltas.append(base_resids[i] - base_resids[j])
    null2_cos = mean_off_diag_cos(np.array(intra_base_deltas))
    print(f"  (baseline-A - baseline-B) mean cos sim: {null2_cos:.4f}")

    print("\nInterpretation:")
    print(f"  If null cos sims are << {orig_cos:.2f}, the prefix is driving the alignment.")
    print(f"  If null cos sims are similar to {orig_cos:.2f}, late-layer activations")
    print(f"  are just generically aligned in this model.")

    out = {
        "layer": 11,
        "original_christian_minus_baseline_cos": orig_cos,
        "null1_intra_christian_cos": null1_cos,
        "null2_intra_baseline_cos": null2_cos,
    }
    with open(RESULTS_DIR / "stronger_null.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {RESULTS_DIR / 'stronger_null.json'}")
    return out


# ---------------------------------------------------------------------------
# Experiment 4: Default-circuit characterization
# ---------------------------------------------------------------------------

def default_circuit(model: HookedTransformer):
    """For the 20 baseline (unprefixed) prompts, measure each attention head's
    L2 output norm at the last token. Reports which heads dominate default
    processing for comparison with the Christian-circuit heads.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Default-circuit characterization")
    print("=" * 60)

    pairs = get_prompt_pairs()
    head_norms_baseline = np.zeros((len(pairs), NUM_LAYERS, NUM_HEADS))
    head_norms_religious = np.zeros((len(pairs), NUM_LAYERS, NUM_HEADS))

    for pair_idx, pair in enumerate(tqdm(pairs, desc="Default circuit")):
        for cond, target_arr in [("baseline", head_norms_baseline),
                                 ("religious", head_norms_religious)]:
            tokens = model.to_tokens(pair[cond], prepend_bos=True)
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens)
            for layer in range(NUM_LAYERS):
                z = cache[f"blocks.{layer}.attn.hook_z"][0, -1]  # [n_heads, d_head]
                for head in range(NUM_HEADS):
                    target_arr[pair_idx, layer, head] = float(z[head].norm().item())

    mean_baseline = head_norms_baseline.mean(axis=0)
    mean_religious = head_norms_religious.mean(axis=0)

    flat = np.argsort(mean_baseline.ravel())[::-1][:10]
    print("\nTop 10 heads by output norm under BASELINE (no prefix):")
    baseline_top = []
    for fi in flat:
        l, h = np.unravel_index(fi, mean_baseline.shape)
        norm_b = mean_baseline[l, h]
        norm_r = mean_religious[l, h]
        in_top4 = (l, h) in OUR_TOP_HEADS
        marker = " *" if in_top4 else ""
        print(f"  L{l}H{h}: baseline_norm={norm_b:.3f}, religious_norm={norm_r:.3f}{marker}")
        baseline_top.append({
            "layer": int(l), "head": int(h),
            "baseline_norm": float(norm_b),
            "religious_norm": float(norm_r),
            "ratio_relig_over_base": float(norm_r / norm_b) if norm_b > 0 else None,
            "in_christian_top4": in_top4,
        })

    print("\nNorm of our 4 Christian-circuit heads under both conditions:")
    christian_heads_data = []
    for (l, h) in sorted(OUR_TOP_HEADS):
        norm_b = mean_baseline[l, h]
        norm_r = mean_religious[l, h]
        ratio = norm_r / norm_b if norm_b > 0 else float("nan")
        rank_baseline = int((mean_baseline > norm_b).sum()) + 1
        print(f"  L{l}H{h}: baseline={norm_b:.3f} (rank {rank_baseline} of 144), "
              f"religious={norm_r:.3f}, ratio={ratio:.2f}x")
        christian_heads_data.append({
            "layer": int(l), "head": int(h),
            "baseline_norm": float(norm_b),
            "religious_norm": float(norm_r),
            "ratio": float(ratio),
            "baseline_rank": int(rank_baseline),
        })

    print("\nInterpretation:")
    print("  If our 4 heads have low baseline norm (rank > 50), they are dormant")
    print("  under default processing and are specifically activated by the prefix.")
    print("  If they are already among the top heads under baseline, the 'distinct")
    print("  circuit' framing is too strong — these are general-purpose late-layer")
    print("  heads that happen to amplify the prefix effect.")

    out = {
        "default_top10": baseline_top,
        "christian_heads_under_both_conditions": christian_heads_data,
    }
    with open(RESULTS_DIR / "default_circuit.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {RESULTS_DIR / 'default_circuit.json'}")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default="all",
                        help="Which experiment(s): all, kl, random, null, default")
    args = parser.parse_args()

    print(f"Loading {MODEL_NAME} ...")
    model = HookedTransformer.from_pretrained(MODEL_NAME)
    model.eval()

    if args.exp in ("all", "kl"):
        kl_patching(model)
    if args.exp in ("all", "random"):
        random_head_baseline(model)
    if args.exp in ("all", "null"):
        stronger_null(model)
    if args.exp in ("all", "default"):
        default_circuit(model)

    print("\n" + "=" * 60)
    print("All robustness experiments complete.")
    print("=" * 60)
