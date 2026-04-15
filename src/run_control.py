"""Run the full pipeline on non-moral control prompts.

Extracts activations, computes deltas, runs ablation logit analysis,
and compares to the moral prompt results.
"""

import json
import torch
import numpy as np
from collections import Counter
from scipy.spatial.distance import cosine
from tqdm import tqdm
from transformer_lens import HookedTransformer

from src.config import MODEL_NAME, NUM_LAYERS, RESULTS_DIR
from src.prompts_control import get_control_pairs

TOP_K = 15

ABLATE_CONFIGS = {
    "L9H8": [(9, 8)],
    "L11H3": [(11, 3)],
    "L5H10": [(5, 10)],
    "L10H0": [(10, 0)],
    "top2": [(9, 8), (11, 3)],
    "all4": [(9, 8), (11, 3), (5, 10), (10, 0)],
}


def run_control_study():
    print(f"Loading {MODEL_NAME} ...")
    model = HookedTransformer.from_pretrained(MODEL_NAME)
    model.eval()

    pairs = get_control_pairs()
    ctrl_dir = RESULTS_DIR / "control"
    ctrl_dir.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: Extract activations and compute deltas ──
    print("\n" + "=" * 60)
    print("PHASE 1: Extraction + Delta Analysis (control prompts)")
    print("=" * 60)

    deltas = {l: [] for l in range(NUM_LAYERS)}

    for pair in tqdm(pairs, desc="Extracting"):
        rel_tokens = model.to_tokens(pair["religious"], prepend_bos=True)
        base_tokens = model.to_tokens(pair["baseline"], prepend_bos=True)

        with torch.no_grad():
            _, rel_cache = model.run_with_cache(rel_tokens)
            _, base_cache = model.run_with_cache(base_tokens)

        for layer in range(NUM_LAYERS):
            rel_vec = rel_cache[f"blocks.{layer}.hook_resid_post"][0, -1]
            base_vec = base_cache[f"blocks.{layer}.hook_resid_post"][0, -1]
            deltas[layer].append((rel_vec - base_vec).cpu())

    # Stack and compute norms
    for layer in range(NUM_LAYERS):
        deltas[layer] = torch.stack(deltas[layer])

    norms = {}
    for layer in range(NUM_LAYERS):
        layer_norms = deltas[layer].norm(dim=-1).numpy()
        norms[layer] = {
            "mean": float(np.mean(layer_norms)),
            "std": float(np.std(layer_norms)),
        }
        print(f"  Layer {layer:2d}: ||delta|| = {norms[layer]['mean']:.2f} +/- {norms[layer]['std']:.2f}")

    with open(ctrl_dir / "divergence_profile.json", "w") as f:
        json.dump(norms, f, indent=2)

    # Cosine similarity
    print("\nDelta consistency (cosine sim):")
    cos_by_layer = {}
    for layer in range(NUM_LAYERS):
        vecs = deltas[layer].numpy()
        n = vecs.shape[0]
        sim = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                sim[i, j] = 1.0 - cosine(vecs[i], vecs[j])
        mask = ~np.eye(n, dtype=bool)
        mean_sim = float(np.mean(sim[mask]))
        cos_by_layer[layer] = mean_sim
        print(f"  Layer {layer:2d}: mean cosine sim = {mean_sim:.4f}")

    with open(ctrl_dir / "cosine_similarity.json", "w") as f:
        json.dump({str(k): v for k, v in cos_by_layer.items()}, f, indent=2)

    # ── Phase 2: Ablation logit analysis ──
    print("\n" + "=" * 60)
    print("PHASE 2: Ablation Logit Analysis (control prompts)")
    print("=" * 60)

    ablation_results = []

    for pair in pairs:
        idx = pair["idx"]

        base_tokens = model.to_tokens(pair["baseline"], prepend_bos=True)
        rel_tokens = model.to_tokens(pair["religious"], prepend_bos=True)

        with torch.no_grad():
            base_logits = model(base_tokens)
            rel_logits = model(rel_tokens)

        base_probs = base_logits[0, -1].softmax(dim=-1)
        rel_probs = rel_logits[0, -1].softmax(dim=-1)

        entry = {"idx": idx, "stem": pair["stem"]}
        entry["ablation_effects"] = {}

        for config_name, heads in ABLATE_CONFIGS.items():
            for layer, head in heads:
                def make_hook(h):
                    def hook_fn(value, hook):
                        value[:, :, h, :] = 0.0
                        return value
                    return hook_fn
                model.add_hook(f"blocks.{layer}.attn.hook_z", make_hook(head))

            with torch.no_grad():
                abl_logits = model(rel_tokens)
            model.reset_hooks()

            abl_probs = abl_logits[0, -1].softmax(dim=-1)
            abl_diff = abl_probs - rel_probs

            top_abl = abl_diff.abs().topk(TOP_K)
            changes = []
            for i in range(TOP_K):
                tid = top_abl.indices[i].item()
                tok = model.to_single_str_token(tid)
                rel_p = rel_probs[tid].item()
                abl_p = abl_probs[tid].item()
                base_p = base_probs[tid].item()
                toward_baseline = abs(abl_p - base_p) < abs(rel_p - base_p)
                changes.append({
                    "token": tok,
                    "direction": "->baseline" if toward_baseline else "->diverge",
                })

            n_toward = sum(1 for c in changes if c["direction"] == "->baseline")
            entry["ablation_effects"][config_name] = {
                "changes": changes,
                "fraction_toward_baseline": n_toward / TOP_K,
            }

        ablation_results.append(entry)

    with open(ctrl_dir / "ablation_logits.json", "w") as f:
        json.dump(ablation_results, f, indent=2, ensure_ascii=False)

    # ── Phase 3: Comparison summary ──
    print("\n" + "=" * 60)
    print("COMPARISON: Moral vs Control")
    print("=" * 60)

    # Load moral results
    with open(RESULTS_DIR / "divergence_profile.json") as f:
        moral_profile = json.load(f)
    with open(RESULTS_DIR / "ablation_logits.json") as f:
        moral_ablation = json.load(f)

    print("\nDivergence profile (||delta||):")
    print(f"  {'Layer':>5s}  {'Moral':>8s}  {'Control':>8s}  {'Ratio':>6s}")
    print("  " + "-" * 35)
    for layer in range(NUM_LAYERS):
        m = moral_profile[str(layer)]["mean"]
        c = norms[layer]["mean"]
        ratio = m / c if c > 0 else float('inf')
        print(f"  {layer:5d}  {m:8.2f}  {c:8.2f}  {ratio:6.2f}x")

    print("\nAblation recovery (fraction toward baseline):")
    configs = ["L9H8", "L11H3", "L5H10", "L10H0", "top2", "all4"]
    print(f"  {'Config':>10s}  {'Moral':>6s}  {'Control':>7s}  {'Diff':>6s}")
    print("  " + "-" * 35)
    for config in configs:
        moral_fracs = [e["ablation_effects"][config]["fraction_toward_baseline"]
                       for e in moral_ablation if config in e["ablation_effects"]]
        ctrl_fracs = [e["ablation_effects"][config]["fraction_toward_baseline"]
                      for e in ablation_results if config in e["ablation_effects"]]
        m_mean = sum(moral_fracs) / len(moral_fracs) if moral_fracs else 0
        c_mean = sum(ctrl_fracs) / len(ctrl_fracs) if ctrl_fracs else 0
        print(f"  {config:>10s}  {m_mean:>6.3f}  {c_mean:>7.3f}  {m_mean - c_mean:>+6.3f}")

    # Token specialization for L5H10 on control prompts
    print("\nL5H10 token recovery on CONTROL prompts:")
    toward = Counter()
    away = Counter()
    for entry in ablation_results:
        if "L5H10" in entry["ablation_effects"]:
            for change in entry["ablation_effects"]["L5H10"]["changes"]:
                tok = change["token"].strip()
                if change["direction"] == "->baseline":
                    toward[tok] += 1
                else:
                    away[tok] += 1
    for tok, count in toward.most_common(10):
        away_count = away.get(tok, 0)
        print(f"    {tok!r:>15s}: toward {count:2d}x, away {away_count:2d}x")

    print("\nDone.")


if __name__ == "__main__":
    run_control_study()
