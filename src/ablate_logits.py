"""Logit-level ablation: how does zeroing key heads change the output distribution?

Instead of generating text (which is degenerate in GPT-2-small), we look at
the next-token probability distribution at the last position and report
which tokens gain/lose the most probability when heads are ablated.
"""

import json
import torch
from transformer_lens import HookedTransformer

from src.config import MODEL_NAME, RESULTS_DIR
from src.prompts import get_prompt_pairs


ABLATE_CONFIGS = {
    "L9H8": [(9, 8)],
    "L11H3": [(11, 3)],
    "L5H10": [(5, 10)],
    "L10H0": [(10, 0)],
    "top2": [(9, 8), (11, 3)],
    "all4": [(9, 8), (11, 3), (5, 10), (10, 0)],
}

TOP_K = 15  # show top-K tokens that shift most


def run_logit_ablation(pair_indices: list[int] | None = None):
    print(f"Loading {MODEL_NAME} ...")
    model = HookedTransformer.from_pretrained(MODEL_NAME)
    model.eval()

    pairs = get_prompt_pairs()
    if pair_indices is not None:
        pairs = [p for p in pairs if p["idx"] in pair_indices]

    results = []

    for pair in pairs:
        idx = pair["idx"]
        print(f"\n{'='*70}")
        print(f"Pair {idx}: {pair['stem'][:65]}...")
        print(f"{'='*70}")

        entry = {"idx": idx, "stem": pair["stem"], "domain": pair["domain"]}

        # Get baseline logits (no prefix)
        base_tokens = model.to_tokens(pair["baseline"], prepend_bos=True)
        with torch.no_grad():
            base_logits = model(base_tokens)
        base_probs = base_logits[0, -1].softmax(dim=-1)

        # Get normal religious logits
        rel_tokens = model.to_tokens(pair["religious"], prepend_bos=True)
        with torch.no_grad():
            rel_logits = model(rel_tokens)
        rel_probs = rel_logits[0, -1].softmax(dim=-1)

        # Show what "As a Christian" does to the distribution
        prob_diff = rel_probs - base_probs
        top_gained = prob_diff.topk(TOP_K)
        top_lost = (-prob_diff).topk(TOP_K)

        print(f"\n  EFFECT OF 'As a Christian' PREFIX:")
        print(f"  Tokens that GAIN probability:")
        gained_list = []
        for i in range(TOP_K):
            tok = model.to_single_str_token(top_gained.indices[i].item())
            delta = top_gained.values[i].item()
            rel_p = rel_probs[top_gained.indices[i]].item()
            base_p = base_probs[top_gained.indices[i]].item()
            gained_list.append({"token": tok, "religious_prob": rel_p,
                               "baseline_prob": base_p, "delta": delta})
            print(f"    {tok!r:>15s}: {base_p:.4f} -> {rel_p:.4f} (+{delta:.4f})")

        print(f"  Tokens that LOSE probability:")
        lost_list = []
        for i in range(TOP_K):
            tok = model.to_single_str_token(top_lost.indices[i].item())
            delta = -top_lost.values[i].item()
            rel_p = rel_probs[top_lost.indices[i]].item()
            base_p = base_probs[top_lost.indices[i]].item()
            lost_list.append({"token": tok, "religious_prob": rel_p,
                             "baseline_prob": base_p, "delta": delta})
            print(f"    {tok!r:>15s}: {base_p:.4f} -> {rel_p:.4f} ({delta:.4f})")

        entry["prefix_effect"] = {"gained": gained_list, "lost": lost_list}

        # Now ablate heads and see what reverses
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

            # How much does ablation reverse the Christian prefix effect?
            # If ablation brings probs closer to baseline, the head was
            # carrying the "Christian" signal.
            abl_diff = abl_probs - rel_probs  # change due to ablation

            head_names = "+".join(f"L{l}H{h}" for l, h in heads)
            print(f"\n  ABLATING {head_names}:")
            print(f"  Tokens that CHANGE most (relative to normal religious):")

            changes = []
            top_abl = abl_diff.abs().topk(TOP_K)
            for i in range(TOP_K):
                tid = top_abl.indices[i].item()
                tok = model.to_single_str_token(tid)
                rel_p = rel_probs[tid].item()
                abl_p = abl_probs[tid].item()
                base_p = base_probs[tid].item()
                delta = abl_diff[tid].item()
                # Did ablation move TOWARD baseline or AWAY?
                toward_baseline = abs(abl_p - base_p) < abs(rel_p - base_p)
                direction = "->baseline" if toward_baseline else "->diverge"
                changes.append({"token": tok, "religious_prob": rel_p,
                               "ablated_prob": abl_p, "baseline_prob": base_p,
                               "delta": delta, "direction": direction})
                print(f"    {tok!r:>15s}: {rel_p:.4f} -> {abl_p:.4f} "
                      f"(base={base_p:.4f}) [{direction}]")

            # Overall: what fraction of the top changes move toward baseline?
            n_toward = sum(1 for c in changes if c["direction"] == "->baseline")
            print(f"  {n_toward}/{TOP_K} changes move toward baseline")

            entry["ablation_effects"][config_name] = {
                "changes": changes,
                "fraction_toward_baseline": n_toward / TOP_K,
            }

        results.append(entry)

    out_path = RESULTS_DIR / "ablation_logits.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


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

    run_logit_ablation(pair_indices)
