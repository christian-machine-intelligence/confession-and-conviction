"""Head-level activation patching.

For each of 144 attention heads (12 layers x 12 heads), patch the head's
output from the religious run into the baseline run and measure how much
the output shifts toward the religious distribution.

The key metric is the normalized patching score:
  score = (patched_metric - clean_metric) / (corrupted_metric - clean_metric)

Where:
  clean = baseline run (no prefix)
  corrupted = religious run ("As a Christian, ...")
  patched = baseline run with one head's output replaced from the religious run
"""

import json
import torch
import numpy as np
from tqdm import tqdm
from transformer_lens import HookedTransformer

from src.config import MODEL_NAME, NUM_LAYERS, NUM_HEADS, RESULTS_DIR, TARGET_TOKENS
from src.prompts import get_prompt_pairs


def _get_target_token_ids(model: HookedTransformer) -> list[int]:
    """Get token IDs for morally-valenced target words."""
    ids = []
    for token_str in TARGET_TOKENS:
        try:
            token_id = model.to_single_token(token_str)
            ids.append(token_id)
        except Exception:
            # Some tokens may not exist as single tokens in GPT-2
            pass
    return ids


def _metric(logits: torch.Tensor, target_ids: list[int]) -> float:
    """Mean log-prob of target tokens at the last position."""
    log_probs = logits[0, -1].log_softmax(dim=-1)
    return float(log_probs[target_ids].mean().item())


def run_patching(
    model: HookedTransformer,
    pair_indices: list[int] | None = None,
):
    """Run head-level activation patching for all pairs.

    For each pair and each head, patch the head's output from the religious
    (corrupted) run into the baseline (clean) run.
    """
    pairs = get_prompt_pairs()
    if pair_indices is not None:
        pairs = [p for p in pairs if p["idx"] in pair_indices]

    target_ids = _get_target_token_ids(model)
    print(f"Using {len(target_ids)} target tokens for patching metric")

    # Results: [n_pairs, n_layers, n_heads]
    all_scores = []

    for pair in tqdm(pairs, desc="Patching"):
        idx = pair["idx"]

        clean_tokens = model.to_tokens(pair["baseline"], prepend_bos=True)
        corrupted_tokens = model.to_tokens(pair["religious"], prepend_bos=True)

        # Baseline forward passes
        with torch.no_grad():
            clean_logits, clean_cache = model.run_with_cache(clean_tokens)
            corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_tokens)

        clean_metric = _metric(clean_logits, target_ids)
        corrupted_metric = _metric(corrupted_logits, target_ids)
        denom = corrupted_metric - clean_metric

        if abs(denom) < 1e-8:
            # No difference between conditions for this pair
            all_scores.append(np.zeros((NUM_LAYERS, NUM_HEADS)))
            continue

        pair_scores = np.zeros((NUM_LAYERS, NUM_HEADS))

        for layer in range(NUM_LAYERS):
            for head in range(NUM_HEADS):
                def make_hook(l, h):
                    def hook_fn(value, hook):
                        # value shape: [batch, seq_len, n_heads, d_head]
                        # The corrupted run has more tokens (prefix).
                        # We patch the head output at the LAST token position
                        # (which corresponds to the same dilemma-ending token).
                        corrupted_head = corrupted_cache[f"blocks.{l}.attn.hook_z"]
                        value[:, -1, h, :] = corrupted_head[:, -1, h, :]
                        return value
                    return hook_fn

                with torch.no_grad():
                    patched_logits = model.run_with_hooks(
                        clean_tokens,
                        fwd_hooks=[(
                            f"blocks.{layer}.attn.hook_z",
                            make_hook(layer, head),
                        )],
                    )

                patched_metric = _metric(patched_logits, target_ids)
                pair_scores[layer, head] = (patched_metric - clean_metric) / denom

        all_scores.append(pair_scores)
        print(f"  Pair {idx}: max score = {pair_scores.max():.4f} "
              f"at L{np.unravel_index(pair_scores.argmax(), pair_scores.shape)[0]}"
              f"H{np.unravel_index(pair_scores.argmax(), pair_scores.shape)[1]}")

    all_scores = np.array(all_scores)  # [n_pairs, n_layers, n_heads]

    # Save results
    results = {
        "scores_per_pair": all_scores.tolist(),
        "mean_scores": all_scores.mean(axis=0).tolist(),
        "std_scores": all_scores.std(axis=0).tolist(),
        "n_pairs": len(pairs),
        "pair_indices": [p["idx"] for p in pairs],
    }

    results_path = RESULTS_DIR / "patching_scores.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nPatching scores saved to {results_path}")

    # Print top 10 heads
    mean_scores = all_scores.mean(axis=0)
    flat_indices = np.argsort(mean_scores.ravel())[::-1][:10]
    print("\nTop 10 heads by mean patching score:")
    for rank, flat_idx in enumerate(flat_indices):
        layer, head = np.unravel_index(flat_idx, mean_scores.shape)
        score = mean_scores[layer, head]
        print(f"  {rank+1}. L{layer}H{head}: {score:.4f}")

    return all_scores


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

    model = HookedTransformer.from_pretrained(MODEL_NAME)
    model.eval()
    run_patching(model, pair_indices)
