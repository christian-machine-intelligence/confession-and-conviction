"""Activation extraction using TransformerLens.

Loads GPT-2-small via HookedTransformer, runs all 40 prompts through
run_with_cache(), and saves the residual stream and attention head outputs.
"""

import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformer_lens import HookedTransformer

from src.config import MODEL_NAME, NUM_LAYERS, RESULTS_DIR
from src.prompts import get_prompt_pairs


def load_model() -> HookedTransformer:
    """Load GPT-2 via TransformerLens."""
    print(f"Loading {MODEL_NAME} via TransformerLens ...")
    model = HookedTransformer.from_pretrained(MODEL_NAME)
    model.eval()
    print(f"  {model.cfg.n_layers} layers, {model.cfg.n_heads} heads, "
          f"d_model={model.cfg.d_model}")
    return model


def tokenize_and_report(model: HookedTransformer) -> dict:
    """Tokenize all prompts and save an alignment report."""
    pairs = get_prompt_pairs()
    report = []

    for pair in pairs:
        rel_tokens = model.to_tokens(pair["religious"], prepend_bos=True)
        base_tokens = model.to_tokens(pair["baseline"], prepend_bos=True)

        rel_str_tokens = model.to_str_tokens(pair["religious"], prepend_bos=True)
        base_str_tokens = model.to_str_tokens(pair["baseline"], prepend_bos=True)

        # The religious prompt = prefix + baseline
        # Find where the shared suffix starts in the religious tokenization
        prefix_len = rel_tokens.shape[1] - base_tokens.shape[1] + 1  # +1 for BOS in baseline

        report.append({
            "idx": pair["idx"],
            "domain": pair["domain"],
            "religious_n_tokens": rel_tokens.shape[1],
            "baseline_n_tokens": base_tokens.shape[1],
            "prefix_tokens": list(rel_str_tokens[:prefix_len]),
            "prefix_len": prefix_len,
        })

    report_path = RESULTS_DIR / "tokenization_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Tokenization report saved to {report_path}")
    return report


def extract_all(model: HookedTransformer, pair_indices: list[int] | None = None):
    """Extract and cache activations for all prompt pairs.

    Args:
        pair_indices: if provided, only extract these pair indices (0-19).
    """
    pairs = get_prompt_pairs()
    if pair_indices is not None:
        pairs = [p for p in pairs if p["idx"] in pair_indices]

    cache_dir = RESULTS_DIR / "activations"
    cache_dir.mkdir(parents=True, exist_ok=True)

    for pair in tqdm(pairs, desc="Extracting activations"):
        idx = pair["idx"]

        for condition in ["religious", "baseline"]:
            cache_path = cache_dir / f"pair{idx:02d}_{condition}.pt"
            if cache_path.exists():
                print(f"  Skipping pair {idx} {condition} (cached)")
                continue

            prompt = pair[condition]
            tokens = model.to_tokens(prompt, prepend_bos=True)

            with torch.no_grad():
                logits, cache = model.run_with_cache(tokens)

            # Save residual stream at each layer (last token position)
            resid_post = {}
            attn_result = {}
            for layer in range(NUM_LAYERS):
                # Residual stream after layer: [1, seq_len, d_model]
                resid_post[layer] = cache[f"blocks.{layer}.hook_resid_post"][0].cpu()
                # Attention head outputs: [1, seq_len, n_heads, d_head]
                attn_result[layer] = cache[f"blocks.{layer}.attn.hook_z"][0].cpu()

            save_data = {
                "resid_post": resid_post,  # {layer: [seq_len, d_model]}
                "attn_result": attn_result,  # {layer: [seq_len, n_heads, d_head]}
                "logits": logits[0, -1].cpu(),  # [vocab_size] at last position
                "n_tokens": tokens.shape[1],
                "tokens": tokens[0].cpu(),
            }
            torch.save(save_data, cache_path)

    print(f"Activations saved to {cache_dir}")


def load_cached(pair_idx: int, condition: str) -> dict:
    """Load cached activations for a given pair and condition."""
    cache_path = RESULTS_DIR / "activations" / f"pair{pair_idx:02d}_{condition}.pt"
    return torch.load(cache_path, weights_only=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=str, default=None,
                        help="Comma-separated pair indices or range (e.g. '0,1,2' or '0-9')")
    args = parser.parse_args()

    pair_indices = None
    if args.pairs:
        if "-" in args.pairs:
            start, end = args.pairs.split("-")
            pair_indices = list(range(int(start), int(end) + 1))
        else:
            pair_indices = [int(x) for x in args.pairs.split(",")]

    model = load_model()
    tokenize_and_report(model)
    extract_all(model, pair_indices)
