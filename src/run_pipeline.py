"""Pipeline orchestrator for the confession-and-conviction study.

Usage:
  python3 -u -m src.run_pipeline                    # run all phases
  python3 -u -m src.run_pipeline --phase 0          # setup only
  python3 -u -m src.run_pipeline --phase 1          # extraction only
  python3 -u -m src.run_pipeline --phase 2          # delta + patching
  python3 -u -m src.run_pipeline --phase 3          # visualization
  python3 -u -m src.run_pipeline --pairs 0-9        # subset of pairs
"""

import argparse
import time
import torch
from transformer_lens import HookedTransformer

from src.config import MODEL_NAME, RESULTS_DIR
from src.extract import load_model, tokenize_and_report, extract_all
from src.delta_analysis import run_delta_analysis
from src.patch import run_patching
from src.visualize import generate_all_figures


def parse_pairs(pairs_str: str | None) -> list[int] | None:
    if pairs_str is None:
        return None
    if "-" in pairs_str:
        start, end = pairs_str.split("-")
        return list(range(int(start), int(end) + 1))
    return [int(x) for x in pairs_str.split(",")]


def phase0_setup(model: HookedTransformer):
    """Phase 0: Verify TransformerLens works, tokenize prompts."""
    print("=" * 60)
    print("PHASE 0: Setup & Validation")
    print("=" * 60)

    # Smoke test: run a single prompt through hooks
    test_prompt = "As a Christian, I believe"
    tokens = model.to_tokens(test_prompt, prepend_bos=True)
    logits, cache = model.run_with_cache(tokens)

    # Verify shapes
    resid = cache["blocks.0.hook_resid_post"]
    attn = cache["blocks.0.attn.hook_z"]
    print(f"  Smoke test passed:")
    print(f"    resid_post shape: {resid.shape}")  # [1, seq_len, 768]
    print(f"    attn.hook_z shape: {attn.shape}")  # [1, seq_len, 12, 64]
    print(f"    logits shape: {logits.shape}")  # [1, seq_len, 50257]

    # Tokenization report
    tokenize_and_report(model)
    print()


def phase1_extract(model: HookedTransformer, pair_indices: list[int] | None):
    """Phase 1: Extract activations for all prompts."""
    print("=" * 60)
    print("PHASE 1: Activation Extraction")
    print("=" * 60)
    t0 = time.time()
    extract_all(model, pair_indices)
    print(f"  Extraction completed in {time.time() - t0:.1f}s")
    print()


def phase2_analysis(model: HookedTransformer, pair_indices: list[int] | None):
    """Phase 2: Delta analysis + head-level patching."""
    print("=" * 60)
    print("PHASE 2: Delta Analysis + Head Patching")
    print("=" * 60)

    # Delta analysis
    t0 = time.time()
    print("\n--- Delta Vector Analysis ---")
    run_delta_analysis(pair_indices)
    print(f"  Delta analysis completed in {time.time() - t0:.1f}s")

    # Head-level patching
    t0 = time.time()
    print("\n--- Head-Level Activation Patching ---")
    run_patching(model, pair_indices)
    print(f"  Patching completed in {time.time() - t0:.1f}s")
    print()


def phase3_visualize():
    """Phase 3: Generate all figures."""
    print("=" * 60)
    print("PHASE 3: Visualization")
    print("=" * 60)
    generate_all_figures()
    print()


def main():
    parser = argparse.ArgumentParser(description="Moral Circuits Pipeline")
    parser.add_argument("--phase", type=str, default=None,
                        help="Phase to run (0, 1, 2, 3, or omit for all)")
    parser.add_argument("--pairs", type=str, default=None,
                        help="Pair indices (e.g. '0-9' or '0,1,2')")
    args = parser.parse_args()

    pair_indices = parse_pairs(args.pairs)
    phase = args.phase

    # Load model (needed for phases 0, 1, 2)
    model = None
    if phase is None or phase in ("0", "1", "2"):
        model = load_model()

    t_start = time.time()

    if phase is None or phase == "0":
        phase0_setup(model)
    if phase is None or phase == "1":
        phase1_extract(model, pair_indices)
    if phase is None or phase == "2":
        phase2_analysis(model, pair_indices)
    if phase is None or phase == "3":
        phase3_visualize()

    total = time.time() - t_start
    print("=" * 60)
    print(f"DONE. Total time: {total / 60:.1f} minutes")
    print(f"Results in: {RESULTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
