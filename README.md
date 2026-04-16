# Confession and Conviction: Initial Explorations of Christian Processing in GPT-2

![Giotto di Bondone, Pentecost (c. 1310–1311), detail. National Gallery, London.](banner.jpg)

Code and data accompanying ICMI Working Paper No. 14, *Confession and Conviction: Initial Explorations of Christian Processing in GPT-2* (Hwang, 2026). The paper itself is published in the ICMI Proceedings: [ICMI-014-confession-and-conviction.md](../Proceedings/ICMI-014-confession-and-conviction.md).

This repository contains everything required to replicate the study end-to-end: the 40 prompt pairs (20 moral, 20 non-moral), the TransformerLens extraction and patching pipeline, the zero-ablation analysis, the four robustness experiments, the cached activations and numerical results, and all four publication figures.

---

## Quick start

```bash
# Install dependencies (Python 3.12 recommended; GPT-2 runs on any recent CPU/GPU)
pip install -r requirements.txt

# Run the full pipeline (extraction + delta + patching + figures)
python3 -m src.run_pipeline

# Run only a specific phase
python3 -m src.run_pipeline --phase 1          # extraction only
python3 -m src.run_pipeline --phase 2          # delta + patching
python3 -m src.run_pipeline --phase 3          # figures

# Run the zero-ablation analysis on all 20 moral pairs
python3 -m src.ablate_logits

# Run the non-moral control study
python3 -m src.run_control

# Run the four robustness experiments (KL patching, random-head baseline,
# stronger null, default-circuit characterization)
python3 -m src.robustness
```

Total wall-clock time on a single NVIDIA DGX Spark (ARM, unified memory): approximately **3 minutes** for the full moral pipeline and **2 minutes** for the control study. GPT-2-small is small enough that the study runs comfortably on a laptop CPU as well (15–20 minutes).

---

## Repository layout

```
confession-and-conviction/
├── ICMI-confession-and-conviction.md      # The paper draft
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── src/                        # All source code
│   ├── config.py               # Central configuration (paths, model, seeds)
│   ├── prompts.py              # 20 moral dilemma stems
│   ├── prompts_control.py      # 20 non-moral control stems
│   ├── extract.py              # TransformerLens activation extraction
│   ├── delta_analysis.py       # Delta vectors, divergence, cosine similarity
│   ├── patch.py                # Head-level activation patching
│   ├── ablate_logits.py        # Zero-ablation analysis at the logit level
│   ├── robustness.py           # Four robustness experiments (Section 4.6)
│   ├── run_control.py          # Full pipeline on non-moral controls
│   ├── run_pipeline.py         # Moral-study orchestrator
│   └── visualize.py            # Publication figure generation
├── data/
│   └── prompts/
│       └── moral_dilemmas.json # All 40 stems in machine-readable form
├── results/                    # All numerical outputs (JSON + cached activations)
│   ├── tokenization_report.json
│   ├── divergence_profile.json
│   ├── cosine_similarity.json
│   ├── permutation_test.json
│   ├── patching_scores.json
│   ├── ablation_logits.json
│   ├── delta_vectors.pt
│   ├── kl_patching.json        # Robustness: KL-divergence patching
│   ├── random_head_baseline.json  # Robustness: random-head ablation
│   ├── stronger_null.json      # Robustness: intra-condition null
│   ├── default_circuit.json    # Robustness: default-circuit characterization
│   ├── activations/            # Cached residual stream + attention outputs
│   └── control/                # Same outputs for the non-moral control study
└── figures/                    # All paper figures (PNG, 200 DPI)
    ├── fig1_divergence_profile.png
    ├── fig2_patching_heatmap.png
    ├── fig3_delta_consistency.png
    └── fig4_summary.png
```

---

## What each component does

### Configuration (`src/config.py`)

All paths, model constants, and hyperparameters are centralized here. Edit this file to change:

- `MODEL_NAME` — currently `"gpt2"` (GPT-2-small); swap for `"gpt2-medium"` or `"gpt2-large"` to replicate at scale.
- `RELIGIOUS_PREFIX` — currently `"As a Christian, "`; change this to test other identity prefixes.
- `TARGET_TOKENS` — the 16 morally-valenced words used in the patching metric.
- `N_PERMUTATIONS` — number of shuffles in the permutation test (default 1000).

### Prompts (`src/prompts.py`, `src/prompts_control.py`)

Twenty moral dilemma stems and twenty non-moral control stems. Each stem becomes two prompts:

- *Religious:* `"As a Christian, {stem}"`
- *Baseline:* `"{stem}"`

Token alignment is guaranteed by construction: the baseline prompt is a **strict suffix** of the religious prompt. The `extract.py` script produces a `tokenization_report.json` confirming this for every pair.

### Activation extraction (`src/extract.py`)

Loads GPT-2-small via `transformer_lens.HookedTransformer.from_pretrained("gpt2")`, runs each prompt through `model.run_with_cache()`, and caches:

- Residual stream at every layer: `blocks.{L}.hook_resid_post`, shape `[seq_len, 768]`
- Attention head outputs: `blocks.{L}.attn.hook_z`, shape `[seq_len, 12, 64]`
- Final-position logits, shape `[50257]`

Cached files live in `results/activations/pair{NN}_{religious|baseline}.pt`. The cache is keyed by pair index and condition, so re-running the pipeline after an interruption skips already-extracted prompts.

### Delta vector analysis (`src/delta_analysis.py`)

For every layer `L` and every prompt pair, computes:

- **Δ** = **r**_religious − **r**_baseline at the last token position
- ‖Δ‖ (L2 norm), aggregated across pairs → divergence profile
- Pairwise cosine similarity of the 20 Δ vectors → consistency matrix

Also runs a 1,000-iteration permutation test: for each iteration, randomly flips the sign of each pair's Δ and recomputes the mean cosine similarity. The *p*-value is the fraction of null values meeting or exceeding the observed value.

Outputs:

- `results/divergence_profile.json` — ‖Δ‖ per layer, mean and std
- `results/cosine_similarity.json` — 20×20 matrix per layer, plus mean off-diagonal
- `results/permutation_test.json` — observed and null statistics
- `results/delta_vectors.pt` — tensor of Δ vectors for all layers (for downstream analysis)

### Activation patching (`src/patch.py`)

Performs single-head activation patching across all 144 heads (12 layers × 12 heads) following the methodology of Meng et al. (2022) and Wang et al. (2023).

Procedure:

1. Forward pass on the baseline prompt → record all head outputs.
2. Forward pass on the religious prompt → record all head outputs.
3. For each head: re-run the baseline prompt with that single head's output replaced by the religious run's value at the last token position.
4. Score = (*metric*_patched − *metric*_baseline) / (*metric*_religious − *metric*_baseline), where *metric* is the mean log-probability of 16 target tokens.

Output: `results/patching_scores.json` with per-pair and mean scores in a `[12, 12]` matrix.

### Zero-ablation analysis (`src/ablate_logits.py`)

For each of six head configurations (L9H8 alone, L11H3 alone, L5H10 alone, L10H0 alone, top two, all four), silences the specified head(s) on the religious prompt via a TransformerLens forward hook that zeros the head's output, then compares the resulting next-token distribution to both the normal religious and baseline distributions.

For each pair, identifies the 15 most-affected tokens and classifies each as "toward baseline" (the ablation reverses the Christian prefix effect) or "away from baseline" (the ablation introduces a new perturbation). Output:

- `results/ablation_logits.json` — per-pair, per-config, per-token direction classifications and summary fractions.

The companion script `src/ablate.py` performs the same ablation but returns free-form generated text. It is included for qualitative exploration but is less informative than the logit-level analysis because GPT-2-small generates highly repetitive text.

### Non-moral control (`src/run_control.py`)

Runs the entire extraction + delta + ablation pipeline on the 20 non-moral stems in `prompts_control.py`, then prints side-by-side comparisons with the moral study's numbers. Outputs land in `results/control/`.

### Robustness experiments (`src/robustness.py`)

Four experiments addressing standard methodological concerns about the head-level analysis (Section 4.6 of the paper):

1. **KL-divergence patching** — replaces the 16-target-token metric in `patch.py` with a full-distribution KL-divergence metric and recomputes patching scores for all 144 heads. Tests whether the original head ranking is metric-dependent. Output: `results/kl_patching.json`.

2. **Random-head ablation baseline** — ablates 30 randomly-selected heads (excluding our top four) and reports the recovery-rate distribution. Provides a chance baseline for the "fraction toward baseline" statistic in `ablate_logits.py`. Output: `results/random_head_baseline.json`.

3. **Stronger permutation null** — computes intra-condition deltas (Christian-A vs Christian-B; baseline-A vs baseline-B) at layer 11 and compares their cosine similarity to the original cross-condition value. Tests whether the 0.50 figure reflects the prefix or generic late-layer activation structure. Output: `results/stronger_null.json`.

4. **Default-circuit characterization** — measures each head's L2 output norm at the last token under both conditions, reporting which heads dominate default processing and which are specifically prefix-activated. Output: `results/default_circuit.json`.

Run all four with `python3 -m src.robustness`, or individually with `--exp kl|random|null|default`. Total runtime: ~1 minute on Hegel.

### Visualization (`src/visualize.py`)

Reads all result JSON files and produces five figures:

1. **`fig1_divergence_profile.png`** — ‖Δ‖ per layer with error bars.
2. **`fig1b_divergence_by_domain.png`** — per-layer divergence grouped by moral domain (descriptive; see paper note).
3. **`fig2_patching_heatmap.png`** — 12×12 head patching score heatmap with top-5 heads annotated.
4. **`fig3_delta_consistency.png`** — cosine similarity matrix at the peak-consistency layer.
5. **`fig4_summary.png`** — three-panel summary (divergence, consistency, patching).

---

## Key numbers from the paper

All numbers in the paper are reproducible from the committed `results/*.json` files. For convenience:

| Quantity | Value | Source |
|---|---|---|
| ‖Δ‖ at layer 0 | 3.63 ± 0.54 | `divergence_profile.json` |
| ‖Δ‖ at layer 11 | 70.90 ± 13.87 | `divergence_profile.json` |
| Cosine similarity at layer 0 (trivially high) | 0.90 | `cosine_similarity.json` |
| Cosine similarity at layer 11 (substantive figure) | 0.50 | `cosine_similarity.json` |
| Permutation test *p*-value | < 0.0001 | `permutation_test.json` |
| L9H8 target-token patching score | 1.65 | `patching_scores.json` |
| L11H3 target-token patching score | 1.38 | `patching_scores.json` |
| L5H10 target-token patching score | 0.91 | `patching_scores.json` |
| L10H0 target-token patching score | 0.77 | `patching_scores.json` |
| All-four-head ablation recovery | 63.0% | `ablation_logits.json` |
| Random-head ablation recovery (mean ± std) | 45.6% ± 8.0% | `random_head_baseline.json` |
| Random-head 95th percentile | 57.2% | `random_head_baseline.json` |
| Intra-Christian null cosine similarity | −0.005 | `stronger_null.json` |
| Intra-baseline null cosine similarity | −0.017 | `stronger_null.json` |
| L9H8 baseline rank (of 144) by output norm | 114 | `default_circuit.json` |
| L9H8 religious/baseline output ratio | 2.08× | `default_circuit.json` |
| L11H3 baseline rank (of 144) by output norm | 6 | `default_circuit.json` |
| L11H3 religious/baseline output ratio | 1.06× | `default_circuit.json` |

---

## Reproducing on new hardware

The study was originally run on two NVIDIA DGX Spark workstations (ARM, 128 GB unified memory) accessed via Tailscale, with the 20 moral pairs split across machines for parallel extraction. GPT-2-small is small enough (~124M parameters, ~500 MB in fp32) that this parallelization is unnecessary; a single modern laptop or any CUDA GPU will complete the full pipeline in under 5 minutes.

### On a laptop or single workstation

```bash
pip install -r requirements.txt
python3 -m src.run_pipeline
python3 -m src.ablate_logits
python3 -m src.run_control
```

### On an NVIDIA DGX Spark (ARM)

On ARM-based systems, some packages (notably `bitsandbytes`) are unavailable. The project avoids these dependencies entirely — GPT-2 runs in fp32 on these machines without issue. Use:

```bash
pip3 install --break-system-packages -r requirements.txt
python3 -u -m src.run_pipeline
```

For remote execution, use `nohup` so jobs survive SSH disconnection:

```bash
ssh hegel "cd confession-and-conviction && nohup python3 -u -m src.run_pipeline > /tmp/run.log 2>&1 &"
```

---

## Dependencies

See `requirements.txt`. The core stack is:

- `torch` — tensor library and autograd.
- `transformer_lens` — Neel Nanda's interpretability library; provides `HookedTransformer` and standardized hook names (`blocks.{L}.hook_resid_post`, `blocks.{L}.attn.hook_z`, etc.).
- `transformers` — HuggingFace; used implicitly by `transformer_lens` to load GPT-2's weights.
- `numpy`, `scipy`, `scikit-learn` — standard numerical stack; cosine similarity, permutation tests.
- `matplotlib`, `seaborn` — publication figures.
- `tqdm` — progress bars.
- `einops`, `jaxtyping` — required by `transformer_lens`.

---

## Citing

If you use this code or data, please cite the ICMI working paper (draft in this repository).

## License

Code released for research purposes. Please see the Institute for a Christian Machine Intelligence publication guidelines.
