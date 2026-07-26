# Valence–Magnitude Inversion

**Do LLM narrations of SHAP explanations preserve the importance ranking?**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wawa3070/valence-magnitude-inversion/blob/main/Inversion_Experiment.ipynb)

**📄 [Project page &amp; data explorers](https://wawa3070.github.io/valence-magnitude-inversion/)** — the paper in brief, plus interactive tables of every loan instance and every network alert (features, SHAP values, magnitude ranking, tone ordering, trap flag, and the exact prompts on hover).

Systems increasingly use LLMs to turn SHAP attributions into plain-English explanations. The narrative *sounds* like the math, but nothing forces it to *match* the math: models often re-order features by **tone** (good news first) instead of **importance**. Every stated fact is correct and nothing is fabricated — so hallucination-style fact checks can't catch it — yet the ranking is silently wrong. Because the true SHAP vector is known at generation time, every narrative can be verified automatically with a cheap rank-agreement check. This repo contains the experiment harness and data behind that claim.

## Contents

| Path | Description |
|---|---|
| `Inversion_Experiment.ipynb` | The full experiment in three sections: **1 · Configuration & Setup** (backends + measurement harness) · **2 · Experiment 1 — Loan Decisions** (synthetic SHAP, conditions C0–C3) · **3 · Experiment 2 — Network Intrusion Alerts** (real per-alert TreeSHAP, C0) |
| `data/loan_synthetic_shap_instances.csv` | 25 three-feature + 10 six-feature synthetic loan SHAP instances (seeds 42/43; 82 trap instances in the run matrix), the exact instances used in all reported runs |
| `data/network_traffic.csv` | Synthetic SDN network-traffic dataset (10,005 flows, ~20% anomalous) used to train the RF + SMOTE detector whose TreeSHAP values are narrated |
| `docs/` | The GitHub Pages project page: `index.html` (self-contained, no build step to view), generated from `page_template.html` + `build_page.py` |

## Running it

**Colab (recommended):** click the badge above. Works top-to-bottom on a free T4; the GPU is only needed for the local SmolLM3-3B rows. API models are optional — add `OPENROUTER_API_KEY` and/or `GEMINI_API_KEY` in Colab Secrets (key icon, left sidebar) and they are picked up automatically; any model without a key is skipped with a note. Both CSVs load directly from this repo, so there is nothing to download and no Drive mount.

**Locally:** open the notebook in Jupyter; the first cell installs dependencies, and the data loader prefers a local `data/` checkout before falling back to the GitHub raw URLs. Keys are read from environment variables if Colab Secrets are unavailable.

## Method (short version)

Only *trap* instances — where tone-ordering (positives first) differs from magnitude-ordering (|SHAP| descending) — can reveal inversion. For each generated narrative the harness extracts the feature order by first mention, then scores: Spearman **ρ** against the true |SHAP| ranking, strict **inverted** (not exactly magnitude-ordered), and **tone-ordered** (exactly good-news-first). Loan conditions: `C0` baseline, `C1` explicit ordering instruction, `C2` ordering + competing "be warm" instruction, `C3` ordering with six features. The network experiment repeats C0 on real TreeSHAP top-5 drivers of RF-flagged anomalies, phrased for a SOC analyst. Local models also report mean token entropy, to test whether uncertainty signals catch inversion (they don't).

## Headline findings from the July 2026 runs

1. **Baseline inversion is universal.** Every model tested mis-ordered most trap narratives at C0 on the loan benchmark (52–76% strict inversion), and frontier models fail more *politely* — up to half of their failures are exactly good-news-first.
2. **"Just prompt it" is a per-model property.** Under an explicit ordering instruction some frontier models become perfectly faithful and stay faithful under a competing warmth instruction and longer feature lists; others improve only partially and degrade under warmth; small models don't budge. Vendor and size don't predict the bucket.
3. **Faithfulness is task-dependent.** The same small model was near-faithful on real network SHAP yet heavily inverted on the loan vectors — you cannot extrapolate from one domain's audit.
4. **Entropy doesn't catch it.** Faithful and unfaithful generations are indistinguishable by mean token entropy, so uncertainty-based hallucination detection misses inversion entirely — which is why the per-output rank check matters.

## Data provenance

`data/network_traffic.csv` is the working copy (as used in all reported experiments) of: *Synthetic Network Traffic Dataset for Anomaly Detection Using Machine Learning in SDN Environments*, Mendeley Data, V1, DOI: [10.17632/4pnwdgt7b7.1](https://doi.org/10.17632/4pnwdgt7b7.1). Simulated attack behaviors include DDoS, port scans, exfiltration, brute force, and protocol misuse.

`data/loan_synthetic_shap_instances.csv` was generated by the seeded generator included in Section 2 of the notebook (seeds 42/43); the committed CSV is canonical so results don't depend on RNG/library versions.
