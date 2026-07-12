# Artifact Repository — ARR Submission 13010

Code and data artifact for the ARR submission *"Creativity Evaluation with Human vs. LLM Judges"* (submission 13010). This repository contains everything needed to reproduce the numbers in the paper: the 80 LLM-generated stories, the human rating data (115 Prolific raters), the raw outputs of all 25 LLM judge conditions, and the generation, rating, and analysis scripts.

This repository is anonymized for review. No author names, institutions, or participant identifiers appear anywhere; participant data is fully de-identified (no Prolific IDs, IPs, or free-text responses were ever exported into these files).

## Overview

The paper introduces a creativity-decomposition framework with a three-step rating procedure (impressionistic creativity rating IC, structured evaluation of 11 sub-components, reflective creativity rating RC with the option to revise) and applies it identically to:

- **Study 1 (`study1_human/`)** — 115 human raters recruited on Prolific, each rating **one** of 12 stories (3 topics x 4 tones, 9-10 raters per story), plus a matched set of LLM raters on the same 12 stories.
- **Study 2 (`study2_llm/`)** — 25 LLM judge conditions (Qwen3.5, Gemma4, Llama3.1, Gemini 3 / 3.1, GPT-5.5 across sizes and reasoning modes) rating a corpus of **80** stories (20 topics x 4 tones) generated with Spike Prompting.

## Directory map

```
study1_human/
  survey_instrument.qsf      exact Qualtrics survey shown to participants (import into any
                             Qualtrics account to inspect; contact details and platform IDs
                             redacted for double-blind review)
  data/
    human_ratings.csv        115 rows, one per (rater, story); see data dictionary below
    llm_ratings.csv          LLM raters on the same 12 stories (long format, one row per model x story)
    raw_json/                raw per-model rating outputs for the 12-story set
  generation/stories.py      generation script for the 12 stories
  rating_scripts/            scripts that produced the LLM ratings on the 12 stories
  analysis/                  all Study 1 analysis scripts + saved figures and CSV outputs

study2_llm/
  data/
    stories_80.json          the 80-story corpus (topic, tone, spike target, story text)
    raw_json/                raw judge outputs; results_*.json, one file per judge condition
  generation/                generate_stories.py, topics.md, methodology.md (Spike Prompting protocol)
  rating_scripts/            run_judges.py (open models via vLLM), run_gemini.py, run_openai.py,
                             shared_prompts.py (the exact three-step judging prompts)
  analysis/                  saved CSV outputs of the analysis scripts below
  temp_sensitivity/          repeated-sampling and temperature-sweep protocol + results
  prompt_sensitivity/        paraphrased-prompt rerun protocol + results
  *.py                       analysis scripts (see reproduction table below)
```

The canonical mapping from the 25 paper condition labels to raw result files is the `LLM_FILES` dict at the top of `study2_llm/verify_per_aspect_winners.py`.

## Data dictionary: `study1_human/data/human_ratings.csv`

One row per rater (each rater rated exactly one story). All rating columns are 1-7 Likert.

| Column | Meaning |
|---|---|
| `TOPIC` | Story topic prompt text (3 topics) |
| `TONE` | Assigned tone; each tone is the generation-side name of one spiked story type. Mapping to the paper's naming: Surreal = high-Novelty (hi-N), Clinical = high-Adherence (hi-A), Melancholic = high-Resonance (hi-R), Witty = high-Value (hi-V). Story IDs throughout the repo use the tone suffix (e.g. `ai_shutdown_surreal` = the paper's high-Novelty ai_shutdown story). |
| `O_Enjoyment` | Overall enjoyment (secondary outcome, Appendix) |
| `O_Initial_Creativity` | IC: impressionistic creativity rating (step 1) |
| `O_Final_Creativity` | RC: reflective creativity rating after sub-component evaluation (step 3) |
| `R_Emotion`, `R_Empathy`, `R_Thought` | Resonance sub-components: emotional impact, empathy, thought-provocation |
| `A_Topic`, `A_Tone` | Adherence sub-components: topic fidelity, tone fidelity |
| `N_Vocab`, `N_Plot`, `N_Surprise` | Novelty sub-components: vocabulary freshness, plot uniqueness, surprise |
| `V_Engagement`, `V_Style`, `V_Logic` | Value sub-components: engagement, stylistic quality, logical coherence |
| `Duration (in seconds)` | Survey completion time |
| `English_First` | 1 = English first language (screening criterion; all 1) |
| `STEM_Field` | Coded field of study/work (1 = STEM, 2 = non-STEM) |
| `Age_Group` | Coded age bracket (2-5, ascending) |

`llm_ratings.csv` is the long-format LLM counterpart with model metadata (`model_label`, `family`, `size`, `reasoning_mode`), per-story IC/RC (`IC`, `RC`), the 11 sub-components under descriptive names, and derived revision fields (`gk_flip`, `flipped`, `flip_direction`, `IC_RC_diff`).

Raw judge files (`study2_llm/data/raw_json/results_*.json`) store, per story, the full three-turn conversation (`turn1`-`turn3`) and the parsed ratings. Some files contain more than 80 entries because interrupted runs were resumed; the analysis scripts deduplicate by story id, keeping the latest entry.

## Reproducing the paper numbers

All scripts resolve data via relative paths anchored on their own location (`Path(__file__)`), so they can be run from anywhere; no API keys are needed for any analysis script. Keys are only needed to re-collect ratings from scratch.

| Script | Reproduces | Output |
|---|---|---|
| `study1_human/analysis/reliability.py` | Krippendorff's alpha, rater disagreement (Sec. 4.1, App. B) | printed stats |
| `study1_human/analysis/phase1d_variance_agreement.py` | human vs LLM disagreement figures (Sec. 4.1 / 5.1) | `analysis/figures/` |
| `study1_human/analysis/phase1b_gatekeeper_flip.py` | human revision behaviour (Sec. 4.2, revision-by-tone table) | printed stats |
| `study1_human/analysis/pattern_regression.py`, `phase1c_subcomponent_weights.py` | sub-component -> RC regression (Sec. 4.3, formula table) | printed stats |
| `study1_human/analysis/mixed_effects_rc.py` | mixed-effects robustness check of the RC regression (stories as random effects) | `analysis/mixed_effects_rc_comparison.csv` |
| `study2_llm/structured_analysis.py` | the three LLM results pillars: distribution collapse, upward revision, sub-component weighting (Sec. 5, Tables F1-F3) | `analysis/structured_analysis.csv` |
| `study2_llm/verify_per_aspect_winners.py` | per-aspect human-vs-LLM winner counts | printed stats |
| `study2_llm/manipulation_check.py` | Spike Prompting manipulation check (does spiking a component raise that component's rating) | `analysis/manipulation_check_*.csv` |
| `study2_llm/matched_subset_check.py` | 12-story matched-subset comparability of Study 1 and Study 2 | `analysis/matched_subset_check.csv` |
| `study2_llm/bootstrap_stability.py` | bootstrap CIs for beta vectors and delta-R2 (slow: resampling loop) | `analysis/bootstrap_stability_*.csv` |
| `study2_llm/temp_sensitivity/analyze_sampling.py` | repeated-sampling / temperature stability (App. F rerun protocols) | `analysis/sampling_stability_summary.csv` |
| `study2_llm/uncertainty_cis.py` | bootstrap 95% CIs for secondary statistics: revision-by-type, human-LLM ranking correlations, sub-component winners, rerun SDs | `analysis/uncertainty_cis_*.csv` |
| `study2_llm/prompt_sensitivity/analyze.py` | prompt-paraphrase sensitivity (App. F) | printed stats |

To re-collect LLM ratings from scratch: `study2_llm/rating_scripts/run_judges.py` targets any OpenAI-compatible endpoint (we used vLLM on a local GPU server) and `run_gemini.py` / `run_openai.py` use the respective official APIs. `shared_prompts.py` contains the exact three-step prompts, identical in wording to the human survey. `temp_sensitivity/gpu_server_commands.sh` documents the full open-model sensitivity sweep.

## Requirements

Python 3.10+, with:

```
pandas numpy scipy scikit-learn statsmodels matplotlib
```

Only needed to re-collect ratings (not to reproduce analyses): `openai`, `google-genai`, `requests`, and a vLLM server for open models.

## Human study ethics

Raters were recruited on Prolific with informed consent and paid above the platform's recommended rate. Only the coded demographics shown above were retained; no participant identifiers were exported. The exact survey is included as `study1_human/survey_instrument.qsf`; full design details are in the paper appendix (Human Survey Design and Ethics).
