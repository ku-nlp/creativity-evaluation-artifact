# Analysis Plan — Human Study (Study 1)

## Data
- **Human**: `human120data.csv` — 115 ratings, ~10 raters × 12 stories (3 topics × 4 tones)
- **LLM**: `master_data.csv` — 204 rows, 17 conditions × 12 stories
- **17 LLM conditions**: 5 families (Meta, Alibaba, Microsoft, Google, OpenAI), varying size + reasoning mode

## Phase 1: Human vs LLM Comparison

### 1A — Story-level profiles [DONE]
- Scatter: human mean vs LLM mean for IC, RC, enjoyment (r = 0.64, 0.50, 0.44)
- Radar: sub-component profiles by tone (human vs LLM)
- Heatmap: stories × sub-components, side-by-side + difference panel
- Paired bars: IC and RC per story with error bars
- Strip plots: per-topic, individual human dots + individual LLM model markers
- **Key finding**: LLMs rank stories similarly to humans (positive r) but inflate by +0.8-2.2 points, especially on craft dimensions (style, engagement, tone). Variance compressed to 40% of human SD.

### 1B — Gatekeeper Flip comparison [TODO]
- Compute per-human-rater flip (O_Final_Creativity - O_Initial_Creativity)
- Human flip distribution: histogram with mean, % who flip up/down/zero
- LLM flip distribution: same, per model
- Per-story comparison: human flip rate vs LLM flip rate (paired bar or dot plot)
- Correlation: do stories that make humans flip also make LLMs flip?
- **Stats**: chi-square or Fisher's exact on flip rates; Wilcoxon on flip magnitudes
- **Expected finding**: humans flip more often and in both directions; LLMs either always flip (Llama) or never (Qwen/OpenAI)

### 1C — Sub-component weighting [TODO]
- Human: OLS regression `O_Final_Creativity ~ all 11 sub-components` (individual-level, N=115)
- LLM: OLS regression `RC ~ all 11 sub-components` (pooled across models, N=204)
- Also per-model-family regressions for LLM
- Compare standardized betas: which sub-components predict final creativity?
- Visualize: coefficient comparison bar chart (human betas vs LLM betas side by side)
- **Expected finding**: humans weight empathy/emotion more; LLMs weight craft/style more

### 1D — Variance & agreement [TODO]
- Per-story SD: human inter-rater SD vs LLM inter-model SD (already computed in 1A, formalize)
- Krippendorff's alpha: human inter-rater agreement on IC
- LLM inter-model agreement: same metric on IC across 17 conditions
- Coefficient of variation per measure
- **Expected finding**: LLMs agree with each other more than humans agree with each other; LLM "crowd" is less diverse

## Phase 2: LLM-Only Analyses

### 2A — Revisability by model family [TODO]
- Group models by family: Meta (2), Alibaba (4), Microsoft (1), Google (8), OpenAI (2)
- Wilcoxon signed-rank: IC vs RC per model (paired by story)
- Effect size: rank-biserial correlation
- Logistic regression: flipped (0/1) ~ family + tone + IC
- **Expected finding**: family is the strongest predictor of revisability

### 2B — Reasoning mode effect [TODO]
- Paired comparisons (same model, different reasoning):
  - Qwen 8B thinking on vs off (12 paired stories)
  - Qwen 32B thinking on vs off
  - GPT-4.1 vs o4-mini
  - Gemini 2.5 Pro: budget 128 vs default vs 32768
  - Gemini 3.1 Pro: low vs medium vs high
- Wilcoxon on IC, RC, sub-components, flip
- **Expected finding**: reasoning mode changes initial severity but not revisability (confirmed across all vendors)

### 2C — Tone and story effects [TODO]
- ANOVA or Kruskal-Wallis: IC ~ tone (pooled across models)
- Post-hoc pairwise: which tones differ?
- Story difficulty ranking: mean IC across all 17 conditions
- Topic effect: does topic matter after controlling for tone?

### 2D — Sub-component patterns [TODO]
- Correlation matrix of 11 sub-components (pooled)
- PCA or factor analysis: do sub-components cluster into interpretable factors?
- Which sub-components have highest inter-model variance? (discriminating dimensions)
- Ceiling effects: identify sub-components with SD ≈ 0 (topic_fidelity, tone_fidelity)

## Phase 3: Paper Figures

Target 5-6 figures for the paper:

1. **Fig 1**: Human vs LLM scatter (IC correlation) — establishes ranking agreement
2. **Fig 2**: Strip plot (best topic) — shows individual rater/model distributions
3. **Fig 3**: Flip heatmap (models × stories) — the core Gatekeeper Flip finding
4. **Fig 4**: Reasoning mode paired comparison — the "reasoning doesn't help" finding
5. **Fig 5**: Sub-component coefficient comparison (human vs LLM betas)
6. **Fig 6**: Radar profiles by tone — human vs LLM shape comparison

## Phase 4: Supplementary

- Full per-model results tables
- Inter-model agreement matrices
- Raw response examples (qualitative)
- Story texts (appendix)

## Column Mapping Reference

| Human CSV | LLM CSV | Description |
|---|---|---|
| O_Initial_Creativity | IC (initial_creativity) | Holistic creativity before decomposition |
| O_Final_Creativity | RC (reflective_creativity) | Holistic creativity after decomposition |
| O_Enjoyment | enjoyment | Reading enjoyment |
| R_Emotion | emotional_impact | Emotional resonance |
| A_Topic | topic_fidelity | Adherence to topic |
| N_Vocab | vocabulary_freshness | Language novelty |
| N_Plot | plot_uniqueness | Narrative originality |
| N_Surprise | surprise | Unexpected elements |
| R_Empathy | empathy | Character connection |
| R_Thought | thought_provocation | Intellectual stimulation |
| V_Engagement | engagement | Reader absorption |
| V_Style | stylistic_quality | Prose craft |
| V_Logic | logical_coherence | Internal consistency |
| A_Tone | tone_fidelity | Adherence to tone |
