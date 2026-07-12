Key Findings from Phase 1A

  Scatter plots — LLMs are positively correlated with humans but systematically inflated:
  - IC: r=0.64, LLMs rate +0.76 higher on average
  - RC: r=0.50, LLMs +0.96 higher
  - Enjoyment: r=0.44, LLMs +1.06 higher
  - Almost all points sit above the diagonal — LLMs are more generous

  Best-aligned sub-components (high r, low delta):
  - Surprise: r=0.75, delta=+0.02 — near-perfect alignment
  - Empathy: r=0.77, delta=+0.06 — near-perfect alignment
  - Emotional impact: r=0.79, delta=+0.39

  Worst-aligned (LLM inflates most):
  - Stylistic quality: r=0.41, delta=+2.21 — LLMs rate style ~2 points higher
  - Engagement: r=0.41, delta=+1.44
  - Tone fidelity: r=0.42, delta=+1.33

  Variance: LLM SD is 40-60% of human SD across all measures. LLMs compress the scale.

  Radar plots — The shapes are similar (same peaks/valleys), but the LLM polygon is consistently larger, especially on craft dimensions (style, vocab, tone). Clinical tone shows the best
  shape match — both humans and LLMs see the same weaknesses (empathy, surprise).

  This is a strong story for the paper: LLMs rank stories similarly to humans but are systematically lenient and compress variance, especially on craft/style dimensions. Want to continue
  to Phase 1B (flip comparison)?

  Bar chart (IC & RC)

  - LLMs consistently inflate surreal stories — humans rate ~5.3-5.9, LLMs push to 6.9-7.0 (ceiling)
  - Clinical stories are the best-aligned — ai_shutdown_clinical is nearly identical (human 4.1, LLM 4.1). Both sources agree these are the weakest stories
  - Error bars tell the real story — human SD is 2-3x larger than LLM SD on every story. LLMs compress variance dramatically
  - Melancholic/witty stories — LLMs rate slightly higher but the gap is smaller than surreal

  Heatmap (sub-components)

  - Style, engagement, tone are where LLMs diverge most (difference panel is deep red = LLM rates +1.5 to +2.5 higher)
  - Surprise, empathy are near-zero difference across all stories — LLMs and humans agree on emotional dimensions
  - Clinical stories show the most similar profiles overall — the difference panel is closest to white/blue for those rows

  Flip chart (most interesting for the paper)

  - Humans and LLMs flip in opposite directions on surreal stories — humans revise downward (-0.2 to -0.3), LLMs revise upward (+0.1) or stay flat
  - ai_shutdown_clinical — both flip upward, but humans flip more (+0.50 vs +0.18)
  - konbini_melancholic — biggest disagreement: humans flip down hard (-0.44), LLMs don't flip at all (0.00)
  - thief_melancholic — LLMs flip up strongly (+0.47), humans barely move (-0.10)

  The core narrative: LLMs and humans rank stories similarly but disagree on the direction of revision after analysis. Humans tend to revise down (especially surreal/melancholic), while
  LLMs tend to revise up.

  These strip plots reveal the key patterns per topic:

  What stands out

  Variance gap is stark — Human dots spread across 3-5 points (e.g., thief_clinical: 1 to 7), while LLM markers cluster in a 2-3 point band. LLMs never go below 3, humans regularly hit
  1-2.

  Surreal — LLMs pile up at 7 (ceiling), humans spread 3-7. The LLM mean (dashed) sits above the human mean (solid) by ~1 point on all three topics.

  Clinical — Best alignment. Both means are close (especially ai_shutdown_clinical). But look at the spread — humans range 1-7, LLMs range 3-6. A few Google models (blue squares) dip to
  3, matching the harshest humans.

  Model family clustering — You can see the Alibaba triangles (Qwen) always cluster at 6-7 together. Google squares (Gemini) spread the most across scores. Meta diamonds (Llama) tend to
  sit lower.

  IC → RC shift — Compare left vs right panels. For LLMs on melancholic stories, you can see some markers move up from IC to RC (the flip). Human dots shift more unpredictably — some up,
  some down.

  Konbini melancholic — Most dramatic disagreement. Humans range 3-7 with mean ~5.1, then drop to 4.7 on RC. LLMs cluster at 5-7 and don't drop at all.