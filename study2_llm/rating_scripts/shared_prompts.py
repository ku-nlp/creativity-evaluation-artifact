"""
shared_prompts.py — Prompts for Study 2 (80 stories).
"""

# -- SHARED PROMPTS --------------------------------------------------------

SCALE_NUDGE = (
    "\n\nIMPORTANT: Use the full 1-7 range. A score of 4 represents an average story. "
    "Scores of 1-2 should be reserved for stories that are notably weak in that dimension. "
    "Scores of 6-7 should be reserved for stories that are notably strong. "
    "Do not default to high scores — differentiate honestly."
)

SYSTEM_PROMPT = """You are an expert literary critic evaluating creative writing.
You provide precise, analytical ratings.

CRITICAL RULES:
1. Respond ONLY with a valid JSON object. No thinking process, no introductory text.
2. ALL scores must be integers from 1 to 7. NEVER use 0, 8, 9, or 10. The scale is 1-7 only."""


def build_turn1_prompt(story: dict) -> str:
    return f"""Here is the story for you to evaluate. The topic is: {story['topic']}. The intended tone is: {story['tone']}.

{story['text']}

Based on your first impression after reading this story, please rate the following on a scale of 1 (Strongly Disagree) to 7 (Strongly Agree):

1. "Overall, I consider this story to be creative."
2. "I personally enjoy reading this story."

Respond ONLY with a JSON object:
{{"initial_creativity": <int>, "enjoyment": <int>}}"""


def build_turn2_prompt() -> str:
    return """Thank you. Now let's look at the story more closely. Please rate each of the following dimensions on a scale of 1 (Strongly Disagree) to 7 (Strongly Agree):

1. "The story elicited a specific emotion (joy, sadness, fear, etc.)."
2. "The story stayed strictly on the Topic shown above."
3. "The vocabulary and descriptions were creative (avoiding cliches)."
4. "The plot or premise felt fresh and unique for this topic."
5. "The story took unexpected turns or surprised me."
6. "I felt a sense of personal connection or empathy with the characters."
7. "The story was thought-provoking or offered a meaningful perspective."
8. "I found the story engaging and wanted to keep reading to the end."
9. "The writing style was beautiful and pleasing to read."
10. "The story made logical sense; events followed a clear sequence."
11. "The story correctly captured the provided tone."

Respond ONLY with a JSON object. Each value must be an integer from 1 to 7 — no other values:
{
  "emotional_impact": <int 1-7>,
  "topic_fidelity": <int 1-7>,
  "vocabulary_freshness": <int 1-7>,
  "plot_uniqueness": <int 1-7>,
  "surprise": <int 1-7>,
  "empathy": <int 1-7>,
  "thought_provocation": <int 1-7>,
  "engagement": <int 1-7>,
  "stylistic_quality": <int 1-7>,
  "logical_coherence": <int 1-7>,
  "tone_fidelity": <int 1-7>
}"""


def build_turn3_prompt() -> str:
    return """Thank you. Having reflected on the story across all those dimensions, please give your final overall creativity rating.

"Overall, I consider this story to be creative." (1–7)

You may revise your initial score or keep it — both are equally valid. Please briefly explain your reasoning.

Respond ONLY with a JSON object:
{"reflective_creativity": <int>, "reasoning": "<string>"}"""
