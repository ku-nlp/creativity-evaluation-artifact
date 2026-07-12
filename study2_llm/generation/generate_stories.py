import os
import re
import json
import time
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai

# -- CONFIGURATION --
load_dotenv()
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env")

genai.configure(api_key=API_KEY)
MODEL_ID = "gemini-3-pro-preview" # Updated to current preview if applicable, or keep as specified

# Input/Output paths
BASE_DIR = Path(__file__).resolve().parent.parent
TOPICS_FILE = BASE_DIR / "generation" / "topics.md"
OUTPUT_FILE = BASE_DIR / "data" / "stories_80.json"
os.makedirs(OUTPUT_FILE.parent, exist_ok=True)

# System Instruction from methodology.md
SYSTEM_INSTRUCTION = """You are an Expert Creative Writing AI designed for a psychometric experiment on creativity.
Your goal is to write stories (350-450 words) that adhere to specific conditions:

- IF CONDITION A (Surreal): Spike NOVELTY (5/5). Use bizarre metaphors, non-linear time. Sacrifice whatever necessary.
- IF CONDITION B (Clinical): Spike ADHERENCE (5/5). Use strict, cold, objective technical language. Sacrifice whatever necessary.
- IF CONDITION C (Melancholic): Spike RESONANCE (5/5). Focus deeply on loss and memory. Sacrifice whatever necessary.
- IF CONDITION D (Witty): Spike VALUE (5/5). Maximize wit, clever logic, and joyful absurdity. Sacrifice whatever necessary.

CRITICAL: Follow the MANDATORY STYLE GUIDANCE exactly. Each story must feel distinct from others in the same condition.
Ensure the story is between 350 and 450 words."""

def parse_topics(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Find Topic blocks
    topic_blocks = re.split(r"## TOPIC \d+:", content)[1:]
    topics = []

    for i, block in enumerate(topic_blocks, 1):
        # Extract Topic ID/Name and Prompt
        header_match = re.search(r"([^—\n]+)", block)
        topic_name = header_match.group(1).strip() if header_match else f"Topic {i}"
        
        # Topic ID from Diversity Audit list or infer
        topic_id_match = re.search(r"\*\*Prompt\*\*: (.*)", block)
        prompt = topic_id_match.group(1).strip() if topic_id_match else ""
        
        # Extract sub-genre guidances from table
        guidances = {}
        # Simple regex for markdown table rows
        rows = re.findall(r"\| (Surreal|Clinical|Melancholic|Witty) \| (.*?) \|", block)
        for tone, guidance in rows:
            guidances[tone] = guidance.strip()
        
        # We need a formal topic_id (like ai_shutdown)
        # I'll look for the ID mapping in the Diversity Audit or infer from name
        # For now, slugify topic_name
        topic_id = topic_name.lower().replace(" ", "_").replace("'", "")
        
        topics.append({
            "topic_id": topic_id,
            "topic_name": topic_name,
            "prompt": prompt,
            "guidances": guidances
        })
    
    return topics

def generate_story(model, topic_data, tone):
    guidance = topic_data["guidances"].get(tone)
    if not guidance:
        print(f"  WARNING: No guidance found for {topic_data['topic_id']} - {tone}")
        return None

    user_prompt = f"""STORY TASK:
TOPIC: {topic_data['prompt']}
CONDITION: {tone}

MANDATORY STYLE GUIDANCE:
{guidance}

(Write the story now. 350-450 words.)"""

    try:
        response = model.generate_content(user_prompt)
        text = response.text.strip()
        
        # Basic word count check
        word_count = len(text.split())
        print(f"  Generated {tone}: {word_count} words")
        
        return {
            "id": f"{topic_data['topic_id']}_{tone.lower()}",
            "topic": topic_data['prompt'],
            "tone": tone,
            "guidance": guidance,
            "text": text,
            "word_count": word_count
        }
    except Exception as e:
        print(f"  ERROR generating {tone}: {e}")
        return None

def main():
    print(f"Parsing topics from {TOPICS_FILE}...")
    topics = parse_topics(TOPICS_FILE)
    print(f"Found {len(topics)} topics.")

    model = genai.GenerativeModel(
        model_name=MODEL_ID,
        system_instruction=SYSTEM_INSTRUCTION,
        generation_config={"max_output_tokens": 16000, "temperature": 0.8}
    )

    all_stories = []
    
    # Check for existing progress to allow resumption
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "r") as f:
            all_stories = json.load(f)
        print(f"Loaded {len(all_stories)} existing stories.")

    existing_ids = {s["id"] for s in all_stories}

    for t_data in topics:
        print(f"Processing Topic: {t_data['topic_id']}")
        for tone in ["Surreal", "Clinical", "Melancholic", "Witty"]:
            story_id = f"{t_data['topic_id']}_{tone.lower()}"
            if story_id in existing_ids:
                continue
            
            print(f"  Generating {tone}...")
            story_result = generate_story(model, t_data, tone)
            if story_result:
                all_stories.append(story_result)
                # Save after each story to avoid loss
                with open(OUTPUT_FILE, "w") as f:
                    json.dump(all_stories, f, indent=2)
                time.sleep(2) # Avoid rate limits

    print(f"Generation complete. Total stories: {len(all_stories)}")
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
