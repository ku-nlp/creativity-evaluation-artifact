import json
from pathlib import Path

def create_failed_set():
    stories_path = Path("study2_llm/data/stories_80.json")
    with open(stories_path, "r") as f:
        all_stories = json.load(f)
    
    # Combined list of unique failed story IDs from both Think and Nothink modes
    failed_ids = {
        "ai_shutdown_surreal", "ai_shutdown_clinical",
        "the_creature_melancholic", "the_creature_witty",
        "the_door_below_clinical", "the_recipe_clinical",
        "the_elevator_melancholic", "the_bus_stop_surreal",
        "the_bus_stop_witty"
    }
    
    failed_stories = [s for s in all_stories if s['id'] in failed_ids]
    
    with open("study2_llm/data/failed_stories.json", "w") as f:
        json.dump(failed_stories, f, indent=2)
    
    print(f"Created failed_stories.json with {len(failed_stories)} stories.")

if __name__ == "__main__":
    create_failed_set()
