import json
import os
from pathlib import Path

def clean_duplicates(file_path):
    if not Path(file_path).exists():
        return
        
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    results = data.get('results', [])
    if not results:
        return

    # Use a dict to keep only the LATEST entry for each story_id
    unique_results = {}
    for r in results:
        story_id = r.get('story_id')
        # Only keep if it was a successful parse
        if r.get('parse_ok'):
            unique_results[story_id] = r

    # Convert back to list and sort for consistency
    cleaned_list = [unique_results[sid] for sid in sorted(unique_results.keys())]
    
    data['results'] = cleaned_list
    
    # Save back
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    return len(results), len(cleaned_list)

def main():
    raw_json_dir = Path("study2_llm/data/raw_json/")
    files = list(raw_json_dir.glob("*.json"))
    
    print(f"{'File':<50} | {'Before'} | {'After'} | {'Status'}")
    print("-" * 80)
    
    for f in sorted(files):
        # Skip backup files
        if ".bak.json" in f.name: continue
        
        before, after = clean_duplicates(f)
        status = "FIXED" if before != after else "CLEAN"
        print(f"{f.name:<50} | {before:<6} | {after:<5} | {status}")

if __name__ == "__main__":
    main()
