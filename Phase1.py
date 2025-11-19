import json
from collections import defaultdict

SPEAKER_CHUNKS_PATH = "julius_caesar_speaker_chunks.jsonl"
CLEAN_SCENE_CHUNKS_PATH = "julius_caesar_scene_chunks_CLEAN.jsonl"

print("Starting: Re-stitching speaker chunks into clean scene chunks...")

try:
    scene_data = defaultdict(list)

    with open(SPEAKER_CHUNKS_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            chunk = json.loads(line)
            scene_key = (chunk['act'], chunk['scene'])

            if chunk['speaker'] == "NARRATOR":
                scene_data[scene_key].append(chunk['text'])
            else:
                scene_data[scene_key].append(f"{chunk['speaker']}:\n{chunk['text']}")

    print(f"Read and grouped {len(scene_data)} scenes from speaker chunks.")

    with open(CLEAN_SCENE_CHUNKS_PATH, 'w', encoding='utf-8') as f:
        for (act, scene), text_list in scene_data.items():
            # Join all dialogue from that scene with newlines
            full_scene_text = "\n\n".join(text_list)

            new_chunk = {
                "act": act,
                "scene": scene,
                "text": full_scene_text
            }
            f.write(json.dumps(new_chunk) + '\n')

    print(f"SUCCESS: Created clean scene file at '{CLEAN_SCENE_CHUNKS_PATH}'")

    # Print a sample from the new file
    with open(CLEAN_SCENE_CHUNKS_PATH, 'r', encoding='utf-8') as f:
        first_scene = json.loads(f.readline())
        print(json.dumps(first_scene, indent=2))


except FileNotFoundError:
    print(f"ERROR: Could not find {SPEAKER_CHUNKS_PATH}")
except Exception as e:
    print(f"An error occurred: {e}")