# FILE NAME: inject_syllabus.py (Corrected and Final)
# PURPOSE: To merge the topics from the platinum dataset into the official syllabus.

import json
import os, sys

# This is needed to import from the src directory
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from data_ingestion.syllabus_data import ALL_SYLLABUS_DATA

PLATINUM_DATASET_PATH = "final_dataset/platinum_dataset.json"
# Corrected file path
NEW_SYLLABUS_FILE_PATH = "src/data_ingestion/syllabus_data_ultimate.py"

def inject_and_create_ultimate_syllabus():
    print("======================================================")
    print(" GATE-ASTRA: SYLLABUS INJECTION ENGINE")
    print("======================================================")
    
    # Step 1: Load the official syllabus from the imported list
    print(f"Loading {len(ALL_SYLLABUS_DATA)} official syllabus topics...")
    # Using a dictionary is better to avoid duplicates and preserve the full item
    ultimate_syllabus_map = {
        # Create a unique tuple key for each syllabus item
        (item.get('subject', '').strip(), item.get('topic', '').strip(), item.get('sub_topic', '').strip() if item.get('sub_topic') is not None else None): item
        for item in ALL_SYLLABUS_DATA
    }

    # Step 2: Load your platinum dataset
    print(f"Loading your platinum dataset from '{PLATINUM_DATASET_PATH}'...")
    with open(PLATINUM_DATASET_PATH, 'r', encoding='utf-8') as f:
        platinum_data = json.load(f)
    print(f"Found {len(platinum_data)} question entries to scan.")

    # Step 3: Find and inject new topics
    new_topics_found = 0
    for question in platinum_data:
        # Create a clean key for the current question's topic
        question_topic_key = (
            question.get('subject', '').strip(),
            question.get('topic', '').strip(),
            question.get('sub_topic', '').strip() if question.get('sub_topic') is not None else None
        )
        
        # If this key is NOT in our map, it's a new one.
        if question_topic_key not in ultimate_syllabus_map:
            new_topics_found += 1
            new_topic_item = {
                "subject": question['subject'],
                "topic": question['topic'],
                "sub_topic": question['sub_topic']
            }
            # Add the new item to our map
            ultimate_syllabus_map[question_topic_key] = new_topic_item
    
    print(f"Discovered and injected {new_topics_found} new, detailed topics from your dataset.")
    
    # Step 4: Write the new, ultimate syllabus file
    final_syllabus_list = list(ultimate_syllabus_map.values())
    print(f"The new ultimate syllabus will have {len(final_syllabus_list)} total topics.")
    
    print(f"Writing the ultimate syllabus to '{NEW_SYLLABUS_FILE_PATH}'...")
    with open(NEW_SYLLABUS_FILE_PATH, 'w', encoding='utf-8') as f:
        f.write("# THIS IS THE ULTIMATE, AUTO-GENERATED SYLLABUS FILE\n")
        f.write("# It merges the official syllabus with the detailed topics from the platinum dataset.\n\n")
        f.write("ALL_SYLLABUS_DATA = [\n")
        # Sort for consistency
        for item in sorted(final_syllabus_list, key=lambda x: str(x)):
            f.write(f"    {json.dumps(item)},\n")
        f.write("]\n")

    print("\n✅✅✅ Ultimate syllabus created successfully! ✅✅✅")
    print("======================================================")

if __name__ == "__main__":
    inject_and_create_ultimate_syllabus()