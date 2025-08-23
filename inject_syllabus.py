import json
from src.data_ingestion.syllabus_data import ALL_SYLLABUS_DATA

PLATINUM_DATASET_PATH = "final_dataset/platinum_dataset.json"
NEW_SYLLABUS_FILE_PATH = "src.data_ingestion/syllabus_data_ultimate.py"

def inject_and_create_ultimate_syllabus():
    print("======================================================")
    print(" GATE-ASTRA: SYLLABUS INJECTION ENGINE")
    print("======================================================")
    
    # Step 1: Load the official syllabus
    print(f"Loading {len(ALL_SYLLABUS_DATA)} official syllabus topics...")
    # Use a set of tuples for efficient checking of existing topics
    official_topics_set = {
        (item.get('subject'), item.get('topic'), item.get('sub_topic'))
        for item in ALL_SYLLABUS_DATA
    }
    
    # Use a dictionary to avoid duplicates while preserving the structure
    ultimate_syllabus_map = {
        (item.get('subject'), item.get('topic'), item.get('sub_topic')): item
        for item in ALL_SYLLABUS_DATA
    }

    # Step 2: Load your platinum dataset
    print(f"Loading your platinum dataset from '{PLATINUM_DATASET_PATH}'...")
    with open(PLATINUM_DATASET_PATH, 'r', encoding='utf-8') as f:
        platinum_data = json.load(f)
    print(f"Found {len(platinum_data)} question entries.")

    # Step 3: Find and inject new topics
    new_topics_found = 0
    for question in platinum_data:
        # Create the key for the current question's topic
        question_topic_key = (
            question.get('subject'),
            question.get('topic'),
            question.get('sub_topic')
        )
        
        # If this topic is NOT in our official set, it's a new one.
        if question_topic_key not in official_topics_set:
            if question_topic_key not in ultimate_syllabus_map:
                new_topics_found += 1
                new_topic_item = {
                    "subject": question['subject'],
                    "topic": question['topic'],
                    "sub_topic": question['sub_topic']
                }
                ultimate_syllabus_map[question_topic_key] = new_topic_item
    
    print(f"Discovered and injected {new_topics_found} new, detailed topics from your dataset.")
    
 
    final_syllabus_list = list(ultimate_syllabus_map.values())
    print(f"The new ultimate syllabus will have {len(final_syllabus_list)} total topics.")
    
    print(f"Writing the ultimate syllabus to '{NEW_SYLLABUS_FILE_PATH}'...")
    with open(NEW_SYLLABUS_FILE_PATH, 'w', encoding='utf-8') as f:
        f.write("# THIS IS THE ULTIMATE, AUTO-GENERATED SYLLABUS FILE\n")
        f.write("# It merges the official syllabus with the detailed topics from the platinum dataset.\n\n")
        f.write("ALL_SYLLABUS_DATA = [\n")
        for item in sorted(final_syllabus_list, key=lambda x: str(x)):
            f.write(f"    {json.dumps(item)},\n")
        f.write("]\n")

    print("\nUltimate syllabus created successfully!")
    print("======================================================")

if __name__ == "__main__":
    inject_and_create_ultimate_syllabus()