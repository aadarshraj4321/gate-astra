import json

PLATINUM_DATASET_PATH = "final_dataset/platinum_dataset.json"
NEW_SYLLABUS_FILE_PATH = "src/data_ingestion/new_syllabus_data.py"

def harvest():
    print("Reading platinum dataset...")
    with open(PLATINUM_DATASET_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    unique_topics = set()
    for question in data:
        topic_tuple = (
            question.get('subject'),
            question.get('topic'),
            question.get('sub_topic')
        )
        unique_topics.add(topic_tuple)

    print(f"Found {len(unique_topics)} unique topic combinations.")

    # Convert set of tuples to a list of dictionaries for our syllabus format
    syllabus_list = []
    for subject, topic, sub_topic in sorted(list(unique_topics), key=str):
        syllabus_list.append({
            "subject": subject,
            "topic": topic,
            "sub_topic": sub_topic
        })
        
    print("Writing new syllabus file...")
    with open(NEW_SYLLABUS_FILE_PATH, 'w', encoding='utf-8') as f:
        f.write("# This file was auto-generated from your platinum dataset.\n")
        f.write("# This is the new, definitive source of truth for the syllabus.\n\n")
        f.write("ALL_SYLLABUS_DATA = [\n")
        for item in syllabus_list:
            f.write(f"    {json.dumps(item)},\n")
        f.write("]\n")
        
    print(f"New syllabus created at: {NEW_SYLLABUS_FILE_PATH}")

if __name__ == "__main__":
    harvest()