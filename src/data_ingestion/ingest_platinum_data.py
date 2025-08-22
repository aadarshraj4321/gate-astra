# FILE NAME: src/data_ingestion/ingest_platinum_data.py
# VERSION: 3.0 (Final, Schema-Aware)

import json
import os
import sys
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm

# --- Path setup ---
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models import Base, Exam, Syllabus, Question, Option, get_db_url

# --- Configuration ---
DATASET_DIR = "final_dataset"
DATASET_FILENAME = "platinum_dataset.json"

def load_lookup_tables(session):
    """
    Loads Exams and Syllabus tables into memory for fast lookups.
    VERSION 2.0: Now uses a more robust key for exams.
    """
    print("Loading lookup tables (Exams, Syllabus) into memory...")
    
    # NEW, MORE ROBUST LOOKUP for exams: (year, subject, set) -> exam_id
    exams = session.query(Exam).all()
    exam_lookup = {
        (e.exam_year, e.paper_subject, e.paper_set): e.exam_id for e in exams
    }
    
    # Syllabus lookup remains the same
    syllabus_topics = session.query(Syllabus).all()
    syllabus_lookup = {
        (s.subject, s.topic, s.sub_topic): s.topic_id for s in syllabus_topics
    }
    
    print(f"Loaded {len(exam_lookup)} exam entries and {len(syllabus_lookup)} syllabus entries.")
    return exam_lookup, syllabus_lookup

def get_question_type(question_data):
    """Determines the question type (MCQ, MSQ, NAT) based on the data."""
    options = question_data.get('options')
    if not options or len(options) == 0:
        return "NAT"
    answer_key = str(question_data.get('answer_key', ""))
    if ';' in answer_key or ',' in answer_key:
        return "MSQ"
    return "MCQ"

def main():
    """
    Main function to orchestrate the ingestion of the platinum dataset.
    VERSION 3.0: Works with the upgraded database schema.
    """
    print("======================================================")
    print(" GATE-ASTRA: THE GREAT INGESTION (DAY 9 - FINAL RUN)")
    print("======================================================")

    engine = create_engine(get_db_url())
    Session = sessionmaker(bind=engine)
    session = Session()

    exam_lookup, syllabus_lookup = load_lookup_tables(session)
    
    dataset_path = os.path.join(DATASET_DIR, DATASET_FILENAME)
    if not os.path.exists(dataset_path):
        print(f"FATAL ERROR: Dataset file not found at '{dataset_path}'")
        return
        
    print(f"Loading platinum dataset from '{dataset_path}'...")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        all_questions_data = json.load(f)
    
    print(f"Found {len(all_questions_data)} questions in the JSON file to process.")

    questions_added_count = 0
    options_added_count = 0
    failed_exam_lookups = set()
    failed_topic_lookups = set()
    
    for question_data in tqdm(all_questions_data, desc="Ingesting All Papers"):
        try:
            exam_key = (
                question_data['paper_year'], 
                question_data['paper_subject'], 
                question_data.get('paper_set', 1) # Default to set 1 if not specified
            )
            exam_id = exam_lookup.get(exam_key)
            
            sub_topic_val = question_data.get('sub_topic')
            topic_key = (question_data['subject'], question_data['topic'], sub_topic_val)
            topic_id = syllabus_lookup.get(topic_key)

            if not exam_id:
                if exam_key not in failed_exam_lookups:
                    failed_exam_lookups.add(exam_key)
                continue

            if not topic_id:
                if topic_key not in failed_topic_lookups:
                    failed_topic_lookups.add(topic_key)
                continue

            q_type = get_question_type(question_data)
            new_question = Question(exam_id=exam_id, topic_id=topic_id, question_text=question_data['question_text'], question_type=q_type, marks=question_data['marks'])
            session.add(new_question)
            session.flush()
            
            if q_type in ["MCQ", "MSQ"]:
                raw_options = question_data.get('options', [])
                correct_keys_str = str(question_data.get('answer_key', ""))
                correct_keys = [key.strip() for key in correct_keys_str.replace(";", ",").split(',')]
                for option in raw_options:
                    option_text, option_label = "", ""
                    if isinstance(option, dict):
                        option_label, option_text = option.get('option_label', ""), option.get('option_text', "")
                    elif isinstance(option, str):
                        parts = option.split(')', 1)
                        if len(parts) == 2:
                            option_label, option_text = parts[0].replace('(', '').strip(), parts[1].strip()
                        else:
                            option_text = option
                    is_correct = (option_label in correct_keys)
                    new_option = Option(question_id=new_question.question_id, option_text=f"{option_label}) {option_text}", is_correct=is_correct)
                    session.add(new_option)
                    options_added_count += 1
            questions_added_count += 1
        except Exception as e:
            print(f"\nERROR processing a question. Data: {question_data}. Reason: {e}")
            session.rollback()

    print("\n--- FINAL DEBUG SUMMARY ---")
    if failed_exam_lookups:
        print(f"🔴 {len(failed_exam_lookups)} unique EXAM lookups failed:")
        for key in sorted(list(failed_exam_lookups)): print(f"  - (Year: {key[0]}, Subject: '{key[1]}', Set: {key[2]})")
    if failed_topic_lookups:
        print(f"🔴 {len(failed_topic_lookups)} unique TOPIC lookups failed.")
    if not failed_exam_lookups and not failed_topic_lookups:
        print("✅✅✅ ALL LOOKUPS WERE SUCCESSFUL! ✅✅✅")
    
    if questions_added_count > 0:
        print(f"\nAttempting to commit {questions_added_count} questions...")
        session.commit()
        print("✅ Commit successful!")
    else:
        print("\nNo new questions were added. Check the debug summary.")
    
    session.close()

    print("\n======================================================")
    print(" THE GREAT INGESTION IS COMPLETE (CS + DA).")
    print(f"   - Total Questions Added this run: {questions_added_count}")
    print(f"   - Total Options Added this run:   {options_added_count}")
    print("======================================================")

if __name__ == "__main__":
    main()