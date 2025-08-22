# FILE NAME: find_mismatches.py
# PURPOSE: To find the final 130 inconsistencies and generate a clear report.

import json
import os
import sys
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm
from difflib import get_close_matches

# --- Imports ---
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
    from models import Base, Exam, Syllabus, Question, Option, get_db_url
except ImportError as e:
    print(f"FATAL ERROR: Could not import modules: {e}")
    sys.exit(1)

# --- Configuration ---
PLATINUM_DATASET_PATH = "final_dataset/platinum_dataset.json"
FAILURE_REPORT_PATH = "final_mismatch_report.txt"

def find_final_mismatches():
    print("======================================================")
    print(" GATE-ASTRA: FINAL MISMATCH DIAGNOSTIC TOOL")
    print("======================================================")
    
    engine = create_engine(get_db_url())
    Session = sessionmaker(bind=engine)
    session = Session()
    
    print("\n[PHASE 1/3] Loading data from the database...")
    # Load lookups from the database that master_setup just created
    exams = session.query(Exam).all()
    exam_lookup = {(e.exam_year, e.paper_subject, e.paper_set): e.exam_id for e in exams}
    
    syllabus_topics = session.query(Syllabus).all()
    syllabus_lookup = {(s.subject, s.topic, s.sub_topic): s.topic_id for s in syllabus_topics}
    # For fuzzy matching
    syllabus_key_strings = [f"{s.subject} | {s.topic} | {s.sub_topic}" for s in syllabus_topics]
    
    print("\n[PHASE 2/3] Loading your platinum dataset...")
    with open(PLATINUM_DATASET_PATH, 'r', encoding='utf-8') as f:
        all_questions_data = json.load(f)
    print(f"Loaded {len(all_questions_data)} questions to check.")

    print("\n[PHASE 3/3] Analyzing for mismatches...")
    failure_report = []

    for question_data in tqdm(all_questions_data, desc="Analyzing"):
        exam_key = (question_data['paper_year'], question_data['paper_subject'], question_data.get('paper_set', 1))
        # Use .strip() and safety checks to handle potential whitespace issues
        topic_key = (
            question_data.get('subject', '').strip(),
            question_data.get('topic', '').strip(),
            question_data.get('sub_topic', '').strip() if question_data.get('sub_topic') is not None else None
        )
        
        exam_id = exam_lookup.get(exam_key)
        topic_id = syllabus_lookup.get(topic_key)

        # If the lookup fails for either, it's a mismatch we need to report
        if not exam_id or not topic_id:
            original_key_str = f"{topic_key[0]} | {topic_key[1]} | {topic_key[2]}"
            best_guess = get_close_matches(original_key_str, syllabus_key_strings, n=1, cutoff=0.8)
            
            report_entry = {
                "question": f"Year: {exam_key[0]}, Subject: {exam_key[1]}, Num: {question_data.get('question_number')}",
                "failed_key_from_json": topic_key,
                "best_guess_from_db": best_guess[0] if best_guess else "No close match found."
            }
            failure_report.append(report_entry)

    session.close()

    # --- Final Report Generation ---
    print("\n================= DIAGNOSTIC REPORT =================")
    if failure_report:
        print(f"🔴 Found {len(failure_report)} inconsistencies to fix.")
        print(f"   A detailed report has been generated: '{FAILURE_REPORT_PATH}'")
        with open(FAILURE_REPORT_PATH, 'w', encoding='utf-8') as f:
            f.write("ACTION: Open your platinum_dataset.json and for each entry below, update the subject/topic/sub_topic to exactly match the BEST GUESS.\n\n")
            for entry in failure_report:
                f.write(f"Question: {entry['question']}\n")
                f.write(f"  - YOUR VERSION (from JSON): {entry['failed_key_from_json']}\n")
                f.write(f"  - BEST GUESS (from DB):   {entry['best_guess_from_db']}\n\n")
    else:
        print("✅✅✅ No mismatches found! Your dataset is perfectly aligned with the database. ✅✅✅")
        
    print("======================================================")

if __name__ == "__main__":
    find_final_mismatches()