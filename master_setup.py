# FILE NAME: master_setup.py (The One Script to Rule Them All)
# VERSION: 5.0 - Final & Unbreakable
# PURPOSE: Wipes, creates, and populates the entire database from the platinum dataset.
#          It builds the syllabus dynamically from YOUR data, guaranteeing a perfect match.

import json
import os
import sys
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm

# --- Imports ---
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
    from models import Base, Exam, Syllabus, Question, Option, get_db_url
    # We only need EXAM_HISTORY from the other file, not the syllabus data
    from data_ingestion.ingest_exam_metadata import EXAM_HISTORY
except ImportError as e:
    print("FATAL ERROR: Could not import necessary modules.")
    print("Please ensure this script is in the root 'gate-astra' directory and your 'src' folder is structured correctly.")
    print(f"Details: {e}")
    sys.exit(1)

# --- Configuration ---
PLATINUM_DATASET_PATH = "final_dataset/platinum_dataset.json"

def get_question_type(d):
    """Determines question type (MCQ, MSQ, NAT)."""
    if not d.get('options') or len(d['options']) == 0: return "NAT"
    if ';' in str(d.get('answer_key', "")): return "MSQ"
    return "MCQ"

# ==============================================================================
# THE MASTER FUNCTION
# ==============================================================================
def setup_and_ingest_everything():
    print("======================================================")
    print(" GATE-ASTRA: THE ULTIMATE DATABASE SETUP ENGINE")
    print("======================================================")
    
    # --- PHASE 1: Connect to DB and WIPE EVERYTHING ---
    engine = create_engine(get_db_url())
    Session = sessionmaker(bind=engine)
    session = Session()
    print("\n[PHASE 1/4] Connecting to DB and wiping all existing tables...")
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    print("✅ Tables wiped and new schema created successfully.")

    # --- PHASE 2: HARVEST & POPULATE THE ULTIMATE SYLLABUS ---
    print("\n[PHASE 2/4] Harvesting syllabus directly from your platinum dataset...")
    with open(PLATINUM_DATASET_PATH, 'r', encoding='utf-8') as f:
        all_questions_data = json.load(f)
    
    unique_topics_set = set()
    for question in all_questions_data:
        topic_tuple = (
            question.get('subject'),
            question.get('topic'),
            question.get('sub_topic')
        )
        unique_topics_set.add(topic_tuple)
    
    print(f"Discovered {len(unique_topics_set)} unique topics in your dataset. Populating syllabus...")
    syllabus_objects = [
        Syllabus(subject=s, topic=t, sub_topic=st) for s, t, st in unique_topics_set
    ]
    session.bulk_save_objects(syllabus_objects)
    session.commit()
    print("✅ Ultimate syllabus populated successfully.")

    # --- PHASE 3: Populate Exams Table ---
    print("\n[PHASE 3/4] Populating the 'Exams' table...")
    exam_objects = [Exam(**item) for item in EXAM_HISTORY]
    session.bulk_save_objects(exam_objects)
    session.commit()
    print(f"✅ Populated {len(exam_objects)} exam records.")

    # --- PHASE 4: THE GUARANTEED INGESTION ---
    print("\n[PHASE 4/4] Ingesting Platinum Dataset with perfect lookups...")
    exams = session.query(Exam).all()
    exam_lookup = {(e.exam_year, e.paper_subject, e.paper_set): e.exam_id for e in exams}
    
    syllabus_topics = session.query(Syllabus).all()
    syllabus_lookup = {(s.subject, s.topic, s.sub_topic): s.topic_id for s in syllabus_topics}
    
    questions_added, options_added, skipped = 0, 0, 0

    for q_data in tqdm(all_questions_data, desc="Ingesting"):
        exam_key = (q_data['paper_year'], q_data['paper_subject'], q_data.get('paper_set', 1))
        topic_key = (q_data['subject'], q_data['topic'], q_data.get('sub_topic'))
        
        exam_id = exam_lookup.get(exam_key)
        topic_id = syllabus_lookup.get(topic_key)

        if not exam_id or not topic_id:
            # This should now never happen, but it's good practice to keep the check
            skipped += 1
            continue

        new_q = Question(exam_id=exam_id, topic_id=topic_id, question_text=q_data['question_text'], question_type=get_question_type(q_data), marks=q_data['marks'])
        session.add(new_q)
        session.flush()

        if get_question_type(q_data) != "NAT":
            keys = [k.strip() for k in str(q_data.get('answer_key', "")).replace(";", ",").split(',')]
            for opt in q_data.get('options', []):
                label, text = "", ""
                if isinstance(opt, dict):
                    label, text = opt.get('option_label', ""), opt.get('option_text', "")
                elif isinstance(opt, str):
                    parts = opt.split(')', 1)
                    label, text = (parts[0].replace('(', '').strip(), parts[1].strip()) if len(parts) == 2 else ("", opt)
                session.add(Option(question_id=new_q.question_id, option_text=f"{label}) {text}", is_correct=(label in keys)))
                options_added += 1
        questions_added += 1
    
    session.commit()
    session.close()

    print("\n--- FINAL REPORT ---")
    print(f"✅ Questions Added: {questions_added}")
    print(f"✅ Options Added:   {options_added}")
    print(f"🔴 Questions Skipped: {skipped}")
    print("--- INGESTION COMPLETE ---")

if __name__ == "__main__":
    setup_and_ingest_everything()