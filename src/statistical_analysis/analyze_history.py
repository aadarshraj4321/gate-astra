# FILE NAME: src/statistical_analysis/analyze_history.py
# VERSION 2.0: Now includes sub-topic probability analysis.

import os
import sys
import pandas as pd
from sqlalchemy import create_engine
import json

# --- Path setup ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import get_db_url # We only need the URL function

# ==============================================================================
# CONFIGURATION
# ==============================================================================
OUTPUT_DIR = "analysis_results"
OUTPUT_FILENAME = os.path.join(OUTPUT_DIR, "historical_patterns.json")

# ==============================================================================
# THE ANALYSIS ENGINE
# ==============================================================================

def analyze_historical_data():
    """
    Connects to the database and performs a deep statistical analysis,
    now including sub-topic distributions within each subject.
    """
    print("======================================================")
    print(" GATE-ASTRA: HISTORICAL ANALYSIS ENGINE (V2 - with Sub-Topics)")
    print("======================================================")

    # --- Step 1: Connect and Query ---
    try:
        engine = create_engine(get_db_url())
        print("✅ Connected to the database.")
    except Exception as e:
        print(f"❌ FATAL ERROR: Could not connect to database. Reason: {e}")
        return

    # THE UPGRADED MASTER QUERY: Now selects topic and sub_topic
    query = """
    SELECT 
        q.marks,
        e.exam_year,
        e.organizing_iit,
        s.subject,
        s.topic,
        s.sub_topic
    FROM questions q
    JOIN exams e ON q.exam_id = e.exam_id
    JOIN syllabus s ON q.topic_id = s.topic_id
    """
    print("Executing master query to fetch all historical data...")
    df = pd.read_sql(query, engine)
    print(f"✅ Fetched {len(df)} records for analysis.")

    # --- Step 2: Calculate Overall Subject Probabilities ---
    print("Calculating overall subject probabilities...")
    total_marks = df['marks'].sum()
    subject_marks = df.groupby('subject')['marks'].sum()
    overall_subject_probabilities = (subject_marks / total_marks).to_dict()
    
    # --- Step 3: Calculate IIT-Specific Biases ---
    print("Calculating IIT-specific biases...")
    iit_biases = {}
    all_iits = df['organizing_iit'].unique()
    
    for iit in all_iits:
        iit_df = df[df['organizing_iit'] == iit]
        iit_total_marks = iit_df['marks'].sum()
        if iit_total_marks == 0: continue
        
        iit_subject_marks = iit_df.groupby('subject')['marks'].sum()
        iit_subject_probabilities = (iit_subject_marks / iit_total_marks)
        
        bias = (iit_subject_probabilities - pd.Series(overall_subject_probabilities)).fillna(0)
        iit_biases[iit] = bias.to_dict()

    # --- Step 4: Calculate Mark Distribution ---
    print("Calculating 1-mark vs 2-mark distributions per subject...")
    mark_distribution = df.groupby(['subject', 'marks']).size().unstack(fill_value=0)
    # Ensure both 1 and 2 mark columns exist to avoid errors
    if 1 not in mark_distribution.columns: mark_distribution[1] = 0
    if 2 not in mark_distribution.columns: mark_distribution[2] = 0
    mark_distribution['total'] = mark_distribution[1] + mark_distribution[2]
    mark_distribution['1_mark_prob'] = mark_distribution[1] / mark_distribution['total']
    mark_distribution['2_mark_prob'] = mark_distribution[2] / mark_distribution['total']
    mark_distribution_dict = mark_distribution[['1_mark_prob', '2_mark_prob']].to_dict('index')

    # --- NEW Step 4.5: Calculate Sub-Topic Probabilities within each Subject ---
    print("Calculating sub-topic probabilities within each subject...")
    sub_topic_probabilities = {}
    
    # Create a detailed "full_topic" name for clarity in the output
    df['full_topic'] = df['topic'] + " - " + df['sub_topic'].fillna('General')

    for subject in df['subject'].unique():
        subject_df = df[df['subject'] == subject]
        subject_total_marks = subject_df['marks'].sum()
        if subject_total_marks == 0: continue
        
        sub_topic_marks = subject_df.groupby('full_topic')['marks'].sum()
        sub_topic_probs = (sub_topic_marks / subject_total_marks).to_dict()
        sub_topic_probabilities[subject] = sub_topic_probs

    # --- Step 5: Assemble and Save the Final Patterns ---
    print("Assembling final patterns file...")
    final_patterns = {
        "overall_subject_probabilities": overall_subject_probabilities,
        "iit_biases": iit_biases,
        "mark_distribution_by_subject": mark_distribution_dict,
        "sub_topic_probabilities_by_subject": sub_topic_probabilities, # <-- THE NEW DATA
        "metadata": {
            "total_questions_analyzed": len(df),
            "total_marks_analyzed": int(total_marks),
            "iits_in_dataset": list(all_iits)
        }
    }
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
        json.dump(final_patterns, f, indent=4)
        
    print(f"✅ Historical patterns (V2) successfully analyzed and saved to '{OUTPUT_FILENAME}'")
    print("======================================================")

if __name__ == "__main__":
    analyze_historical_data()