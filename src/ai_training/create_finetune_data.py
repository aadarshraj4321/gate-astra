# FILE NAME: src/ai_training/create_finetune_data.py
# PURPOSE: To create the training 'textbook' for our custom AI model.

import os
import sys
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm

# --- Path setup to import from parent directory (src) ---
# This allows us to import the 'models' file
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import Question, Syllabus, get_db_url

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# The folder where we will save our AI model and its training data.
AI_DATA_DIR = "ai_model_files"
OUTPUT_FILENAME = os.path.join(AI_DATA_DIR, "finetune_dataset.csv")

# ==============================================================================
# THE DATA CREATION ENGINE
# ==============================================================================

def create_training_pairs():
    """
    Connects to the database, queries the questions and their corresponding syllabus topics,
    and formats them into training pairs.
    """
    print("--- Starting Training Data Creation ---")
    
    # --- Step 1: Connect to the database ---
    try:
        engine = create_engine(get_db_url())
        Session = sessionmaker(bind=engine)
        session = Session()
        print("✅ Successfully connected to the database.")
    except Exception as e:
        print(f"❌ FATAL ERROR: Could not connect to the database. Reason: {e}")
        return

    # --- Step 2: Query the data ---
    try:
        print("Querying the database to fetch questions and their linked syllabus topics...")
        # This SQLAlchemy query joins the Question and Syllabus tables using their relationship.
        query = session.query(Question.question_text, Syllabus.subject, Syllabus.topic, Syllabus.sub_topic) \
                       .join(Syllabus, Question.topic_id == Syllabus.topic_id)
        
        # Execute the query and get all results
        results = query.all()
        print(f"✅ Successfully fetched {len(results)} question-topic pairs.")
        
    except Exception as e:
        print(f"❌ FATAL ERROR: Could not execute the database query. Reason: {e}")
        session.close()
        return
    finally:
        session.close()

    # --- Step 3: Format the data ---
    print("Formatting data into training pairs...")
    training_data = []
    for q_text, subject, topic, sub_topic in tqdm(results, desc="Formatting Pairs"):
        # We create a single, descriptive text for the syllabus topic.
        # This gives the AI more context.
        syllabus_full_text = f"{subject}: {topic} - {sub_topic if sub_topic else 'General'}"
        
        # The sentence-transformers library often uses a simple structure like this.
        # We can also use a more complex triplet format later if needed.
        training_data.append({
            "anchor": q_text, # The question is the 'anchor'
            "positive": syllabus_full_text # The correct topic is the 'positive' example
        })
        
    # --- Step 4: Save the data to a CSV file ---
    if not training_data:
        print("❌ No training data was generated. Halting.")
        return

    # Create the output directory if it doesn't exist
    if not os.path.exists(AI_DATA_DIR):
        os.makedirs(AI_DATA_DIR)
        
    # Use Pandas to easily save to a CSV
    df = pd.DataFrame(training_data)
    try:
        df.to_csv(OUTPUT_FILENAME, index=False, encoding='utf-8')
        print(f"\n✅ Training textbook created successfully!")
        print(f"   - Total Pairs: {len(df)}")
        print(f"   - Saved to: '{OUTPUT_FILENAME}'")
    except Exception as e:
        print(f"❌ FATAL ERROR: Could not save the CSV file. Reason: {e}")
        
    print("--- Training Data Creation Complete ---")

if __name__ == "__main__":
    create_training_pairs()