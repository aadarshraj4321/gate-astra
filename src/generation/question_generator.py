# FILE NAME: src/generation/question_generator.py
# VERSION 3.1: Corrected Gemini model name for API stability.

import os
import sys
import pandas as pd
import joblib
import json
import time
import google.generativeai as genai
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm

# --- Path setup ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from statistical_analysis.create_features import FeatureEngineerV2
from models import get_db_url, Question

# ==============================================================================
# CONFIGURATION
# ==============================================================================
AI_DATA_DIR = "ai_model_files"
GENERATION_DIR = "generation_results"
FUSION_ENGINE_PATH = os.path.join(AI_DATA_DIR, "fusion_engine_v1.joblib")

NUM_HOT_TOPICS = 30
EXAMPLES_PER_TOPIC = 3
QUESTIONS_TO_GENERATE_PER_TOPIC = 5

# <<< --- THE FIX IS HERE --- >>>
# Use the stable, versioned model name.
GEMINI_MODEL = "gemini-1.5-flash"
# <<< --- END OF FIX --- >>>


# ==============================================================================
# THE MASTER PROMPT (Optimized for Gemini)
# ==============================================================================
MASTER_PROMPT_TEMPLATE = """
You are an expert GATE Exam Question Setter from {iit_name}, specializing in Computer Science. You are known for creating challenging, concept-linking questions that are fair but rigorous.

Your task is to create {num_questions} new, original, high-quality mock questions based on the provided topic.

**TOPIC CONTEXT:**
The questions MUST be strictly about this topic: "{topic_name}"

**STYLE GUIDE / EXAMPLES:**
To ensure your questions match the authentic GATE style, here are some real past questions on this topic. Learn from their structure, complexity, and language. Do not copy them.
---
{example_questions}
---

**DIFFICULTY:**
The questions must be "Hard" difficulty, suitable for a 2-mark question. They should require deep conceptual understanding, not just simple recall.

**OUTPUT FORMAT:**
Your response MUST be ONLY a single, raw JSON array containing {num_questions} question objects. It is critical that your output starts with `[` and ends with `]` and contains nothing else. Do not use markdown like ```json.

**JSON Object Structure for each question:**
{{
  "question_text": "The full, unique text of the question you created.",
  "options": [
    {{ "option_label": "A", "option_text": "Text for plausible but incorrect option A." }},
    {{ "option_label": "B", "option_text": "Text for the correct option B." }},
    {{ "option_label": "C", "option_text": "Text for plausible but incorrect option C." }},
    {{ "option_label": "D", "option_text": "Text for plausible but incorrect option D." }}
  ],
  "answer_key": "B",
  "explanation": "A clear, concise explanation of why the correct answer is right and why the others are plausible distractors."
}}
"""

# ==============================================================================
# THE GENERATION PIPELINE CLASS
# ==============================================================================
class MockQuestionPipeline:
    def __init__(self, organizing_iit):
        print(f"--- Initializing Gemini-Powered Pipeline for {organizing_iit} ---")
        self.organizing_iit = organizing_iit
        self.fusion_model = joblib.load(FUSION_ENGINE_PATH)
        self.sql_engine = create_engine(get_db_url())
        self.SqlSession = sessionmaker(bind=self.sql_engine)
        
        load_dotenv()
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file.")
        genai.configure(api_key=gemini_api_key)
        self.gemini_model = genai.GenerativeModel(GEMINI_MODEL)

    def generate_final_prediction(self):
        print(f"\n[STEP 1/3] Generating final prediction for {self.organizing_iit}...")
        engineer = FeatureEngineerV2()
        features_df = engineer.create_master_feature_set(self.organizing_iit)
        engineer.close()
        feature_columns = ['monte_carlo_prob', 'nptel_heat_score', 'prof_bias_score']
        features_df['predicted_weight'] = self.fusion_model.predict(features_df[feature_columns])
        features_df['predicted_weight'] = features_df['predicted_weight'].clip(lower=0)
        features_df['predicted_weight'] /= features_df['predicted_weight'].sum()
        print("✅ Final prediction generated.")
        return features_df.sort_values(by='predicted_weight', ascending=False)

    def retrieve_example_questions(self, topic_id):
        session = self.SqlSession()
        try:
            questions = session.query(Question.question_text).filter(Question.topic_id == topic_id).limit(EXAMPLES_PER_TOPIC).all()
            return [q[0] for q in questions]
        finally:
            session.close()

    def create_generation_tasks(self):
        final_prediction_df = self.generate_final_prediction()
        hot_topics = final_prediction_df.head(NUM_HOT_TOPICS)
        print(f"\n[STEP 2/3] Retrieving example questions for the top {NUM_HOT_TOPICS} hot topics...")
        generation_tasks = []
        for _, row in tqdm(hot_topics.iterrows(), total=len(hot_topics), desc="Retrieving Examples"):
            examples = self.retrieve_example_questions(row['topic_id'])
            if examples:
                generation_tasks.append({
                    "topic_name": row['topic_name'],
                    "example_questions": "\n---\n".join(examples)
                })
        print(f"\n✅ Pipeline complete. Created {len(generation_tasks)} tasks for Gemini.")
        return generation_tasks

    def generate_mock_questions(self, tasks):
        print(f"\n[STEP 3/3] Generating mock questions from {len(tasks)} tasks with {GEMINI_MODEL}...")
        all_mock_questions = []
        for task in tqdm(tasks, desc="Generating Questions with Gemini"):
            prompt = MASTER_PROMPT_TEMPLATE.format(
                iit_name=self.organizing_iit,
                num_questions=QUESTIONS_TO_GENERATE_PER_TOPIC,
                topic_name=task['topic_name'],
                example_questions=task['example_questions']
            )
            try:
                response = self.gemini_model.generate_content(prompt)
                response_text = response.text
                cleaned_response = response_text.strip().replace("```json", "").replace("```", "").strip()
                generated_qs = json.loads(cleaned_response)
                for q in generated_qs:
                    q['topic_name'] = task['topic_name']
                all_mock_questions.extend(generated_qs)
                time.sleep(2)
            except json.JSONDecodeError:
                print(f"\n  -> ⚠️ WARNING: Gemini returned invalid JSON for topic '{task['topic_name']}'. Skipping.")
                print(f"     Raw response: {response_text[:100]}...")
            except Exception as e:
                print(f"\n  -> ❌ WARNING: An API error occurred for topic '{task['topic_name']}'. Error: {e}")

        return all_mock_questions


def main(organizing_iit="IIT Kanpur"):
    print("======================================================")
    print(" GATE-ASTRA: MOCK QUESTION FACTORY (DAY 22 - GEMINI EDITION)")
    print("======================================================")
    
    pipeline = MockQuestionPipeline(organizing_iit=organizing_iit)
    tasks = pipeline.create_generation_tasks()
    
    if tasks:
        mock_questions = pipeline.generate_mock_questions(tasks)
        if mock_questions:
            output_filename = f"mock_questions_{organizing_iit.replace(' ', '_')}.json"
            output_path = os.path.join(GENERATION_DIR, output_filename)
            if not os.path.exists(GENERATION_DIR): os.makedirs(GENERATION_DIR)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(mock_questions, f, indent=4)
            print("\n--- MOCK QUESTION GENERATION COMPLETE ---")
            print(f"✅ Generated a total of {len(mock_questions)} mock questions.")
            print(f"✅ Saved to '{output_path}'.")
    
    print("======================================================")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_iit = " ".join(sys.argv[1:])
        main(organizing_iit=target_iit)
    else:
        main()