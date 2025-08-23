# FILE NAME: src/student_tools/study_planner.py
# VERSION 5.0: Definitive, feature-complete backend logic.

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
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AI_DATA_DIR = os.path.join(PROJECT_ROOT, 'ai_model_files')
GENERATION_DIR = os.path.join(PROJECT_ROOT, 'generation_results')
FUSION_ENGINE_PATH = os.path.join(AI_DATA_DIR, "fusion_engine_v1.joblib")

PROFICIENCY_SCORES = {"Strong": 0.9, "Medium": 0.5, "Weak": 0.1}
GEMINI_MODEL_NAME = "gemini-1.5-flash"
QUESTIONS_PER_DRILL_TOPIC = 3
NUM_WEAK_TOPICS_TO_TARGET = 5

# ==============================================================================
# STUDY PLANNER CLASS
# ==============================================================================
class StudyPlanner:
    def __init__(self, organizing_iit):
        """Initializes the planner for a specific organizing IIT."""
        self.organizing_iit = organizing_iit
        try:
            self.fusion_model = joblib.load(FUSION_ENGINE_PATH)
        except FileNotFoundError:
            raise FileNotFoundError(f"FATAL: Fusion Engine model not found at {FUSION_ENGINE_PATH}.")
        self.prediction_df = None

    def _generate_prediction(self):
        """Runs the full prediction pipeline to get the final topic heatmap."""
        print(f"Generating final prediction for {self.organizing_iit}...")
        engineer = FeatureEngineerV2()
        features_df = engineer.create_master_feature_set(self.organizing_iit)
        engineer.close()
        
        feature_columns = ['monte_carlo_prob', 'nptel_heat_score', 'prof_bias_score']
        features_df['predicted_weight'] = self.fusion_model.predict(features_df[feature_columns])
        
        features_df['predicted_weight'] = features_df['predicted_weight'].clip(lower=0)
        if features_df['predicted_weight'].sum() > 0:
            features_df['predicted_weight'] /= features_df['predicted_weight'].sum()
        
        subject_weights = features_df.groupby('subject')['predicted_weight'].sum()
        self.prediction_df = subject_weights.reset_index()
        print("✅ Prediction generated and aggregated by subject.")

    def create_plan(self, user_proficiency_map):
        """Generates the personalized study plan."""
        if self.prediction_df is None:
            self._generate_prediction()
            
        plan_df = self.prediction_df.copy() # Use .copy() to avoid SettingWithCopyWarning
        plan_df['proficiency_text'] = plan_df['subject'].map(user_proficiency_map).fillna("Medium")
        plan_df['proficiency_score'] = plan_df['proficiency_text'].map(PROFICIENCY_SCORES)
        plan_df['priority_score'] = plan_df['predicted_weight'] * (1 - plan_df['proficiency_score'])
        
        if plan_df['priority_score'].sum() > 0:
            total_priority = plan_df['priority_score'].sum()
            plan_df['study_time_percent'] = (plan_df['priority_score'] / total_priority) * 100
        else:
            plan_df['study_time_percent'] = 0
            
        return plan_df.sort_values(by='priority_score', ascending=False)

# ==============================================================================
# DRILL GENERATOR CLASS
# ==============================================================================
class DrillGenerator:
    def __init__(self, organizing_iit, prediction_df):
        print(f"\n--- Initializing Drill Generator for {organizing_iit} ---")
        self.organizing_iit = organizing_iit
        self.prediction_df = prediction_df
        
        load_dotenv()
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key: raise ValueError("GEMINI_API_KEY not found.")
        genai.configure(api_key=gemini_api_key)
        safety_settings=[{"category": c, "threshold": "BLOCK_NONE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
        self.gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME, safety_settings=safety_settings)
        print("✅ Gemini model initialized for Drill Generator.")

    def analyze_weaknesses(self, mock_test_csv_path):
        """Reads a mock test result CSV and identifies the top weak topics."""
        print(f"\n[STEP 1/2] Analyzing mock test results from '{mock_test_csv_path}'...")
        results_df = pd.read_csv(mock_test_csv_path)
        incorrect_df = results_df[results_df['result'].str.lower() == 'incorrect']
        
        error_counts = incorrect_df['topic_name'].value_counts()
        total_counts = results_df['topic_name'].value_counts()
        error_rate = (error_counts / total_counts).fillna(0)
        
        weakness_df = pd.DataFrame({'error_rate': error_rate}).reset_index()
        weakness_df.rename(columns={'index': 'subject'}, inplace=True)

        merged_df = pd.merge(weakness_df, self.prediction_df, on='subject', how='inner')
        merged_df['weakness_priority'] = merged_df['error_rate'] * merged_df['predicted_weight']
        top_weaknesses = merged_df.sort_values(by='weakness_priority', ascending=False).head(NUM_WEAK_TOPICS_TO_TARGET)
        print(f"✅ Identified top {len(top_weaknesses)} weaknesses to target.")
        return top_weaknesses['subject'].tolist()

    def generate_drill_set(self, weak_topics):
        """Generates a new set of questions targeting the identified weak topics."""
        print(f"\n[STEP 2/2] Generating drill questions for {len(weak_topics)} topics...")
        # We need the master prompt for generation
        from src.generation.question_generator import MASTER_PROMPT_TEMPLATE
        
        drill_set = []
        for topic in tqdm(weak_topics, desc="Generating Drill Set"):
            prompt = MASTER_PROMPT_TEMPLATE.format(
                iit_name=self.organizing_iit,
                num_questions=QUESTIONS_PER_DRILL_TOPIC,
                topic_name=topic,
                example_questions="N/A" # No examples needed for this targeted drill
            )
            try:
                response = self.gemini_model.generate_content(prompt)
                cleaned_response = response.text.strip().replace("```json", "").replace("```", "").strip()
                generated_qs = json.loads(cleaned_response)
                for q in generated_qs: q['targeted_weakness'] = topic
                drill_set.extend(generated_qs)
                time.sleep(2)
            except Exception as e:
                print(f"\n  -> ⚠️ WARNING: Could not generate questions for '{topic}'. Error: {e}")
        
        return drill_set

# This main block is for direct testing of this script
if __name__ == "__main__":
    print("--- Running study_planner.py in direct test mode ---")
    target_iit = "IIT Kanpur"
    
    my_proficiency = {
        "CS - Algorithms": "Strong", "CS - Compiler Design": "Weak",
        "CS - Databases": "Weak", "GA - Quantitative Aptitude": "Strong"
    }
    planner = StudyPlanner(organizing_iit=target_iit)
    final_plan_df = planner.create_plan(user_proficiency_map=my_proficiency)
    print("\n--- PERSONALIZED STUDY PLAN PREVIEW ---")
    print(final_plan_df[['subject', 'study_time_percent']].head())

    # Example of running the drill generator
    drill_generator = DrillGenerator(organizing_iit=target_iit, prediction_df=planner.prediction_df)
    sample_csv_path = os.path.join(PROJECT_ROOT, "student_inputs/sample_mock_test_results.csv")
    if os.path.exists(sample_csv_path):
        weak_topics = drill_generator.analyze_weaknesses(sample_csv_path)
        if weak_topics:
            drill_questions = drill_generator.generate_drill_set(weak_topics)
            if drill_questions:
                output_path = os.path.join(GENERATION_DIR, f"test_drill_set_{target_iit.replace(' ', '_')}.json")
                if not os.path.exists(GENERATION_DIR): os.makedirs(GENERATION_DIR)
                with open(output_path, 'w') as f: json.dump(drill_questions, f, indent=4)
                print("\n--- PERSONALIZED DRILL SET COMPLETE ---")
                print(f"✅ Generated {len(drill_questions)} questions targeting specific weaknesses.")
                print(f"✅ Saved to '{output_path}'.")
    else:
        print(f"\nCould not find sample CSV for drill generator test at: {sample_csv_path}")