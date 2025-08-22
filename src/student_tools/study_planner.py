# FILE NAME: src/student_tools/study_planner.py
# VERSION 2.0: Now includes the DrillGenerator class.

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
import json


# --- Path setup ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from statistical_analysis.create_features import FeatureEngineerV2
# We now need the Question model to retrieve examples for the generator
from models import get_db_url, Question

# ==============================================================================
# CONFIGURATION
# ==============================================================================
AI_DATA_DIR = "ai_model_files"
GENERATION_DIR = "generation_results"
FUSION_ENGINE_PATH = os.path.join(AI_DATA_DIR, "fusion_engine_v1.joblib")

PROFICIENCY_SCORES = {"Strong": 0.9, "Medium": 0.5, "Weak": 0.1}
GEMINI_MODEL_NAME = "gemini-1.5-flash"
QUESTIONS_PER_DRILL_TOPIC = 3 # Generate 3 questions for each weak topic
NUM_WEAK_TOPICS_TO_TARGET = 5 # Create a drill set focusing on the top 5 weaknesses

# ==============================================================================
# STUDY PLANNER CLASS (from Day 23)
# ==============================================================================
class StudyPlanner:
    # ... (The StudyPlanner class from yesterday remains exactly the same) ...
    def __init__(self, organizing_iit):
        self.organizing_iit = organizing_iit
        self.fusion_model = joblib.load(FUSION_ENGINE_PATH)
        self.prediction_df = None
    def _generate_prediction(self):
        print(f"Generating final prediction for {self.organizing_iit}...")
        engineer = FeatureEngineerV2()
        features_df = engineer.create_master_feature_set(self.organizing_iit)
        engineer.close()
        feature_columns = ['monte_carlo_prob', 'nptel_heat_score', 'prof_bias_score']
        features_df['predicted_weight'] = self.fusion_model.predict(features_df[feature_columns])
        features_df['predicted_weight'] = features_df['predicted_weight'].clip(lower=0)
        features_df['predicted_weight'] /= features_df['predicted_weight'].sum()
        subject_weights = features_df.groupby('subject')['predicted_weight'].sum()
        self.prediction_df = subject_weights.reset_index()
        print("✅ Prediction generated and aggregated by subject.")
    def create_plan(self, user_proficiency_map):
        if self.prediction_df is None: self._generate_prediction()
        print("\n--- Calculating Priority Scores ---")
        self.prediction_df['proficiency_text'] = self.prediction_df['subject'].map(user_proficiency_map).fillna("Medium")
        self.prediction_df['proficiency_score'] = self.prediction_df['proficiency_text'].map(PROFICIENCY_SCORES)
        self.prediction_df['priority_score'] = self.prediction_df['predicted_weight'] * (1 - self.prediction_df['proficiency_score'])
        total_priority = self.prediction_df['priority_score'].sum()
        self.prediction_df['study_time_percent'] = (self.prediction_df['priority_score'] / total_priority) * 100
        return self.prediction_df.sort_values(by='priority_score', ascending=False)
    def display_plan(self, plan_df):
        print("\n\n======================================================"); print("    GATE-ASTRA PERSONALIZED STUDY PLAN"); print("="*54)
        print(f"Tailored for: {self.organizing_iit}\n")
        high_p = plan_df['study_time_percent'].quantile(0.75)
        med_p = plan_df['study_time_percent'].quantile(0.40)
        def get_tier(row):
            reason = f"(High exam probability, Your self-assessed weakness: {row['proficiency_text']})"
            if row['study_time_percent'] >= high_p: return "🎯 TOP PRIORITY", reason
            elif row['study_time_percent'] >= med_p: return "✍️ CORE FOCUS", reason
            else: return "✅ SECURE MARKS", f"(Relatively lower priority, or Your strength: {row['proficiency_text']})"
        plan_df[['Tier', 'Reason']] = plan_df.apply(get_tier, axis=1, result_type='expand')
        print(f"{'Priority Tier':<20} | {'Subject':<50} | {'Recommended Study Time':<25} | {'Reason'}")
        print("-" * 130)
        for _, row in plan_df.iterrows():
            if row['study_time_percent'] > 0.5:
                print(f"{row['Tier']:<20} | {row['subject']:<50} | {row['study_time_percent']:>22.2f}% | {row['Reason']}")
        print("="*130)

# ==============================================================================
# NEW DRILL GENERATOR CLASS (for Day 24)
# ==============================================================================
class DrillGenerator:
    def __init__(self, organizing_iit, prediction_df):
        print(f"\n--- Initializing Drill Generator for {organizing_iit} ---")
        self.organizing_iit = organizing_iit
        # Reuse the prediction from the planner to save time
        self.prediction_df = prediction_df
        
        # Setup Gemini
        load_dotenv()
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key: raise ValueError("GEMINI_API_KEY not found.")
        genai.configure(api_key=gemini_api_key)
        safety_settings=[{"category": c, "threshold": "BLOCK_NONE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
        self.gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME, safety_settings=safety_settings)
        print("✅ Gemini model initialized.")

    def analyze_weaknesses(self, mock_test_csv_path):
        """Reads a mock test result CSV and identifies the top weak topics."""
        print(f"\n[STEP 1/2] Analyzing mock test results from '{mock_test_csv_path}'...")
        results_df = pd.read_csv(mock_test_csv_path)
        incorrect_df = results_df[results_df['result'].str.lower() == 'incorrect']
        
        # Calculate error rate for each topic
        error_counts = incorrect_df['topic_name'].value_counts()
        total_counts = results_df['topic_name'].value_counts()
        error_rate = (error_counts / total_counts).fillna(0)
        
        weakness_df = pd.DataFrame({'error_rate': error_rate}).reset_index()
        weakness_df.rename(columns={'index': 'subject'}, inplace=True)

        # Merge with our prediction to find high-priority weaknesses
        # We need to map topic names to subject names for the merge
        # This is a simplification; a more robust way would use topic_ids
        merged_df = pd.merge(weakness_df, self.prediction_df, on='subject', how='inner')
        
        # Prioritize topics that have a high error rate AND are predicted to be important
        merged_df['weakness_priority'] = merged_df['error_rate'] * merged_df['predicted_weight']
        
        top_weaknesses = merged_df.sort_values(by='weakness_priority', ascending=False).head(NUM_WEAK_TOPICS_TO_TARGET)
        print(f"✅ Identified top {len(top_weaknesses)} weaknesses to target.")
        return top_weaknesses['subject'].tolist()

    def generate_drill_set(self, weak_topics):
        """Generates a new set of questions targeting the identified weak topics."""
        print(f"\n[STEP 2/2] Generating drill questions for {len(weak_topics)} topics...")
        # We will reuse the Master Prompt from Day 22
        from src.generation.question_generator import MASTER_PROMPT_TEMPLATE
        
        drill_set = []
        for topic in tqdm(weak_topics, desc="Generating Drill Set"):
            # For a real RAG, we would fetch examples. For this drill, we can let the LLM generate from the topic name.
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

def main():
    """Example usage of both the Planner and the Drill Generator."""
    target_iit = "IIT Kanpur"
    
    # --- Run Day 23 Logic: Study Planner ---
    my_proficiency = {
        "CS - Algorithms": "Strong", "CS - Compiler Design": "Weak",
        "CS - Databases": "Weak", "GA - Quantitative Aptitude": "Strong"
    }
    planner = StudyPlanner(organizing_iit=target_iit)
    final_plan_df = planner.create_plan(user_proficiency_map=my_proficiency)
    planner.display_plan(final_plan_df)

    # --- Run Day 24 Logic: Drill Generator ---
    drill_generator = DrillGenerator(organizing_iit=target_iit, prediction_df=planner.prediction_df)
    
    # Analyze the sample mock test results
    weak_topics = drill_generator.analyze_weaknesses("student_inputs/sample_mock_test_results.csv")
    
    if weak_topics:
        # Generate the personalized drill set
        drill_questions = drill_generator.generate_drill_set(weak_topics)
        if drill_questions:
            # Save the drill set
            output_path = f"generation_results/drill_set_{target_iit.replace(' ', '_')}.json"
            if not os.path.exists("generation_results"): os.makedirs("generation_results")
            with open(output_path, 'w') as f: json.dump(drill_questions, f, indent=4)
            print("\n--- PERSONALIZED DRILL SET COMPLETE ---")
            print(f"✅ Generated {len(drill_questions)} questions targeting your specific weaknesses.")
            print(f"✅ Saved to '{output_path}'.")

if __name__ == "__main__":
    main()