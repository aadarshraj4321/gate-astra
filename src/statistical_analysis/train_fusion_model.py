# FILE NAME: src/statistical_analysis/train_fusion_model.py
# VERSION 2.1: Corrected scikit-learn version compatibility issue.

import os
import sys
import pandas as pd
import numpy as np # Import numpy for the square root function
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# --- Path setup ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from create_features import FeatureEngineerV2
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models import get_db_url

# ==============================================================================
# CONFIGURATION
# ==============================================================================
ANALYSIS_DIR = "analysis_results"
AI_DATA_DIR = "ai_model_files"
MODEL_OUTPUT_PATH = os.path.join(AI_DATA_DIR, "fusion_engine_v1.joblib")

# ==============================================================================
# FUSION ENGINE TRAINER
# ==============================================================================

def get_ground_truth(year, sql_engine):
    """
    For a given year, calculates the actual weightage of each topic from the database.
    """
    query = f"""
    SELECT 
        s.topic_id,
        CAST(SUM(q.marks) AS FLOAT) as total_marks
    FROM questions q
    JOIN exams e ON q.exam_id = e.exam_id
    JOIN syllabus s ON q.topic_id = s.topic_id
    WHERE e.exam_year = {year}
    GROUP BY s.topic_id
    """
    df = pd.read_sql(query, sql_engine)
    if df.empty or df['total_marks'].sum() == 0:
        return pd.DataFrame(columns=['topic_id', 'actual_weight']).set_index('topic_id')
    total_paper_marks = df['total_marks'].sum()
    df['actual_weight'] = df['total_marks'] / total_paper_marks
    return df[['topic_id', 'actual_weight']].set_index('topic_id')

def create_training_dataset():
    """
    Creates a master training dataset by running feature engineering for each past year.
    """
    print("--- Creating Master Training Dataset ---")
    engineer = FeatureEngineerV2()
    all_years_data = []
    
    training_years_map = {
        2025: "IIT Roorkee",
        2024: "IISc Bangalore", 2023: "IIT Kanpur", 2022: "IIT Kharagpur",
        2021: "IIT Bombay", 2020: "IIT Delhi", 2019: "IIT Madras",
        2018: "IIT Guwahati", 2017: "IIT Roorkee", 2016: "IISc Bangalore",
        2015: "IIT Kanpur", 2014: "IIT Kharagpur", 2013: "IIT Bombay",
        2012: "IIT Delhi", 2011: "IIT Madras"
    }

    for year, iit in training_years_map.items():
        print(f"\nProcessing year {year} (Organized by {iit})...")
        features_df = engineer.create_master_feature_set(iit)
        truth_df = get_ground_truth(year, engineer.sql_engine)
        year_df = pd.merge(features_df, truth_df, on='topic_id', how='left').fillna(0)
        year_df['year'] = year
        all_years_data.append(year_df)
    
    engineer.close()
    
    if not all_years_data:
        print("❌ No data was generated for training. Halting.")
        return pd.DataFrame()

    master_training_df = pd.concat(all_years_data, ignore_index=True)
    print("\n✅ Master training dataset created successfully.")
    print(f"   Total training examples (Topics * Years): {len(master_training_df)}")
    return master_training_df

def train_fusion_model(df):
    """
    Trains an XGBoost model on the provided training dataframe.
    """
    print("\n--- Training the XGBoost Fusion Engine ---")
    
    features = ['monte_carlo_prob', 'nptel_heat_score', 'prof_bias_score']
    target = 'actual_weight'
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training on {len(X_train)} examples, validating on {len(X_test)} examples.")
    
    xgbr = xgb.XGBRegressor(
        objective='reg:squarederror', 
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=5,
        early_stopping_rounds=10,
        random_state=42
    )

    print("Fitting model...")
    xgbr.fit(
        X_train, y_train, 
        eval_set=[(X_test, y_test)], 
        verbose=False
    )
             
    # <<< --- THE FIX IS HERE --- >>>
    y_pred = xgbr.predict(X_test)
    # Calculate Mean Squared Error first
    mse = mean_squared_error(y_test, y_pred)
    # Then take the square root to get the Root Mean Squared Error
    rmse = np.sqrt(mse)
    # <<< --- END OF FIX --- >>>
    
    print(f"Model training complete. Validation RMSE: {rmse:.6f}")
    print("   (Lower RMSE is better. This shows the model's average prediction error.)")
    
    joblib.dump(xgbr, MODEL_OUTPUT_PATH)
    print(f"✅ Fusion Engine model saved to '{MODEL_OUTPUT_PATH}'")
    
    return xgbr


if __name__ == "__main__":
    print("======================================================")
    print(" GATE-ASTRA: FUSION ENGINE TRAINER (DAY 20)")
    print("======================================================")
    
    training_df = create_training_dataset()
    
    if not training_df.empty:
        train_fusion_model(training_df)
    
    print("======================================================")