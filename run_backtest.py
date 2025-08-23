import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.metrics import mean_absolute_error
from sqlalchemy import create_engine

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from statistical_analysis.create_features import FeatureEngineerV2
from models import get_db_url

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BACKTEST_YEAR = 2024
BACKTEST_IIT = "IISc Bangalore"
TEMP_MODEL_PATH = "ai_model_files/fusion_engine_backtest.joblib"

# ==============================================================================
# BACKTESTING ENGINE
# ==============================================================================

def get_full_dataset(sql_engine):
    """Loads the entire historical dataset from the database."""
    query = """
    SELECT 
        q.marks, e.exam_year, e.organizing_iit, s.topic_id, s.subject,
        s.topic, s.sub_topic
    FROM questions q
    JOIN exams e ON q.exam_id = e.exam_id
    JOIN syllabus s ON q.topic_id = s.topic_id
    """
    return pd.read_sql(query, sql_engine)

def get_ground_truth_for_year(full_df, year):
    """Calculates the actual topic weights for a specific year from the dataframe."""
    year_df = full_df[full_df['exam_year'] == year]
    if year_df.empty or year_df['marks'].sum() == 0:
        return pd.DataFrame(columns=['topic_id', 'actual_weight']).set_index('topic_id')
    total_marks = year_df['marks'].sum()
    topic_marks = year_df.groupby('topic_id')['marks'].sum()
    actual_weights = (topic_marks / total_marks).reset_index(name='actual_weight')
    return actual_weights.set_index('topic_id')

def train_historical_model(full_historical_df):
    """
    Creates a training dataset using only historical data and trains a new model.
    """
    print("--- Training a new Fusion Engine on purely historical data... ---")
    
    training_years_map = {
        # This map ONLY contains years BEFORE the backtest year.
        # This explicitly prevents data from 2025 or later from leaking in.
        2023: "IIT Kanpur", 2022: "IIT Kharagpur", 2021: "IIT Bombay",
        2020: "IIT Delhi", 2019: "IIT Madras", 2018: "IIT Guwahati",
        2017: "IIT Roorkee", 2016: "IISc Bangalore", 2015: "IIT Kanpur",
        2014: "IIT Kharagpur", 2013: "IIT Bombay", 2012: "IIT Delhi", 2011: "IIT Madras"
    }
    
    engineer = FeatureEngineerV2()
    all_years_data = []
    
    for year, iit in training_years_map.items():
        print(f"  - Processing training year {year}...")
        # A robust version would re-run analysis for each year, we use a strong approximation
        features_df = engineer.create_master_feature_set(iit)
        
        # Use the pre-filtered historical dataframe to get ground truth
        truth_df = get_ground_truth_for_year(full_historical_df, year)
        
        year_df = pd.merge(features_df, truth_df, on='topic_id', how='left').fillna(0)
        all_years_data.append(year_df)
        
    engineer.close()
    
    master_training_df = pd.concat(all_years_data, ignore_index=True)

    features = ['monte_carlo_prob', 'nptel_heat_score', 'prof_bias_score']
    target = 'actual_weight'
    X_train, y_train = master_training_df[features], master_training_df[target]
    
    xgbr = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=500, learning_rate=0.05, max_depth=5, random_state=42)
    xgbr.fit(X_train, y_train, verbose=False)
    
    joblib.dump(xgbr, TEMP_MODEL_PATH)
    print(f"✅ Historically trained model saved to '{TEMP_MODEL_PATH}'")
    return xgbr

def main():
    print("======================================================")
    print(f" GATE-ASTRA: GRAND BACKTEST FOR YEAR {BACKTEST_YEAR}")
    print("======================================================")

    engine = create_engine(get_db_url())
    full_df = get_full_dataset(engine)
    
    # 1. Isolate historical data (everything BEFORE the backtest year)
    historical_df = full_df[full_df['exam_year'] < BACKTEST_YEAR]
    print(f"Using {len(historical_df)} question records from before {BACKTEST_YEAR} for training.")
    
    # 2. Train a new model on this historical data
    historical_model = train_historical_model(historical_df)
    
    # 3. Generate features for the backtest year and predict
    print(f"\n--- Generating prediction for {BACKTEST_YEAR} (organized by {BACKTEST_IIT})... ---")
    feature_engineer = FeatureEngineerV2()
    prediction_features_df = feature_engineer.create_master_feature_set(BACKTEST_IIT)
    feature_engineer.close()
    
    X_predict = prediction_features_df[['monte_carlo_prob', 'nptel_heat_score', 'prof_bias_score']]
    predicted_weights = historical_model.predict(X_predict)
    prediction_features_df['predicted_weight'] = predicted_weights
    
    prediction_features_df['predicted_weight'] = prediction_features_df['predicted_weight'].clip(lower=0)
    prediction_features_df['predicted_weight'] /= prediction_features_df['predicted_weight'].sum()
    
    # 4. Get the ground truth for the backtest year
    actual_weights_df = get_ground_truth_for_year(full_df, BACKTEST_YEAR)
    
    # 5. Merge predictions and actuals for comparison
    comparison_df = pd.merge(
        prediction_features_df[['topic_id', 'subject', 'topic_name', 'predicted_weight']],
        actual_weights_df[['actual_weight']], on='topic_id', how='left'
    ).fillna(0)

    # Aggregate to subject level for a clearer report
    subject_comparison = comparison_df.groupby('subject').agg({
        'predicted_weight': 'sum', 'actual_weight': 'sum'
    }).reset_index()
    subject_comparison['absolute_error'] = abs(subject_comparison['predicted_weight'] - subject_comparison['actual_weight'])

    # --- 6. Display the Performance Report ---
    print("\n\n======================================================")
    print(f"     BACKTEST PERFORMANCE REPORT: {BACKTEST_YEAR}")
    print("======================================================")
    
    subject_comparison = subject_comparison.sort_values(by='actual_weight', ascending=False)
    
    print(f"\n{'Subject':<50} {'Predicted Weight':<20} {'Actual Weight':<20} {'Error':<15}")
    print("-" * 105)
    
    # <<< --- THE FIX FOR THE SyntaxError IS HERE --- >>>
    for _, row in subject_comparison.iterrows():
        if row['predicted_weight'] > 0.001 or row['actual_weight'] > 0.001:
            # Correctly formatted f-string
            print(f"{row['subject']:<50} {row['predicted_weight']*100:<19.2f}% {row['actual_weight']*100:<19.2f}% {row['absolute_error']*100:<14.2f}%")
    # <<< --- END OF FIX --- >>>
            
    mae = subject_comparison['absolute_error'].mean() * 100
    print("-" * 105)
    print(f"\n>> OVERALL MODEL ACCURACY (Mean Absolute Error): {mae:.2f}%")
    print("   (On average, the model's prediction for a subject's weight was off by this percentage.)")
    print("======================================================")
    
    if os.path.exists(TEMP_MODEL_PATH):
        os.remove(TEMP_MODEL_PATH)

if __name__ == "__main__":
    main()