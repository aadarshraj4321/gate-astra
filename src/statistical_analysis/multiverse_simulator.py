# FILE NAME: src/statistical_analysis/multiverse_simulator.py
# VERSION 4.0: Added strict input validation. The Oracle is now honest.

import json
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import dirichlet
from tqdm import tqdm as TqdmProgressBar

# --- Path setup ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ==============================================================================
# CONFIGURATION
# ==============================================================================
ANALYSIS_DIR = "analysis_results"
PATTERNS_FILE = os.path.join(ANALYSIS_DIR, "historical_patterns.json")
OUTPUT_FILE = os.path.join(ANALYSIS_DIR, "simulation_results.json")

# --- Simulation Hyperparameters ---
NUM_SIMULATIONS = 1000000
CONFIDENCE_LEVEL = 0.95
TOP_SUBJECTS_TO_DRILLDOWN = 7
TOP_SUB_TOPICS_TO_SHOW = 4

# ==============================================================================
# THE SIMULATOR ENGINE
# ==============================================================================

class MultiverseSimulator:
    def __init__(self, patterns_path):
        print(f"Loading historical patterns from '{patterns_path}'...")
        if not os.path.exists(patterns_path):
            raise FileNotFoundError("Patterns file not found. Please run 'analyze_history.py' first.")
        with open(patterns_path, 'r') as f:
            self.patterns = json.load(f)
            
        self.subjects = sorted(self.patterns['overall_subject_probabilities'].keys())
        self.base_probabilities = np.array([self.patterns['overall_subject_probabilities'][s] for s in self.subjects])
        
        # <<< NEW: Store the list of valid IITs >>>
        self.valid_iits = list(self.patterns['iit_biases'].keys())
        
        print("✅ Patterns loaded successfully.")

    def run_single_simulation(self, organizing_iit):
        biased_probs = self.base_probabilities.copy()
        
        # This check should now always pass because we validate the input beforehand.
        if organizing_iit in self.patterns['iit_biases']:
            iit_bias = self.patterns['iit_biases'][organizing_iit]
            for i, subject in enumerate(self.subjects):
                biased_probs[i] += iit_bias.get(subject, 0)
        
        biased_probs[biased_probs < 0.001] = 0.001
        biased_probs /= np.sum(biased_probs)
        alpha = biased_probs * 100
        return dirichlet.rvs(alpha, size=1)[0]

    def run_multiverse(self, organizing_iit, num_simulations):
        print(f"\n--- Running {num_simulations} simulations for {organizing_iit} ---")
        all_simulations = [self.run_single_simulation(organizing_iit) for _ in TqdmProgressBar(range(num_simulations), desc="Simulating Universes")]
        results_df = pd.DataFrame(all_simulations, columns=self.subjects)
        print("--- Simulation Complete. Analyzing results... ---")
        return self._analyze_results(results_df)

    def _analyze_results(self, results_df):
        analysis = {}
        analysis['mean_weightage'] = results_df.mean().to_dict()
        analysis['std_dev'] = results_df.std().to_dict()
        lower_q = (1 - CONFIDENCE_LEVEL) / 2
        upper_q = 1 - lower_q
        analysis['confidence_interval_lower'] = results_df.quantile(lower_q).to_dict()
        analysis['confidence_interval_upper'] = results_df.quantile(upper_q).to_dict()
        return analysis

def main(organizing_iit="IIT Bombay"):
    print("======================================================")
    print(" GATE-ASTRA: MULTIVERSE SIMULATOR (V3 - Strict & Honest)")
    print("======================================================")
    
    try:
        simulator = MultiverseSimulator(PATTERNS_FILE)

        # <<< NEW: STRICT INPUT VALIDATION >>>
        if organizing_iit not in simulator.valid_iits:
            print(f"\n❌ FATAL ERROR: Invalid organizing IIT '{organizing_iit}'.")
            print("   This IIT was not found in our historical data.")
            print("\n   Please choose from one of the following valid options:")
            for iit in sorted(simulator.valid_iits):
                print(f"     - \"{iit}\"")
            print("\n   Note: The name must be an exact match and is case-sensitive.")
            return # Stop execution
        
        simulation_results = simulator.run_multiverse(organizing_iit, NUM_SIMULATIONS)
        
        # (The rest of the print and save logic remains the same)
        print("\n\n==========================================================================")
        print(f"  TOPIC HEATMAP & SUB-TOPIC DRILL-DOWN FOR {organizing_iit.upper()}")
        print("==========================================================================")
        print(f"Based on {NUM_SIMULATIONS} simulations with a {CONFIDENCE_LEVEL*100}% confidence level.\n")
        mean_weights = simulation_results['mean_weightage']
        sorted_subjects = sorted(mean_weights.keys(), key=lambda s: mean_weights[s], reverse=True)
        print(f"{'Subject':<50} {'Predicted Weight':<20} {'Confidence Interval':<25}")
        print("-" * 95)
        top_subjects_for_drilldown = [s for i, s in enumerate(sorted_subjects) if mean_weights[s]*100 >= 0.5 and i < TOP_SUBJECTS_TO_DRILLDOWN]
        for subject in sorted_subjects:
            mean_val = mean_weights[subject] * 100
            if mean_val < 0.5: continue
            lower_val = simulation_results['confidence_interval_lower'][subject] * 100
            upper_val = simulation_results['confidence_interval_upper'][subject] * 100
            print(f"{subject:<50} {f'{mean_val:.2f}%':<20} {f'({lower_val:.2f}% - {upper_val:.2f}%)':<25}")
        print("\n\n--- SUB-TOPIC DRILL-DOWN (Probable Hot Topics) ---\n")
        sub_topic_patterns = simulator.patterns.get("sub_topic_probabilities_by_subject", {})
        for subject in top_subjects_for_drilldown:
            print(f"🔍 INSIDE: {subject} (Total Weight: {mean_weights.get(subject, 0)*100:.2f}%)")
            print("-" * 70)
            sub_topics = sub_topic_patterns.get(subject, {})
            if not sub_topics:
                print("  -> No detailed sub-topic data available for this subject.\n")
                continue
            sorted_sub_topics = sorted(sub_topics.items(), key=lambda item: item[1], reverse=True)
            for i, (sub_topic_name, prob) in enumerate(sorted_sub_topics):
                if i >= TOP_SUB_TOPICS_TO_SHOW: break
                print(f"  - {sub_topic_name:<60} (Probability within subject: {prob*100:.2f}%)")
            print("\n")
        print("==========================================================================\n")
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(simulation_results, f, indent=4)
        print(f"Detailed simulation results saved to '{OUTPUT_FILE}'")
        
    except Exception as e:
        print(f"\nAN ERROR OCCURRED: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_iit = " ".join(sys.argv[1:])
        main(organizing_iit=target_iit)
    else:
        main()