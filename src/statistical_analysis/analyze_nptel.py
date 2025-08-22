# FILE NAME: src/statistical_analysis/analyze_nptel.py
# VERSION 2.0: Now gracefully handles empty JSON files.

import os
import sys
import json
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from collections import defaultdict

# --- Path setup ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ==============================================================================
# CONFIGURATION
# ==============================================================================
NPTEL_DATA_DIR = "nptel_data"
AI_DATA_DIR = "ai_model_files"
ANALYSIS_DIR = "analysis_results"

FINE_TUNED_MODEL_PATH = os.path.join(AI_DATA_DIR, 'GATE-Astra-Embed-v1')
VECTOR_DB_PATH = os.path.join(AI_DATA_DIR, "chroma_db")
COLLECTION_NAME = "syllabus_vectors"

OUTPUT_FILENAME = os.path.join(ANALYSIS_DIR, "nptel_topic_heat.json")
SIMILARITY_TOP_N = 1

# ==============================================================================
# THE NPTEL ANALYSIS ENGINE (CORRECTED)
# ==============================================================================

def analyze_nptel_data():
    """
    Analyzes NPTEL data to generate academic heat scores for syllabus topics.
    """
    print("======================================================")
    print(" GATE-ASTRA: NPTEL PULSE ANALYZER (DAY 16)")
    print("======================================================")

    # --- Step 1: Load the AI tools ---
    print(f"Loading custom model from '{FINE_TUNED_MODEL_PATH}'...")
    if not os.path.exists(FINE_TUNED_MODEL_PATH):
        print("❌ FATAL ERROR: Fine-tuned model not found. Please run Day 11 script.")
        return
    model = SentenceTransformer(FINE_TUNED_MODEL_PATH)
    print("✅ Custom model loaded.")

    print(f"Connecting to Vector DB at '{VECTOR_DB_PATH}'...")
    try:
        client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        collection = client.get_collection(name=COLLECTION_NAME)
        print(f"✅ Connected to ChromaDB collection '{COLLECTION_NAME}'. It contains {collection.count()} vectors.")
    except Exception as e:
        print(f"❌ FATAL ERROR: Could not connect to ChromaDB. Please run Day 12 script. Details: {e}")
        return

    # --- Step 2: Load and process all NPTEL course files ---
    if not os.path.isdir(NPTEL_DATA_DIR):
        print(f"❌ FATAL ERROR: NPTEL data directory '{NPTEL_DATA_DIR}' not found.")
        return

    all_lecture_titles = []
    nptel_files = [f for f in os.listdir(NPTEL_DATA_DIR) if f.endswith('.json')]
    print(f"Found {len(nptel_files)} NPTEL course files to analyze.")

    for filename in nptel_files:
        filepath = os.path.join(NPTEL_DATA_DIR, filename)
        
        # <<< --- THE FIX IS HERE --- >>>
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                # First, check if the file is empty
                if os.path.getsize(filepath) == 0:
                    print(f"  -> WARNING: Skipping empty file: {filename}")
                    continue
                
                # If not empty, try to load it
                data = json.load(f)
                
            for course in data.get("courses", []):
                all_lecture_titles.extend(course.get("lecture_list", []))
        except json.JSONDecodeError:
            print(f"  -> WARNING: Skipping file with invalid JSON format: {filename}")
            continue
        except Exception as e:
            print(f"  -> WARNING: Could not process file {filename}. Reason: {e}")
            continue
        # <<< --- END OF FIX --- >>>

    print(f"✅ Extracted a total of {len(all_lecture_titles)} lecture titles for analysis.")
    if not all_lecture_titles:
        print("No lecture titles found to analyze. Halting.")
        return

    # --- Step 3: The Semantic Analysis Loop ---
    topic_heat_scores = defaultdict(int)
    print("\nStarting semantic analysis of all lecture titles...")
    print("Encoding all lecture titles into vectors...")
    lecture_vectors = model.encode(all_lecture_titles, show_progress_bar=True, normalize_embeddings=True)

    print("Querying vector database for each lecture...")
    results = collection.query(query_embeddings=lecture_vectors, n_results=SIMILARITY_TOP_N)

    print("Aggregating heat scores...")
    if 'ids' in results:
        for top_matches_for_one_lecture in results['ids']:
            for topic_id_str in top_matches_for_one_lecture:
                topic_id = int(topic_id_str)
                topic_heat_scores[topic_id] += 1
    
    # --- Step 4: Save the final heatmap ---
    if not topic_heat_scores:
        print("⚠️ WARNING: No heat scores were generated.")
        return

    final_heatmap = {}
    print("Fetching topic names for the final report...")
    topic_ids_to_fetch = [str(tid) for tid in topic_heat_scores.keys()]
    
    if topic_ids_to_fetch:
        topic_details = collection.get(ids=topic_ids_to_fetch, include=["metadatas"])
        metadata_map = {topic_details['ids'][i]: topic_details['metadatas'][i] for i in range(len(topic_details['ids']))}
        for topic_id, score in topic_heat_scores.items():
            metadata = metadata_map.get(str(topic_id))
            if metadata:
                topic_name = f"{metadata['subject']} | {metadata['topic']} | {metadata.get('sub_topic') or 'General'}"
                final_heatmap[topic_id] = {"heat_score": score, "topic_name": topic_name}
    
    sorted_heatmap = dict(sorted(final_heatmap.items(), key=lambda item: item[1]['heat_score'], reverse=True))

    if not os.path.exists(ANALYSIS_DIR):
        os.makedirs(ANALYSIS_DIR)
        
    with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
        json.dump(sorted_heatmap, f, indent=4)
        
    print(f"\n✅ NPTEL Academic Heatmap created successfully!")
    print(f"   - Total Topics with Heat: {len(sorted_heatmap)}")
    print(f"   - Saved to: '{OUTPUT_FILENAME}'")
    print("======================================================")

if __name__ == "__main__":
    analyze_nptel_data()