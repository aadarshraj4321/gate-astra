import os
import sys
import json
import pandas as pd
from neo4j import GraphDatabase, basic_auth
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import chromadb
from tqdm import tqdm

# --- Path setup ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import get_db_url, Syllabus

# ==============================================================================
# CONFIGURATION
# ==============================================================================
ANALYSIS_DIR = "analysis_results"
AI_DATA_DIR = "ai_model_files"

SIMULATION_RESULTS_FILE = os.path.join(ANALYSIS_DIR, "simulation_results.json")
NPTEL_HEAT_FILE = os.path.join(ANALYSIS_DIR, "nptel_topic_heat.json")
OUTPUT_FILE = os.path.join(ANALYSIS_DIR, "master_features_v2.csv") # New output file

VECTOR_DB_PATH = os.path.join(AI_DATA_DIR, "chroma_db")
COLLECTION_NAME = "syllabus_vectors"

# --- Spreading Parameters ---
NEIGHBORHOOD_SIZE = 10 # Find the top 10 neighbors
DECAY_FACTOR = 0.8   # Each neighbor gets 80% of the score of the previous one


class FeatureEngineerV2:
    def __init__(self):
        """Initializes connections and loads all necessary data and models."""
        load_dotenv()
        self.neo4j_driver = GraphDatabase.driver(os.getenv("NEO4J_URI"), auth=basic_auth(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")))
        self.sql_engine = create_engine(get_db_url())
        self.SqlSession = sessionmaker(bind=self.sql_engine)
        
        # Load feature files
        with open(SIMULATION_RESULTS_FILE, 'r') as f: self.simulation_results = json.load(f)
        with open(NPTEL_HEAT_FILE, 'r') as f: self.nptel_heat = json.load(f)
        
        # Connect to Vector DB
        self.chroma_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        self.syllabus_collection = self.chroma_client.get_collection(name=COLLECTION_NAME)
        
        # Pre-fetch all topic IDs from Chroma to validate results
        all_ids = self.syllabus_collection.get(include=[])['ids']
        self.all_topic_ids_str = set(all_ids)

    def close(self):
        self.neo4j_driver.close()

    def get_professor_specialization_topics(self, organizing_iit):
        """Queries Neo4j to find the direct specialization topics for an IIT's profs."""
        query = """
        MATCH (i:IIT {name: $iit_name})<-[:WORKS_AT]-(p:Professor)-[r:SPECIALIZES_IN]->(t:Topic)
        RETURN t.topic_id AS topic_id, SUM(r.confidence) AS initial_score
        """
        with self.neo4j_driver.session() as session:
            results = session.run(query, iit_name=organizing_iit)
            return {str(record["topic_id"]): record["initial_score"] for record in results}

    def spread_signal_semantically(self, initial_scores_dict):
        """
        Takes a sparse dictionary of scores and spreads it across semantic neighbors.
        e.g., { "52": 1.0 } -> { "52": 1.0, "850": 0.8, "851": 0.64, ... }
        """
        final_scores = {}
        if not initial_scores_dict:
            return {}

        # Get the vectors for the initial "hot" topics from ChromaDB
        hot_topic_ids = list(initial_scores_dict.keys())
        # Filter out any potential bad IDs before querying
        valid_hot_topic_ids = [tid for tid in hot_topic_ids if tid in self.all_topic_ids_str]
        
        if not valid_hot_topic_ids:
            return {}
            
        initial_vectors = self.syllabus_collection.get(ids=valid_hot_topic_ids, include=['embeddings'])['embeddings']
        
        # For each hot topic, find its neighbors
        neighbor_results = self.syllabus_collection.query(
            query_embeddings=initial_vectors,
            n_results=NEIGHBORHOOD_SIZE
        )

        # Iterate through each initial hot topic and its neighbors to spread the score
        for i, topic_id in enumerate(valid_hot_topic_ids):
            initial_score = initial_scores_dict[topic_id]
            
            # The neighbors are returned as a list of lists of IDs and distances
            neighbor_ids = neighbor_results['ids'][i]
            neighbor_distances = neighbor_results['distances'][i]

            for j, neighbor_id in enumerate(neighbor_ids):
                # The score decays with distance/rank
                decayed_score = initial_score * (DECAY_FACTOR ** j)
                # Add the score, accumulating if the node is a neighbor to multiple hot topics
                final_scores[neighbor_id] = final_scores.get(neighbor_id, 0) + decayed_score
                
        return final_scores

    def create_master_feature_set(self, organizing_iit):
        """Assembles the new, dense features from all sources."""
        sql_session = self.SqlSession()
        all_syllabus_topics = sql_session.query(Syllabus).all()
        sql_session.close()

        print("Assembling features from all sources...")

        # 1. Get sparse initial signals
        print("Step 1: Getting initial sparse signals...")
        initial_prof_scores = self.get_professor_specialization_topics(organizing_iit)
        initial_nptel_scores = {topic_id: data['heat_score'] for topic_id, data in self.nptel_heat.items()}

        # 2. Spread the signals
        print("Step 2: Spreading signals across semantic neighborhoods...")
        dense_prof_bias = self.spread_signal_semantically(initial_prof_scores)
        dense_nptel_heat = self.spread_signal_semantically(initial_nptel_scores)
        
        # 3. Assemble the final DataFrame
        print("Step 3: Assembling final feature DataFrame...")
        mc_subject_probs = self.simulation_results['mean_weightage']
        feature_list = []

        for topic in tqdm(all_syllabus_topics, desc="Building feature set"):
            topic_id_str = str(topic.topic_id)
            
            feature_list.append({
                "topic_id": topic.topic_id,
                "subject": topic.subject,
                "topic_name": f"{topic.topic} - {topic.sub_topic or 'General'}",
                "monte_carlo_prob": mc_subject_probs.get(topic.subject, 0),
                "nptel_heat_score": dense_nptel_heat.get(topic_id_str, 0),
                "prof_bias_score": dense_prof_bias.get(topic_id_str, 0)
            })
            
        df = pd.DataFrame(feature_list)
        # Normalize the scores to be between 0 and 1 for better ML performance
        for col in ['nptel_heat_score', 'prof_bias_score']:
            if df[col].max() > 0:
                df[col] = df[col] / df[col].max()
                
        return df

def main(organizing_iit="IIT Delhi"):
    print("======================================================")
    print(" GATE-ASTRA: SEMANTIC FEATURE ENGINE (DAY 19 V2)")
    print("======================================================")
    
    engineer = FeatureEngineerV2()
    try:
        master_df = engineer.create_master_feature_set(organizing_iit)
        master_df.to_csv(OUTPUT_FILE, index=False)
        
        print(f"\nMaster feature set (V2) created successfully for {organizing_iit}.")
        print(f"   Saved to: '{OUTPUT_FILE}'")
        
        print("\n--- Feature Preview (Sorted by combined score) ---")
        master_df['combined_score'] = master_df['nptel_heat_score'] + master_df['prof_bias_score']
        print(master_df.sort_values(by='combined_score', ascending=False).head(15))
        print("--------------------------------------------------")
        
    finally:
        engineer.close()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_iit = " ".join(sys.argv[1:])
        main(organizing_iit=target_iit)
    else:
        main()