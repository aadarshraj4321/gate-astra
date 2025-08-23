import os
import sys
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from neo4j import GraphDatabase, basic_auth
from sentence_transformers import SentenceTransformer
import chromadb
from tqdm import tqdm

# --- Path setup to import our models ---
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.models import get_db_url, Syllabus, Question, Exam, Professor

# ==============================================================================
# CONFIGURATION
# ==============================================================================
AI_DATA_DIR = "ai_model_files"
FINE_TUNED_MODEL_PATH = os.path.join(AI_DATA_DIR, 'GATE-Astra-Embed-v1')

# ==============================================================================
# THE GRAPH BUILDER ENGINE
# ==============================================================================

class KnowledgeGraphBuilder:
    def __init__(self):
        """Initializes connections to all our data sources."""
        print("Initializing all database and AI model connections...")
        load_dotenv()
        
        # Connect to PostgreSQL
        self.sql_engine = create_engine(get_db_url())
        self.SqlSession = sessionmaker(bind=self.sql_engine)
        
        # Connect to Neo4j
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")
        self.neo4j_driver = GraphDatabase.driver(uri, auth=basic_auth(user, password))
        
        # Load our custom AI model
        self.embed_model = SentenceTransformer(FINE_TUNED_MODEL_PATH)
        print("All connections initialized.")

    def close(self):
        """Closes the Neo4j database connection."""
        self.neo4j_driver.close()
        print("Neo4j connection closed.")

    def wipe_graph(self):
        """Deletes all nodes and relationships from the graph for a fresh start."""
        print("Wiping the entire Neo4j database clean...")
        with self.neo4j_driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("Graph wiped successfully.")

    def run_cypher_tx(self, query, parameters=None):
        """Helper function to run a transaction in Neo4j."""
        with self.neo4j_driver.session() as session:
            session.run(query, parameters)

    def build_graph(self):
        """The main orchestration method to build the entire graph."""
        sql_session = self.SqlSession()
        
        self.wipe_graph()

        print("\n--- PHASE 1: Creating Syllabus and IIT nodes ---")
        all_syllabus = sql_session.query(Syllabus).all()
        all_exams = sql_session.query(Exam).all()
        
        # Create Topic nodes
        for topic in tqdm(all_syllabus, desc="Creating Topic nodes"):
            query = """
            MERGE (t:Topic {topic_id: $topic_id})
            SET t.name = $name, t.subject = $subject
            """
            # Create a clean name for the node
            name = f"{topic.topic} - {topic.sub_topic or 'General'}"
            self.run_cypher_tx(query, parameters={
                "topic_id": topic.topic_id,
                "name": name,
                "subject": topic.subject
            })
            
        # Create IIT nodes
        unique_iits = {exam.organizing_iit for exam in all_exams}
        for iit_name in tqdm(unique_iits, desc="Creating IIT nodes"):
            self.run_cypher_tx("MERGE (i:IIT {name: $name})", parameters={"name": iit_name})
            
        print("Fundamental nodes created.")

        # --- PHASE 2: Create Question nodes and link to Topics and IITs ---
        print("\n--- PHASE 2: Creating Question nodes and relationships ---")
        all_questions = sql_session.query(Question).join(Exam).all()
        for q in tqdm(all_questions, desc="Creating Question nodes"):
            query = """
            MATCH (t:Topic {topic_id: $topic_id})
            MATCH (i:IIT {name: $iit_name})
            CREATE (q:Question {question_id: $q_id, marks: $marks, type: $type})
            MERGE (q)-[:IS_ABOUT]->(t)
            MERGE (q)-[:WAS_IN_EXAM_BY]->(i)
            """
            self.run_cypher_tx(query, parameters={
                "topic_id": q.topic_id,
                "iit_name": q.exam.organizing_iit,
                "q_id": q.question_id,
                "marks": q.marks,
                "type": q.question_type
            })
        print("Question nodes and relationships created.")

        # --- PHASE 3: Create Professor nodes and link to IITs ---
        print("\n--- PHASE 3: Creating Professor nodes and relationships ---")
        all_professors = sql_session.query(Professor).all()
        if not all_professors:
            print("No professors found in PostgreSQL. Skipping professor-related graph creation.")
        else:
            # Create Professor nodes and link them to their IITs
            for prof in tqdm(all_professors, desc="Creating Professor nodes"):
                query = """
                MATCH (i:IIT {name: $iit_name})
                CREATE (p:Professor {name: $prof_name})
                MERGE (p)-[:WORKS_AT]->(i)
                """
                self.run_cypher_tx(query, parameters={
                    "iit_name": prof.current_iit,
                    "prof_name": prof.name
                })
            
            print("\n--- PHASE 4: Creating intelligent Professor->Topic links ---")
            # Create vector embeddings for all of our syllabus topics
            topic_texts = [f"{t.subject} | {t.topic} | {t.sub_topic or 'General'}" for t in all_syllabus]
            topic_embeddings = self.embed_model.encode(topic_texts, normalize_embeddings=True)

            for prof in tqdm(all_professors, desc="Linking Professors to Topics"):
                if not prof.research_interests: continue
                
                # Create a single string of research interests
                interests_text = ", ".join(prof.research_interests)
                # Convert this to a vector
                interests_embedding = self.embed_model.encode(interests_text, normalize_embeddings=True)
                
                # Calculate cosine similarity between professor's interests and all topics
                similarities = interests_embedding @ topic_embeddings.T
                
                # Find the top 3 most similar topics
                top_indices = np.argsort(similarities)[-3:]
                
                for index in top_indices:
                    if similarities[index] > 0.5: # Only create a link if similarity is decent
                        matched_topic = all_syllabus[index]
                        query = """
                        MATCH (p:Professor {name: $prof_name})
                        MATCH (t:Topic {topic_id: $topic_id})
                        MERGE (p)-[r:SPECIALIZES_IN]->(t)
                        SET r.confidence = $confidence
                        """
                        self.run_cypher_tx(query, parameters={
                            "prof_name": prof.name,
                            "topic_id": matched_topic.topic_id,
                            "confidence": float(similarities[index])
                        })

            print("Professor nodes and intelligent links created.")

        sql_session.close()


if __name__ == "__main__":
    # We need numpy for this script
    try:
        import numpy as np
    except ImportError:
        print("Please install numpy: pip install numpy")
        sys.exit(1)
        
    builder = KnowledgeGraphBuilder()
    try:
        builder.build_graph()
        print("\n======================================================")
        print(" KNOWLEDGE GRAPH BUILD COMPLETE!")
        print(" You can now explore the graph in your Neo4j Browser.")
        print("======================================================")
    finally:
        builder.close()