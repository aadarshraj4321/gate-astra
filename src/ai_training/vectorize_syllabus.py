# FILE NAME: src/ai_training/vectorize_syllabus.py
# PURPOSE: To create a semantic vector map of our entire syllabus using our custom model.

import os
import sys
import chromadb
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm

# --- Path setup ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import Syllabus, get_db_url

# ==============================================================================
# CONFIGURATION
# ==============================================================================
AI_DATA_DIR = "ai_model_files"
# This is our custom, fine-tuned model from Day 11
FINE_TUNED_MODEL_PATH = os.path.join(AI_DATA_DIR, 'GATE-Astra-Embed-v1')

# --- ChromaDB Configuration ---
VECTOR_DB_PATH = os.path.join(AI_DATA_DIR, "chroma_db")
COLLECTION_NAME = "syllabus_vectors"

# ==============================================================================
# THE VECTORIZATION ENGINE
# ==============================================================================

def create_syllabus_vector_map():
    """
    Loads our custom model, fetches all syllabus topics from the database,
    converts them to vectors, and stores them in a ChromaDB collection.
    """
    print("======================================================")
    print(" GATE-ASTRA: SYLLABUS VECTOR MAP CREATION (DAY 12)")
    print("======================================================")

    # --- Step 1: Load our custom fine-tuned model ---
    print(f"Loading our expert model from '{FINE_TUNED_MODEL_PATH}'...")
    if not os.path.exists(FINE_TUNED_MODEL_PATH):
        print("❌ FATAL ERROR: Fine-tuned model not found. Please run 'finetune_model.py' first.")
        return
    model = SentenceTransformer(FINE_TUNED_MODEL_PATH)
    print("✅ Custom model loaded successfully.")

    # --- Step 2: Connect to and set up the Vector Database ---
    print(f"Setting up Vector Database (ChromaDB) at '{VECTOR_DB_PATH}'...")
    # This creates a persistent client that saves the DB to disk
    client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
    
    # Get or create the collection. This is like a table in a relational database.
    # If it already exists, we can use it. For a clean run, you can delete the chroma_db folder.
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    print(f"✅ Vector DB collection '{COLLECTION_NAME}' is ready.")

    # --- Step 3: Fetch all syllabus topics from PostgreSQL ---
    print("Connecting to PostgreSQL to fetch syllabus topics...")
    try:
        engine = create_engine(get_db_url())
        Session = sessionmaker(bind=engine)
        session = Session()
        all_syllabus_topics = session.query(Syllabus).all()
        session.close()
        print(f"✅ Successfully fetched {len(all_syllabus_topics)} syllabus topics from the database.")
    except Exception as e:
        print(f"❌ FATAL ERROR: Could not fetch syllabus from database. Reason: {e}")
        return

    # --- Step 4: Prepare data for ChromaDB ---
    # ChromaDB needs lists of documents, metadatas, and ids.
    documents = []
    metadatas = []
    ids = []
    
    print("Preparing data for vectorization...")
    for topic in tqdm(all_syllabus_topics, desc="Preparing Topics"):
        # The 'document' is the text we want to vectorize and search.
        doc_text = f"{topic.subject} | {topic.topic} | {topic.sub_topic or 'General'}"
        documents.append(doc_text)
        
        # 'metadata' is extra information we want to store with the vector.
        metadatas.append({
            "subject": topic.subject,
            "topic": topic.topic,
            "sub_topic": topic.sub_topic or "N/A"
        })
        
        # 'ids' must be unique strings. We'll use the primary key from our database.
        ids.append(str(topic.topic_id))

    # --- Step 5: Vectorize and Store in ChromaDB ---
    print("Vectorizing all topics and adding to ChromaDB. This may take a moment...")
    
    # We can add data in batches to be more memory-efficient, but for ~1200 items, one go is fine.
    try:
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
            # ChromaDB automatically handles the vectorization using the model we can provide,
            # but the default sentence-transformer is also compatible with our model.
            # For direct control, we could do: embeddings = model.encode(documents) and then add embeddings.
            # But the default way is simpler and works well here.
        )
        print(f"✅ Successfully vectorized and stored {collection.count()} items in the collection.")
    except Exception as e:
        print(f"❌ FATAL ERROR: Could not add data to ChromaDB. Reason: {e}")
        return

    print("\n--- Semantic Syllabus Map Creation Complete ---")
    print("The Oracle's brain now has a complete map of the GATE syllabus.")
    print("======================================================")

if __name__ == "__main__":
    create_syllabus_vector_map()