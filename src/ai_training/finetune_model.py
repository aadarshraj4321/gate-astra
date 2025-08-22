# FILE NAME: src/ai_training/finetune_model.py
# PURPOSE: To train our custom AI embedding model.

import os
import sys
import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import math

# --- Path setup to import from parent directory (src) ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ==============================================================================
# CONFIGURATION
# ==============================================================================
AI_DATA_DIR = "ai_model_files"
TRAINING_DATA_FILE = os.path.join(AI_DATA_DIR, "finetune_dataset.csv")

# We use a well-regarded, lightweight base model. It's fast and effective.
BASE_MODEL = 'all-MiniLM-L6-v2'
# The path where our new, powerful model will be saved.
FINE_TUNED_MODEL_PATH = os.path.join(AI_DATA_DIR, 'GATE-Astra-Embed-v1')

# --- Training Hyperparameters ---
# These are settings that control the training process.
NUM_EPOCHS = 3           # How many times the model sees the entire dataset.
BATCH_SIZE = 16          # How many examples the model processes at once.
WARMUP_STEPS_PERCENT = 0.1 # A setting for the learning rate schedule.

# ==============================================================================
# THE TRAINING ENGINE
# ==============================================================================

def train_model():
    """
    Orchestrates the entire fine-tuning process from loading data to saving the model.
    """
    print("======================================================")
    print(" GATE-ASTRA: AI MODEL FINE-TUNING (DAY 11)")
    print("======================================================")

    # --- Step 1: Load the "Textbook" ---
    print(f"Loading the training textbook from '{TRAINING_DATA_FILE}'...")
    if not os.path.exists(TRAINING_DATA_FILE):
        print(f"❌ FATAL ERROR: Training file not found. Please run 'create_finetune_data.py' first.")
        return

    df = pd.read_csv(TRAINING_DATA_FILE)
    # Drop any rows with missing data for safety
    df.dropna(subset=['anchor', 'positive'], inplace=True)
    print(f"✅ Loaded {len(df)} training pairs.")

    # --- Step 2: Prepare the Training Examples ---
    print("Preparing training examples for the model...")
    train_examples = []
    for index, row in df.iterrows():
        # The InputExample format is what sentence-transformers expects.
        # We are telling it that the text from 'anchor' and 'positive' are a pair that should be similar.
        # The '1.0' label means they are a positive pair (maximum similarity).
        example = InputExample(texts=[row['anchor'], row['positive']], label=1.0)
        train_examples.append(example)

    # --- Step 3: Load the Base Model ---
    print(f"Loading the base model: '{BASE_MODEL}'...")
    # This will download the model from Hugging Face the first time you run it.
    model = SentenceTransformer(BASE_MODEL)
    print("✅ Base model loaded.")

    # --- Step 4: Define the Training Setup ---
    print("Defining the training loss and data loader...")
    # The DataLoader handles batching the data efficiently.
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
    
    # We use CosineSimilarityLoss. This is the perfect loss function for our goal.
    # It will adjust the model's weights to maximize the cosine similarity
    # between the vector for the question and the vector for its correct topic.
    train_loss = losses.CosineSimilarityLoss(model)

    # Calculate the number of warmup steps for the learning rate scheduler
    num_training_steps = len(train_dataloader) * NUM_EPOCHS
    warmup_steps = math.ceil(num_training_steps * WARMUP_STEPS_PERCENT)
    print(f"Training parameters: Epochs={NUM_EPOCHS}, Batch Size={BATCH_SIZE}, Total Steps={num_training_steps}, Warmup Steps={warmup_steps}")

    # --- Step 5: RUN THE TRAINING ---
    print("\n--- STARTING FINE-TUNING PROCESS ---")
    print("This may take several minutes depending on your hardware...")
    
    # The .fit() function is the magic. It handles the entire training loop.
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              epochs=NUM_EPOCHS,
              warmup_steps=warmup_steps,
              output_path=FINE_TUNED_MODEL_PATH,
              show_progress_bar=True)

    print("\n--- FINE-TUNING COMPLETE ---")
    print(f"✅ Your new expert model, 'GATE-Astra-Embed-v1', has been saved to:")
    print(f"   '{FINE_TUNED_MODEL_PATH}'")
    print("======================================================")

if __name__ == "__main__":
    train_model()