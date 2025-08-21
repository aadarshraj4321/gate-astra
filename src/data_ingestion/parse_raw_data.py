# FILE NAME: src/data_ingestion/parse_raw_data.py
# VERSION: 1.0

import os
import json
import re

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# The folder where our messy, raw JSON files from Day 3 are.
RAW_DATA_DIR = "raw_extracted_data"

# The folder where we will save our new, clean, structured JSON files.
CLEAN_DATA_DIR = "clean_parsed_data"

# ==============================================================================
# PARSER LOGIC
# ==============================================================================

def preprocess_text(text):
    """
    Performs initial cleaning on the raw text to make parsing easier.
    """
    # Replace common OCR errors or inconsistencies
    text = text.replace('—', '-')
    # Normalize line breaks: reduce 3+ newlines to just 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def extract_components_from_chunk(chunk):
    """
    Takes a single question chunk and extracts its individual parts.
    This is where the detailed Regex magic happens.
    
    Returns:
        A dictionary containing the structured question data, or None if it's not a valid question.
    """
    # --- Pattern to find the question number ---
    # Matches "Q.1", "Q. 25", etc.
    q_num_match = re.search(r'^(Q\.\s*(\d+))', chunk)
    if not q_num_match:
        return None # This chunk is not a standard question block

    question_number_text = q_num_match.group(1) # The full "Q. X" string
    question_number_int = int(q_num_match.group(2)) # The integer part

    # --- Isolate text after the question number ---
    remaining_text = chunk[len(question_number_text):].strip()

    # --- Pattern to find options, e.g., (A), (B), (C), (D) ---
    # This pattern splits the text by the option markers
    # It's a bit complex: `(?=\(A\))` is a "positive lookahead" which splits
    # *before* the pattern without consuming it.
    option_pattern = r'(?=\([A-D]\))'
    parts = re.split(option_pattern, remaining_text, flags=re.IGNORECASE)
    
    if len(parts) < 2:
        return None # No options found, might be a malformed question
    
    # The first part is the question text
    question_text = parts[0].strip()
    
    # The rest of the parts are the options
    options = [opt.strip() for opt in parts[1:]]
    
    # Further clean the question text (remove trailing newlines from splitting)
    question_text = re.sub(r'\n{2,}$', '', question_text).strip()

    # --- Create the final structured object ---
    question_data = {
        "question_number": question_number_int,
        "question_text": question_text,
        "options": options,
        "answer": None,      # We will populate this later
        "explanation": None, # We will populate this later
        "section": None      # We will assign this in the main function
    }
    
    return question_data

def parse_single_file(raw_text):
    """
    Orchestrates the parsing process for the text from a single file.
    """
    # Stage 1: Pre-processing
    cleaned_text = preprocess_text(raw_text)
    
    # Add a marker to split the text by sections (GA vs Core)
    cleaned_text = re.sub(r'(General Aptitude)', r'---SECTION---\1', cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'(Computer Science and Information Technology)', r'---SECTION---\1', cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'(Data Science and AI)', r'---SECTION---\1', cleaned_text, flags=re.IGNORECASE)
    
    sections = cleaned_text.split('---SECTION---')
    
    parsed_questions = []
    current_section_name = "Unknown"
    
    for section_text in sections:
        if not section_text.strip():
            continue
            
        # Determine the name of the current section
        if "General Aptitude" in section_text[:50]:
            current_section_name = "General Aptitude"
        elif "Computer Science" in section_text[:50]:
            current_section_name = "Computer Science"
        elif "Data Science" in section_text[:50]:
            current_section_name = "Data Science"

        # Stage 2: Question Splitting
        # We split the text of the section by the question marker "Q. <number>"
        question_chunks = re.split(r'\n(?=Q\.\s*\d+)', section_text.strip())
        
        for chunk in question_chunks:
            # Stage 3: Component Extraction
            question_data = extract_components_from_chunk(chunk.strip())
            
            if question_data:
                # Add the section name we determined
                question_data["section"] = current_section_name
                parsed_questions.append(question_data)
                
    return parsed_questions


# ==============================================================================
# MAIN ORCHESTRATOR
# ==============================================================================

def main():
    """
    Main function to find all raw JSON files, parse them,
    and save the structured data to new JSON files.
    """
    print("======================================================")
    print(" GATE-ASTRA: INITIATING PARSER ENGINE (DAY 4)")
    print("======================================================")
    
    if not os.path.isdir(RAW_DATA_DIR):
        print(f"FATAL ERROR: Raw data directory '{RAW_DATA_DIR}' not found.")
        return

    # Create the output directory if it doesn't exist
    if not os.path.exists(CLEAN_DATA_DIR):
        os.makedirs(CLEAN_DATA_DIR)
        
    raw_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith('_raw.json')]
    
    if not raw_files:
        print(f"WARNING: No raw JSON files found in '{RAW_DATA_DIR}'. Halting.")
        return
        
    print(f"Found {len(raw_files)} raw files to parse.")

    for raw_filename in raw_files:
        print(f"\n--- Parsing: '{raw_filename}' ---")
        raw_filepath = os.path.join(RAW_DATA_DIR, raw_filename)
        
        try:
            with open(raw_filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                raw_text = data.get("extracted_text", "")
            
            if not raw_text:
                print("  -> WARNING: File is empty or has no 'extracted_text' key. Skipping.")
                continue

            # This is the main call to our parsing logic
            parsed_data = parse_single_file(raw_text)
            
            if parsed_data:
                # Save the clean, structured data
                clean_filename = raw_filename.replace('_raw.json', '_clean.json')
                clean_filepath = os.path.join(CLEAN_DATA_DIR, clean_filename)
                
                with open(clean_filepath, 'w', encoding='utf-8') as f:
                    json.dump(parsed_data, f, ensure_ascii=False, indent=4)
                print(f"  -> Successfully parsed {len(parsed_data)} questions.")
                print(f"  -> Saved clean data to '{clean_filepath}'")
            else:
                print("  -> WARNING: No questions could be parsed from this file.")

        except Exception as e:
            print(f"  -> CRITICAL ERROR processing file {raw_filename}. Reason: {e}")

    print("\n======================================================")
    print(" PARSING PROCESS COMPLETED.")
    print(f" Clean, structured data is now in the '{CLEAN_DATA_DIR}' directory.")
    print("======================================================")

if __name__ == "__main__":
    main()