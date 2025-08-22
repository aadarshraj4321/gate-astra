# FILE NAME: annotation_studio.py
# VERSION: 1.1 (Batch Workflow Enabled)
# Run with: streamlit run annotation_studio.py

import streamlit as st
import os
import json
import re

# --- Configuration ---
RAW_DATA_DIR = "raw_extracted_data"
CLEAN_DATA_DIR = "clean_parsed_data"

# --- Helper Functions ---
def get_session_state():
    """Initializes and manages the session state for our app."""
    if 'current_file_index' not in st.session_state:
        st.session_state.current_file_index = 0
    if 'annotated_questions' not in st.session_state:
        st.session_state.annotated_questions = []
    if 'raw_files' not in st.session_state:
        st.session_state.raw_files = sorted([f for f in os.listdir(RAW_DATA_DIR) if f.endswith('_raw.json')])
    # State for form inputs
    if 'q_num_input' not in st.session_state: st.session_state.q_num_input = 1
    if 'q_text_input' not in st.session_state: st.session_state.q_text_input = ""
    if 'options_input' not in st.session_state: st.session_state.options_input = ""
    if 'text_to_assist' not in st.session_state: st.session_state.text_to_assist = ""

def save_annotations():
    """Saves the currently annotated questions to a clean JSON file."""
    if not st.session_state.annotated_questions:
        st.error("No questions annotated for this file! Nothing to save.")
        return False

    raw_filename = st.session_state.raw_files[st.session_state.current_file_index]
    clean_filename = raw_filename.replace('_raw.json', '_clean.json')
    save_path = os.path.join(CLEAN_DATA_DIR, clean_filename)

    if not os.path.exists(CLEAN_DATA_DIR):
        os.makedirs(CLEAN_DATA_DIR)

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(st.session_state.annotated_questions, f, indent=4, ensure_ascii=False)
    
    st.success(f"✅ Saved {len(st.session_state.annotated_questions)} questions to {clean_filename}")
    return True

def ai_assistant_parser(text_chunk):
    """AI assistant to pre-fill fields from a raw text chunk."""
    if not text_chunk:
        return 0, "", []
        
    chunk = text_chunk.strip()
    
    # Try to find question number
    q_num_match = re.search(r'^(Q\.\s*(\d+))', chunk, re.IGNORECASE)
    q_num = int(q_num_match.group(2)) if q_num_match else st.session_state.q_num_input
    
    # Try to find options
    option_pattern = r'(\([A-D]\).*?)(?=\n\([A-D]\)|$)'
    options = re.findall(option_pattern, chunk, re.DOTALL)
    
    # Try to find question text (text before the first option)
    q_text = chunk
    if options:
        first_option_start = chunk.find(options[0])
        if first_option_start != -1:
            q_text = chunk[:first_option_start]

    # Clean up
    q_text = q_text.replace(q_num_match.group(0), '').strip() if q_num_match else q_text.strip()
    options = [opt.strip() for opt in options]

    return q_num, q_text, options

# --- Main App ---
st.set_page_config(layout="wide", page_title="GATE-Astra Annotation Studio")
st.title("🚀 GATE-Astra Annotation Studio v1.1")
st.info("This version enables a faster **Batch Annotation** workflow. See tips in the form.")

get_session_state()

# --- Sidebar ---
with st.sidebar:
    st.header("File Navigation")
    
    if not st.session_state.raw_files:
        st.error(f"'{RAW_DATA_DIR}' is empty. Run Day 3 script first.")
    else:
        selected_file = st.selectbox(
            "Select file:",
            st.session_state.raw_files,
            index=st.session_state.current_file_index,
            key='file_selector'
        )
        if st.session_state.raw_files[st.session_state.current_file_index] != selected_file:
            st.session_state.current_file_index = st.session_state.raw_files.index(selected_file)
            st.session_state.annotated_questions = []
            st.experimental_rerun()
        
        st.caption(f"File {st.session_state.current_file_index + 1} of {len(st.session_state.raw_files)}.")
        
        if st.button("💾 Save & Next ➡️"):
            if save_annotations():
                if st.session_state.current_file_index < len(st.session_state.raw_files) - 1:
                    st.session_state.current_file_index += 1
                    st.session_state.annotated_questions = []
                else:
                    st.warning("All files annotated!")
                st.experimental_rerun()

# --- Main Annotation Area ---
raw_filename = st.session_state.raw_files[st.session_state.current_file_index]
file_path = os.path.join(RAW_DATA_DIR, raw_filename)

with open(file_path, 'r', encoding='utf-8') as f:
    raw_text = json.load(f)['extracted_text']

col1, col2 = st.columns([0.4, 0.6])

with col1:
    st.header("Raw OCR Text")
    st.text_area("Full Text (Copy from here)", value=raw_text, height=600)

with col2:
    st.header("Annotation Form")
    
    # --- The Form for a Single Question ---
    st.subheader("Add a New Question")
    
    # AI Assistant
    st.markdown("**Workflow Tip:** Copy a large chunk with 5-10 questions into the box below for faster batch processing.")
    text_to_assist = st.text_area("Paste a text chunk here for the AI Assistant", height=200, key="text_to_assist")
    
    if st.button("🤖 Ask AI Assistant to Parse Chunk"):
        q_num, q_text, options = ai_assistant_parser(st.session_state.text_to_assist)
        st.session_state.q_num_input = q_num
        st.session_state.q_text_input = q_text
        st.session_state.options_input = "\n".join(options)

    # --- Manual Input Fields ---
    q_num = st.number_input("Question Number", min_value=1, step=1, key="q_num_input")
    q_text = st.text_area("Question Text", height=150, key="q_text_input")
    options_text = st.text_area("Options (one per line)", height=100, key="options_input")
    section = st.selectbox("Section", ["General Aptitude", "Computer Science", "Data Science"])

    if st.button("➕ Add Question to List"):
        if not q_text or not options_text:
            st.warning("Question Text and Options cannot be empty.")
        else:
            options = [opt.strip() for opt in options_text.split('\n') if opt.strip()]
            new_question = {
                "question_number": q_num, "question_text": q_text, "options": options,
                "answer": None, "explanation": None, "section": section
            }
            st.session_state.annotated_questions.append(new_question)
            
            # --- Auto-clear and prepare for next question ---
            st.session_state.q_num_input += 1
            st.session_state.q_text_input = ""
            st.session_state.options_input = ""
            st.success(f"Question #{new_question['question_number']} added!")

# --- Display Annotated Questions ---
st.header("📝 Annotated Questions for this File")
st.write(f"You have annotated **{len(st.session_state.annotated_questions)}** questions for **{raw_filename}**.")
if st.session_state.annotated_questions:
    st.table(st.session_state.annotated_questions)