# FILE NAME: src/data_ingestion/extract_from_pdfs.py
# VERSION: 2.0 (OCR-Powered)

# --- Core Python Libraries ---
import os
import json
import io  # Used for handling data in memory

# --- Third-Party Libraries ---
# Make sure to install these: pip install PyMuPDF pytesseract Pillow
import fitz          # PyMuPDF, for handling PDF files
import pytesseract   # The Python wrapper for Google's Tesseract OCR engine
from PIL import Image # Pillow library for image manipulation

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# The folder where you have saved all the downloaded PDF files.
# This folder must exist in the root of your project directory (gate-astra/).
PDF_SOURCE_DIR = "pdf_question_papers"

# The folder where the script will save the output JSON files.
# This folder will be created automatically if it doesn't exist.
OUTPUT_DIR = "raw_extracted_data"

# ==============================================================================
# OCR-POWERED PDF EXTRACTION ENGINE
# ==============================================================================

def extract_text_from_pdf_with_ocr(pdf_path):
    """
    Opens a PDF and robustly extracts text using Optical Character Recognition (OCR)
    for each page. This is essential for image-based PDFs where simple text
    extraction returns blank or garbled text.
    
    Args:
        pdf_path (str): The full path to the PDF file to be processed.
        
    Returns:
        str: A single string containing all OCR-recognized text from the PDF.
             Returns None if a critical error occurs (e.g., Tesseract not found).
    """
    try:
        # Step 1: Open the PDF file using PyMuPDF (fitz)
        document = fitz.open(pdf_path)
        
        # Initialize an empty string to accumulate text from all pages
        full_text = ""
        
        num_pages = len(document)
        print(f"  -> Found {num_pages} pages. Starting OCR process...")
        
        # Step 2: Loop through each page in the document
        for page_num in range(num_pages):
            page = document.load_page(page_num)
            
            # Step 3: Convert the PDF page into a high-resolution image
            # The DPI (dots per inch) is crucial for OCR accuracy. 300 is a good standard.
            pixmap = page.get_pixmap(dpi=300)
            
            # Convert the pixmap data into a byte stream that the Pillow library can read
            img_bytes = pixmap.tobytes("png")
            
            # Create an in-memory image object from the byte stream
            image = Image.open(io.BytesIO(img_bytes))

            # Step 4: Use Tesseract to perform OCR on the image
            try:
                # This is the core OCR function call.
                # It "reads" the image and converts the characters into a string.
                # `lang='eng'` tells Tesseract to use the English language model.
                page_text = pytesseract.image_to_string(image, lang='eng')
                
                # Append the extracted text and a page separator to our main string
                full_text += page_text
                full_text += f"\n\n--- PAGE {page_num + 1} ---\n\n"
                print(f"     ... Page {page_num + 1}/{num_pages} successfully processed.")

            except pytesseract.TesseractNotFoundError:
                # This is a critical error if the Tesseract program itself is not installed
                print("\n" + "="*50)
                print("FATAL ERROR: TESSERACT NOT FOUND")
                print("The 'tesseract' command was not found in your system's PATH.")
                print("Please ensure you have installed the Tesseract OCR engine.")
                print(" - macOS: 'brew install tesseract'")
                print(" - Windows: Download from official repo and ensure 'Add to PATH' is checked.")
                print(" - Linux: 'sudo apt install tesseract-ocr'")
                print("="*50 + "\n")
                return None # Halt the entire process
            
            except Exception as ocr_error:
                # Handle errors during the OCR of a single page
                print(f"  -> WARNING: OCR failed for page {page_num + 1}. Error: {ocr_error}")
                full_text += f"\n\n--- OCR FAILED FOR PAGE {page_num + 1} ---\n\n"

        # Step 5: Close the PDF document to release the file handle
        document.close()
        return full_text
        
    except Exception as e:
        # Handle errors related to opening or reading the PDF file itself
        print(f"  -> ERROR: Failed to open or process the PDF file '{pdf_path}'.")
        print(f"     Reason: {e}")
        return None

# ==============================================================================
# FILE HANDLING AND ORCHESTRATION
# ==============================================================================

def save_raw_data_to_json(text_content, original_pdf_name, output_dir):
    """
    Saves the extracted raw text into a structured JSON file.
    
    Args:
        text_content (str): The text extracted from the PDF.
        original_pdf_name (str): The filename of the source PDF.
        output_dir (str): The directory where the JSON file will be saved.
    """
    # Ensure the output directory exists. If not, create it.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create a clean filename for the JSON output
    json_filename = os.path.splitext(original_pdf_name)[0] + "_raw.json"
    file_path = os.path.join(output_dir, json_filename)
    
    # Structure the data to be saved
    output_data = {
        "source_pdf": original_pdf_name,
        "extracted_text": text_content
    }

    # Write the dictionary to a JSON file with pretty printing
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)
        print(f"  -> Successfully saved extracted text to '{file_path}'")
    except Exception as e:
        print(f"  -> ERROR: Failed to write JSON file '{file_path}'. Reason: {e}")

# ==============================================================================
# MAIN ORCHESTRATOR - The script's entry point (CORRECTED VERSION)
# ==============================================================================

def main():
    """
    This main function orchestrates the entire process:
    1. It finds the PDF source directory.
    2. It lists all PDF files inside it.
    3. It loops through each file, extracts the text, and saves it to a JSON.
    """
    print("======================================================")
    print(" GATE-ASTRA: INITIATING OCR-POWERED PDF ENGINE (DAY 3)")
    print("======================================================")
    
    # --- Step 1: Validate the source directory ---
    if not os.path.isdir(PDF_SOURCE_DIR):
        print(f"FATAL ERROR: The source directory '{PDF_SOURCE_DIR}' was not found.")
        print("Please create this folder in the project's root and place your PDF files inside.")
        return

    # --- Step 2: Find all PDF files ---
    try:
        pdf_files = [f for f in os.listdir(PDF_SOURCE_DIR) if f.lower().endswith('.pdf')]
    except Exception as e:
        print(f"FATAL ERROR: Could not read files from '{PDF_SOURCE_DIR}'. Reason: {e}")
        return

    if not pdf_files:
        print(f"WARNING: No PDF files were found in the '{PDF_SOURCE_DIR}' directory. Halting.")
        return
        
    print(f"Found {len(pdf_files)} PDF files to process.")

    # --- Step 3: Loop and Process Each File ---
    for pdf_filename in pdf_files:
        print(f"\n--- Processing: '{pdf_filename}' ---")
        full_pdf_path = os.path.join(PDF_SOURCE_DIR, pdf_filename)
        
        # Call the main OCR extraction function
        # THIS IS THE CORRECTED LINE:
        extracted_text = extract_text_from_pdf_with_ocr(full_pdf_path)
        
        # If the function returns None because Tesseract is not found, we should stop.
        if extracted_text is None:
            # A simple check to see if the error was likely due to Tesseract
            # This is a bit of a hack, but good enough for our script.
            # A more robust solution would involve custom exceptions.
            try:
                pytesseract.get_tesseract_version()
            except pytesseract.TesseractNotFoundError:
                print("Halting process because Tesseract is not configured correctly.")
                break # Exit the loop
        
        # If text was extracted successfully, save it.
        if extracted_text:
            save_raw_data_to_json(extracted_text, pdf_filename, OUTPUT_DIR)
        else:
            print("  -> Skipping this file due to a read error or empty content.")
            
    print("\n======================================================")
    print(" PDF TEXT EXTRACTION PROCESS COMPLETED.")
    print(f" Raw text data saved in the '{OUTPUT_DIR}' directory.")
    print("======================================================")

# This standard Python construct ensures that the main() function is called
# only when the script is executed directly.
if __name__ == "__main__":
    main()
