# FILE NAME: src/data_ingestion/scrape_papers.py

import requests
from bs4 import BeautifulSoup, Tag
import json
import time
import os
import re # We'll use regular expressions for more precise matching
import sys

# Allow importing from the parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# The main table page on GeeksForGeeks containing all the links
LINK_COLLECTOR_URL = "https://www.geeksforgeeks.org/gate/original-gate-previous-year-question-papers-cse-and-it-gq/"

# Keywords to identify the links we care about in the table
# We want "Computer Science" and the new "Data Science" papers
PAPER_KEYWORDS = ["CS", "DA"]

OUTPUT_DIR = "raw_scraped_data"
REQUEST_DELAY_S = 3 # Be polite to GFG servers

# ==============================================================================
# STAGE 1: LINK COLLECTOR
# ==============================================================================
# ==============================================================================
# STAGE 1: LINK COLLECTOR (Corrected & State-Aware Version)
# ==============================================================================
def collect_paper_links(url):
    """
    Visits the main GFG table page and scrapes the URLs for all relevant papers.
    This version is state-aware, meaning it remembers the context (year, paper type)
    from header rows as it iterates through the table.
    """
    print(f"--- Stage 1: Collecting paper links from {url} ---")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36'}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        paper_links = {}
                # The main content area containing the tables
        # NEW RESILIENT CODE
        # Try to find the main content area using a few common GFG selectors in order of priority.
        content_selectors = ['div.entry-content', 'article.type-post', 'div#main']
        entry_content = None
        for selector in content_selectors:
            entry_content = soup.select_one(selector)
            if entry_content:
                print(f"  -> Found main content area using selector: '{selector}'")
                break

        if not entry_content:
            print("  -> FATAL ERROR: Could not find the main content area using any known selectors.")
            # For debugging, we can save the HTML to see what's wrong
            # with open("debug_page.html", "w", encoding="utf-8") as f:
            #     f.write(str(soup))
            return {}

        all_tables = entry_content.find_all('table')
        print(f"  -> Found {len(all_tables)} tables on the page.")

        current_year = ""
        current_paper_type = ""

        # We process all tables, as the content is spread across them
        for table in all_tables:
            for row in table.find_all('tr'):
                header_cell = row.find('th')
                
                # --- State Update Logic ---
                # Check if this row is a main header like "2024 [CS]"
                if header_cell and header_cell.get('colspan'):
                    header_text = header_cell.get_text(strip=True)
                    year_match = re.search(r'(\d{4})', header_text)
                    if year_match:
                        current_year = year_match.group(1)
                        if "[CS]" in header_text:
                            current_paper_type = "CS"
                        elif "[DA]" in header_text:
                            current_paper_type = "DA"
                        print(f"  -> State updated: Year={current_year}, Type={current_paper_type}")
                    continue # Move to the next row after processing header

                # --- Link Extraction Logic ---
                cells = row.find_all('td')
                # A typical link row has 3 or 4 cells
                if len(cells) >= 3 and current_year and current_paper_type:
                    description_cell = cells[0]
                    link_cell = cells[0] # Often the description and link are in the same cell
                    
                    description_text = description_cell.get_text(strip=True)
                    link_tag = link_cell.find('a')

                    # We are interested in the "Original Paper", not Keys or Quizzes
                    if link_tag and "paper" in description_text.lower():
                        href = link_tag.get('href')
                        
                        # Sanitize the description to create a clean name
                        session_text = description_text.replace("Question Paper", "").replace("Paper", "").strip()
                        session_clean = f"_{session_text.replace(' ', '')}" if session_text else ""

                        clean_name = f"GATE_{current_year}_{current_paper_type}{session_clean}"
                        
                        # Avoid duplicates if the table structure is tricky
                        if clean_name not in paper_links:
                             paper_links[clean_name] = href
                             print(f"  -> Found Link: {clean_name} -> {href}")
        
        print(f"\n--- Stage 1 Complete: Collected {len(paper_links)} relevant links. ---\n")
        return paper_links

    except requests.exceptions.RequestException as e:
        print(f"  -> FATAL ERROR in Stage 1: Could not fetch link collector URL: {e}")
        return {}

# ==============================================================================
# STAGE 2: PAPER PARSER
# ==============================================================================
class GFGParser:
    """
    A dedicated scraper for GeeksForGeeks GATE question paper pages.
    """
    def __init__(self):
        self.headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36'}

    def get_soup(self, url):
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return BeautifulSoup(response.text, 'html.parser')
        except requests.exceptions.RequestException as e:
            print(f"  -> ERROR: Could not fetch paper URL {url}: {e}")
            return None

    def parse_paper(self, soup):
        """The core parsing logic for a single GFG paper page."""
        scraped_questions = []
        
        # On GFG, the main content is usually in a <div class='entry-content'>
        content_div = soup.find('div', class_='entry-content')
        if not content_div:
            print("  -> ERROR: Could not find main content div ('entry-content').")
            return []

        # Questions are typically preceded by "Question X:" or "Ques X." in <strong> tags
        # We find all elements and then process them sequentially
        all_elements = content_div.find_all(['p', 'div', 'pre'])
        
        current_question = None
        
        for element in all_elements:
            element_text = element.get_text(strip=True)
            
            # Heuristic to detect the start of a new question
            # Matches "Question 1:", "Ques. 1.", etc.
            if re.match(r'^(Question|Ques)\s*\d+[:.]', element_text, re.IGNORECASE):
                # If we were processing a previous question, save it first
                if current_question:
                    scraped_questions.append(current_question)
                
                # Start a new question object
                current_question = {
                    "question_text": element_text,
                    "question_images": [],
                    "options": [],
                    "answer": "",
                    "explanation": ""
                }
            # If we are inside a question, collect its parts
            elif current_question:
                # Check for images in the question
                for img in element.find_all('img'):
                    if img.get('src'):
                        current_question["question_images"].append(img['src'])

                # Heuristic for detecting options (A), (B), 1., 2. etc.
                if re.match(r'^\(\w\)|^\w\.', element_text):
                    current_question["options"].append(element_text)
                
                # Heuristic for detecting the Answer
                elif element_text.lower().startswith('answer:'):
                    current_question["answer"] = element_text.replace('Answer:', '').strip()

                # Heuristic for detecting the Explanation
                elif element_text.lower().startswith('explanation:'):
                    current_question["explanation"] = element.get_text(separator='\n').replace('Explanation:', '').strip()
                
                # If it's not an option/answer, it's part of the question text
                elif not current_question["options"]:
                    current_question["question_text"] += "\n" + element_text

        # Append the very last question found
        if current_question:
            scraped_questions.append(current_question)

        print(f"  -> Parsed {len(scraped_questions)} questions from this page.")
        return scraped_questions

# ==============================================================================
# ORCHESTRATOR
# ==============================================================================
def save_data(data, name, output_dir):
    """Saves the scraped data into a descriptive JSON file."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    file_path = os.path.join(output_dir, f"{name}_raw.json")
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"  -> Successfully saved raw data to {file_path}")

def main():
    """
    Main orchestrator for the two-stage scraping process.
    """
    print("===================================================")
    print(" GATE-ASTRA: INITIATING GFG SCRAPING ENGINE (DAY 3)")
    print("===================================================")
    
    # STAGE 1
    paper_links = collect_paper_links(LINK_COLLECTOR_URL)
    
    if not paper_links:
        print("Halting process as no paper links were collected.")
        return

    # STAGE 2
    print("\n--- Stage 2: Parsing individual paper pages ---")
    parser = GFGParser()
    
    for name, url in paper_links.items():
        print(f"\nProcessing: {name}")
        
        soup = parser.get_soup(url)
        
        if soup:
            extracted_data = parser.parse_paper(soup)
            if extracted_data:
                save_data(extracted_data, name, OUTPUT_DIR)
            else:
                print(f"  -> WARNING: No data was parsed for {name}. The page structure might be different.")
        
        print(f"Politely waiting for {REQUEST_DELAY_S} seconds...")
        time.sleep(REQUEST_DELAY_S)
            
    print("\n===================================================")
    print(" ALL SCRAPING TASKS COMPLETED.")
    print(f" Raw data saved in '{OUTPUT_DIR}' directory.")
    print("===================================================")

if __name__ == "__main__":
    main()