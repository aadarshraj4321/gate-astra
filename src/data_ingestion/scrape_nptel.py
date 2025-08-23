import json
import os
import time
import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup


PORTAL_URL = "https://nptel.ac.in/courses"
DISCIPLINE_LINK_TEXT = "Computer Science and Engineering"
BASE_URL = "https://nptel.ac.in"

OUTPUT_DIR = "nptel_data"
REQUEST_DELAY_S = 1 # We can be a bit faster as we are loading pages fully


class NptelScraper:
    def __init__(self):
        # Setup Selenium WebDriver
        print("Initializing Selenium WebDriver...")
        options = webdriver.ChromeOptions()
        options.add_argument('--headless') # Run in headless mode (no browser window opens)
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        try:
            # webdriver-manager will automatically download and manage the correct driver
            self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
            print("WebDriver initialized successfully.")
        except Exception as e:
            print(f"FATAL ERROR: Could not initialize Selenium WebDriver. Please ensure Chrome is installed.")
            print(f"   Error details: {e}")
            self.driver = None

    def get_dynamic_soup(self, url, wait_for_element=(By.TAG_NAME, 'body')):
        if not self.driver: return None
        try:
            # print(f"Fetching dynamic URL: {url}")
            self.driver.get(url)
            # Wait for a specific element to be present, this ensures JS has loaded
            WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located(wait_for_element)
            )
            return BeautifulSoup(self.driver.page_source, 'html.parser')
        except Exception as e:
            print(f"  -> Error fetching dynamic URL {url}: {e}")
            return None
            
    def discover_course_links(self):
        print("\n--- STAGE 1: Discovering all CS course links (using Selenium) ---")
        
        # Go to the main portal and wait for the course category links to appear
        portal_soup = self.get_dynamic_soup(PORTAL_URL, wait_for_element=(By.CLASS_NAME, 'cat-container'))
        if not portal_soup:
            print("  -> FATAL: Could not fetch the NPTEL portal. Halting.")
            return []
            
        cs_link_tag = portal_soup.find('a', string=re.compile(DISCIPLINE_LINK_TEXT, re.IGNORECASE))
        if not cs_link_tag or not cs_link_tag.get('href'):
            print(f"  -> FATAL: Could not find the '{DISCIPLINE_LINK_TEXT}' link on the portal page.")
            return []
        
        cs_discipline_url = BASE_URL + cs_link_tag['href']
        print(f"Found CS discipline page: {cs_discipline_url}")

        # Go to the CS discipline page and wait for the course links to load
        course_list_soup = self.get_dynamic_soup(cs_discipline_url, wait_for_element=(By.CLASS_NAME, 'course-link'))
        if not course_list_soup:
            print("  -> FATAL: Could not fetch the CS discipline page. Halting.")
            return []

        course_links = [tag.get('href') for tag in course_list_soup.find_all('a', class_='course-link') if tag.get('href')]
        unique_course_links = sorted(list(set(course_links)))
        
        print(f"Discovered {len(unique_course_links)} unique course links.")
        return unique_course_links

    def parse_course_page(self, url):
        """
        Stage 2: Visits a single course page and extracts detailed information.
        """
        print(f"\n--- STAGE 2: Parsing course details from {url} ---")
        soup = self.get_dynamic_soup(url, wait_for_element=(By.ID, 'module-wrapper'))
        if not soup:
            print("  -> Skipping course due to fetch error.")
            return None

        # (The parsing logic from before is still valid, as it works on the final HTML)
        course_data = {"course_url": url, "course_title": "N/A", "professors": [], "iit": "N/A", "lecture_list": []}
        header_div = soup.find('div', class_='course-header')
        if header_div:
            title_tag = header_div.find('h1')
            if title_tag: course_data['course_title'] = title_tag.get_text(strip=True)
            prof_iit_div = header_div.find('h3')
            if prof_iit_div:
                parts = prof_iit_div.get_text(strip=True).split('|')
                if len(parts) == 2:
                    course_data['professors'], course_data['iit'] = [parts[0].strip()], parts[1].strip()
                else:
                    course_data['professors'] = [prof_iit_div.get_text(strip=True)]

        lecture_accordion = soup.find('div', id='module-wrapper')
        if lecture_accordion:
            for tag in lecture_accordion.find_all('a'):
                lecture_title = tag.get_text(strip=True)
                if lecture_title:
                    cleaned_title = re.sub(r'^(Lecture|Lec)\s*\d*\s*[:.-]*\s*', '', lecture_title, flags=re.IGNORECASE).strip()
                    if cleaned_title: course_data['lecture_list'].append(cleaned_title)

        print(f"  -> Parsed '{course_data['course_title']}' by {course_data['iit']}")
        return course_data

    def save_course_data(self, data, output_dir):
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        safe_filename = re.sub(r'[\\/*?:"<>|]', "", data['course_title'])[:100] + ".json"
        file_path = os.path.join(output_dir, safe_filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"Saved data to '{file_path}'")

    def close(self):
        if self.driver:
            self.driver.quit()
            print("\nWebDriver session closed.")

def main():
    print("======================================================")
    print(" GATE-ASTRA: NPTEL PULSE SCRAPER (DAY 15 - V3 Selenium)")
    print("======================================================")
    
    scraper = NptelScraper()
    if not scraper.driver:
        return

    try:
        course_urls = scraper.discover_course_links()
        if not course_urls:
            print("\nHalting script as no course URLs were found.")
            return

        for i, url in enumerate(course_urls):
            print(f"\nProcessing course {i+1}/{len(course_urls)}...")
            course_details = scraper.parse_course_page(url)
            if course_details and course_details['lecture_list']:
                scraper.save_course_data(course_details, OUTPUT_DIR)
            else:
                print("  -> Skipping save: Incomplete details or no lecture list found.")
            time.sleep(REQUEST_DELAY_S)
    finally:
        scraper.close()
        
    print("\n======================================================")
    print(" NPTEL SCRAPING COMPLETE.")
    print(f" All data saved in the '{OUTPUT_DIR}' directory.")
    print("======================================================")

if __name__ == "__main__":
    main()