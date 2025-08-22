# FILE NAME: src/data_ingestion/ingest_exam_metadata.py
# VERSION: 4.0 (Final, Corrected Keys, and Comprehensive List)

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
import sys

# Allow importing from the parent directory (src)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models import Exam, get_db_url

# This is the single source of truth for exam history.
# The keys in this list of dictionaries (`exam_year`, `paper_subject`, etc.)
# now EXACTLY MATCH the column names in the `Exam` model in `src/models.py`.
EXAM_HISTORY = [
    {"exam_year": 2025, "paper_subject": "CS", "paper_set": 1, "organizing_iit": "IIT Roorkee"},
    {"exam_year": 2025, "paper_subject": "CS", "paper_set": 2, "organizing_iit": "IIT Roorkee"},
    {"exam_year": 2025, "paper_subject": "DA", "paper_set": 1, "organizing_iit": "IIT Roorkee"},

    {"exam_year": 2024, "paper_subject": "CS", "paper_set": 1, "organizing_iit": "IISc Bangalore"},
    {"exam_year": 2024, "paper_subject": "CS", "paper_set": 2, "organizing_iit": "IISc Bangalore"},
    {"exam_year": 2024, "paper_subject": "DA", "paper_set": 1, "organizing_iit": "IISc Bangalore"},

    {"exam_year": 2023, "paper_subject": "CS", "paper_set": 1, "organizing_iit": "IIT Kanpur"},
    {"exam_year": 2023, "paper_subject": "CS", "paper_set": 2, "organizing_iit": "IIT Kanpur"},

    {"exam_year": 2022, "paper_subject": "CS", "paper_set": 1, "organizing_iit": "IIT Kharagpur"},
    {"exam_year": 2022, "paper_subject": "CS", "paper_set": 2, "organizing_iit": "IIT Kharagpur"},

    {"exam_year": 2021, "paper_subject": "CS", "paper_set": 1, "organizing_iit": "IIT Bombay"},
    {"exam_year": 2021, "paper_subject": "CS", "paper_set": 2, "organizing_iit": "IIT Bombay"},

    {"exam_year": 2020, "paper_subject": "CS", "paper_set": 1, "organizing_iit": "IIT Delhi"},
    {"exam_year": 2020, "paper_subject": "CS", "paper_set": 2, "organizing_iit": "IIT Delhi"},

    {"exam_year": 2019, "paper_subject": "CS", "paper_set": 1, "organizing_iit": "IIT Madras"},
    {"exam_year": 2019, "paper_subject": "CS", "paper_set": 2, "organizing_iit": "IIT Madras"},
    
    {"exam_year": 2018, "paper_subject": "CS", "paper_set": 1, "organizing_iit": "IIT Guwahati"},
    {"exam_year": 2018, "paper_subject": "CS", "paper_set": 2, "organizing_iit": "IIT Guwahati"},

    {"exam_year": 2017, "paper_subject": "CS", "paper_set": 1, "organizing_iit": "IIT Roorkee"},
    {"exam_year": 2017, "paper_subject": "CS", "paper_set": 2, "organizing_iit": "IIT Roorkee"},
    
    {"exam_year": 2016, "paper_subject": "CS", "paper_set": 1, "organizing_iit": "IISc Bangalore"},
    {"exam_year": 2016, "paper_subject": "CS", "paper_set": 2, "organizing_iit": "IISc Bangalore"},

    {"exam_year": 2015, "paper_subject": "CS", "paper_set": 1, "organizing_iit": "IIT Kanpur"},
    {"exam_year": 2015, "paper_subject": "CS", "paper_set": 2, "organizing_iit": "IIT Kanpur"},
    {"exam_year": 2015, "paper_subject": "CS", "paper_set": 3, "organizing_iit": "IIT Kanpur"},

    {"exam_year": 2014, "paper_subject": "CS", "paper_set": 1, "organizing_iit": "IIT Kharagpur"},
    {"exam_year": 2014, "paper_subject": "CS", "paper_set": 2, "organizing_iit": "IIT Kharagpur"},
    {"exam_year": 2014, "paper_subject": "CS", "paper_set": 3, "organizing_iit": "IIT Kharagpur"},

    {"exam_year": 2013, "paper_subject": "CS", "paper_set": 1, "organizing_iit": "IIT Bombay"},
    {"exam_year": 2013, "paper_subject": "CS", "paper_set": 2, "organizing_iit": "IIT Bombay"},

    {"exam_year": 2012, "paper_subject": "CS", "paper_set": 1, "organizing_iit": "IIT Delhi"},
    {"exam_year": 2012, "paper_subject": "CS", "paper_set": 2, "organizing_iit": "IIT Delhi"},

    {"exam_year": 2011, "paper_subject": "CS", "paper_set": 1, "organizing_iit": "IIT Madras"},
    {"exam_year": 2011, "paper_subject": "CS", "paper_set": 2, "organizing_iit": "IIT Madras"},
]

def populate_exams():
    """
    This function populates the Exams table.
    It checks for existing entries to avoid creating duplicates.
    """
    engine = create_engine(get_db_url())
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        print("Populating the 'Exams' table with comprehensive history...")
        
        # Create a set of existing exams for a quick, efficient check
        existing_exams = {(e.exam_year, e.paper_subject, e.paper_set) for e in session.query(Exam).all()}
        
        # Create a list of new Exam objects that are not already in the database
        new_exams_to_add = [
            Exam(**item) for item in EXAM_HISTORY 
            if (item["exam_year"], item["paper_subject"], item["paper_set"]) not in existing_exams
        ]
        
        if new_exams_to_add:
            session.bulk_save_objects(new_exams_to_add)
            session.commit()
            print(f"✅ Added {len(new_exams_to_add)} new exam records to the database.")
        else:
            print("✅ 'Exams' table is already up to date.")
            
    except Exception as e:
        print(f"An error occurred while populating exams: {e}")
        session.rollback()
    finally:
        session.close()

if __name__ == "__main__":
    populate_exams()