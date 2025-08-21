from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models import Exam, get_db_url

EXAM_HISTORY = [
    {"year": 2024, "iit": "IISc Bangalore"}, {"year": 2023, "iit": "IIT Kanpur"},
    {"year": 2022, "iit": "IIT Kharagpur"}, {"year": 2021, "iit": "IIT Bombay"},
    {"year": 2020, "iit": "IIT Delhi"}, {"year": 2019, "iit": "IIT Madras"},
    {"year": 2018, "iit": "IIT Guwahati"}, {"year": 2017, "iit": "IIT Roorkee"},
    {"year": 2016, "iit": "IISc Bangalore"}, {"year": 2015, "iit": "IIT Kanpur"},
    {"year": 2014, "iit": "IIT Kharagpur"}, {"year": 2013, "iit": "IIT Bombay"},
    {"year": 2012, "iit": "IIT Delhi"}, {"year": 2011, "iit": "IIT Madras"},
    {"year": 2010, "iit": "IIT Guwahati"},
]

def populate_exams():
    """
    Robustly populates the Exams table, checking for existing years
    to avoid duplicates. This script is idempotent.
    """
    engine = create_engine(get_db_url())
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        print("Fetching existing exam years...")
        existing_years = {exam.exam_year for exam in session.query(Exam.exam_year).all()}
        print(f"Found {len(existing_years)} existing exam records.")
        
        new_exams = []
        for item in EXAM_HISTORY:
            if item["year"] not in existing_years:
                new_exams.append(Exam(exam_year=item["year"], organizing_iit=item["iit"]))

        if not new_exams:
            print("Exam metadata is already up to date. No new records to add.")
            return

        print(f"Adding {len(new_exams)} new exam records...")
        session.bulk_save_objects(new_exams)
        session.commit()
        print("Successfully updated exam history.")

    except Exception as e:
        print(f"An error occurred: {e}")
        session.rollback()
    finally:
        session.close()

if __name__ == "__main__":
    populate_exams()