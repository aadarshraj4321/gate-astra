from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
import sys

# Allow importing from the parent directory (src)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models import Syllabus, get_db_url
# Import the data from our new, clean data file
from .syllabus_data import ALL_SYLLABUS_DATA

def populate_syllabus():
    """
    This function robustly populates the Syllabus table.
    It fetches all existing syllabus entries from the DB and only inserts
    new ones, preventing duplicates. This makes the script idempotent.
    """
    engine = create_engine(get_db_url())
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        print("Fetching existing syllabus entries from the database...")
        # Create a set of tuples for quick lookup of existing entries
        existing_entries = {
            (s.subject, s.topic, s.sub_topic) 
            for s in session.query(Syllabus).all()
        }
        print(f"Found {len(existing_entries)} existing entries.")

        new_syllabus_objects = []
        for item in ALL_SYLLABUS_DATA:
            # Create a tuple for the current item to check for its existence
            entry_tuple = (item["subject"], item["topic"], item.get("sub_topic"))
            
            if entry_tuple not in existing_entries:
                new_entry = Syllabus(
                    subject=item["subject"],
                    topic=item["topic"],
                    sub_topic=item.get("sub_topic")
                )
                new_syllabus_objects.append(new_entry)
        
        if not new_syllabus_objects:
            print("Syllabus is already up to date. No new entries to add.")
            return

        print(f"Adding {len(new_syllabus_objects)} new syllabus topics to the database...")
        session.bulk_save_objects(new_syllabus_objects)
        session.commit()
        print("Successfully updated the syllabus table.")

    except Exception as e:
        print(f"An error occurred: {e}")
        session.rollback()
    finally:
        session.close()

if __name__ == "__main__":
    populate_syllabus()