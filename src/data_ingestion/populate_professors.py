import os
import sys
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import Professor, get_db_url


PROFESSOR_DATA = [
    {"name": "Prof. Sharat Chandran", "current_iit": "IIT Bombay", "research_interests": ["Databases", "Image Processing", "Computer Graphics"]},
    {"name": "Prof. S. Sudarshan", "current_iit": "IIT Bombay", "research_interests": ["Database Systems", "Query Processing", "Information Retrieval"]},
    {"name": "Prof. Mausam", "current_iit": "IIT Delhi", "research_interests": ["Artificial Intelligence", "Natural Language Processing", "Machine Learning"]},
    {"name": "Prof. S. Arun Kumar", "current_iit": "IIT Delhi", "research_interests": ["Theory of Computation", "Programming Languages", "Formal Methods"]},
    {"name": "Prof. Partha Pratim Chakrabarti", "current_iit": "IIT Kharagpur", "research_interests": ["Artificial Intelligence", "Algorithms", "CAD for VLSI"]},
    {"name": "Prof. V. Kamakoti", "current_iit": "IIT Madras", "research_interests": ["Computer Architecture", "Information Security", "VLSI Design"]},
    {"name": "Prof. Hema A. Murthy", "current_iit": "IIT Madras", "research_interests": ["Speech Processing", "Signal Processing", "Machine Learning"]},
    {"name": "Prof. Manindra Agrawal", "current_iit": "IIT Kanpur", "research_interests": ["Complexity Theory", "Cryptography", "Algorithms"]},
]

def populate_professors():
    engine = create_engine(get_db_url())
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        print("Populating the 'Professors' table...")
        existing_profs = {p.name for p in session.query(Professor).all()}
        new_profs = [Professor(**item) for item in PROFESSOR_DATA if item['name'] not in existing_profs]
        
        if new_profs:
            session.bulk_save_objects(new_profs)
            session.commit()
            print(f"Added {len(new_profs)} new professor records.")
        else:
            print("'Professors' table is already up to date.")
    finally:
        session.close()

if __name__ == "__main__":
    populate_professors()