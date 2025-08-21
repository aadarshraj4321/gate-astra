import os
from sqlalchemy import (create_engine, Column, Integer, String, Float,
                          Boolean, Text, ForeignKey, ARRAY)
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func
from dotenv import load_dotenv

Base = declarative_base()

class Exam(Base):
    __tablename__ = 'exams'
    exam_id = Column(Integer, primary_key=True)
    exam_year = Column(Integer, unique=True, nullable=False)
    organizing_iit = Column(String(50), nullable=False)
    
    # Relationship: Ek Exam mein multiple Questions ho sakte hain
    questions = relationship("Question", back_populates="exam")

    def __repr__(self):
        return f"<Exam(year={self.exam_year}, iit='{self.organizing_iit}')>"

class Syllabus(Base):
    __tablename__ = 'syllabus'
    topic_id = Column(Integer, primary_key=True)
    subject = Column(String(100), nullable=False)
    topic = Column(String(255), nullable=False)
    sub_topic = Column(String(255))

    questions = relationship("Question", back_populates="syllabus_topic")

    def __repr__(self):
        return f"<Syllabus(subject='{self.subject}', topic='{self.topic}')>"

class Question(Base):
    __tablename__ = 'questions'
    question_id = Column(Integer, primary_key=True)
    question_text = Column(Text, nullable=False)
    question_type = Column(String(10), nullable=False) # MCQ, MSQ, NAT
    marks = Column(Integer, nullable=False)
    difficulty_score = Column(Float)
    
    exam_id = Column(Integer, ForeignKey('exams.exam_id'))
    topic_id = Column(Integer, ForeignKey('syllabus.topic_id'))

    exam = relationship("Exam", back_populates="questions")
    syllabus_topic = relationship("Syllabus", back_populates="questions")
    options = relationship("Option", back_populates="question", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Question(id={self.question_id}, marks={self.marks})>"

class Option(Base):
    __tablename__ = 'options'
    option_id = Column(Integer, primary_key=True)
    option_text = Column(Text, nullable=False)
    is_correct = Column(Boolean, nullable=False)
    
    question_id = Column(Integer, ForeignKey('questions.question_id'))
    question = relationship("Question", back_populates="options")

    def __repr__(self):
        return f"<Option(text='{self.option_text[:20]}...', correct={self.is_correct})>"

class Professor(Base):
    __tablename__ = 'professors'
    professor_id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    current_iit = Column(String(50), nullable=False)
    research_interests = Column(ARRAY(String))
    
    publications = relationship("Publication", back_populates="author")

    def __repr__(self):
        return f"<Professor(name='{self.name}', iit='{self.current_iit}')>"

class Publication(Base):
    __tablename__ = 'publications'
    publication_id = Column(Integer, primary_key=True)
    title = Column(Text, nullable=False)
    abstract = Column(Text)
    publication_year = Column(Integer)
    
    professor_id = Column(Integer, ForeignKey('professors.professor_id'))
    author = relationship("Professor", back_populates="publications")

    def __repr__(self):
        return f"<Publication(title='{self.title[:30]}...')>"


# database connection and table creation
def get_db_url():
    load_dotenv()
    return os.getenv("DATABASE_URL")

def setup_database():
    engine = create_engine(get_db_url())
    print("Connecting to the database...")
    Base.metadata.create_all(engine)
    print("Database tables are ready!")

if __name__ == "__main__":
    setup_database()