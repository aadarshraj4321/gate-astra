# FILE NAME: src/models.py

import os
from sqlalchemy import (create_engine, Column, Integer, String, Float,
                          Boolean, Text, ForeignKey, ARRAY, UniqueConstraint)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from dotenv import load_dotenv

Base = declarative_base()

class Exam(Base):
    __tablename__ = 'exams'
    exam_id = Column(Integer, primary_key=True)
    exam_year = Column(Integer, nullable=False)
    paper_subject = Column(String(10), nullable=False)
    paper_set = Column(Integer, default=1, nullable=False)
    organizing_iit = Column(String(50), nullable=False)
    __table_args__ = (UniqueConstraint('exam_year', 'paper_subject', 'paper_set', name='_exam_uc'),)
    questions = relationship("Question", back_populates="exam")
    def __repr__(self): return f"<Exam(year={self.exam_year}, subject='{self.paper_subject}', set={self.paper_set})>"

class Syllabus(Base):
    __tablename__ = 'syllabus'
    topic_id = Column(Integer, primary_key=True)
    subject = Column(String(255), nullable=False)
    topic = Column(String(255), nullable=False)
    sub_topic = Column(String(255))
    questions = relationship("Question", back_populates="syllabus_topic")
    def __repr__(self): return f"<Syllabus(subject='{self.subject}', topic='{self.topic}')>"

class Question(Base):
    __tablename__ = 'questions'
    question_id = Column(Integer, primary_key=True)
    question_text = Column(Text, nullable=False)
    question_type = Column(String(10), nullable=False)
    marks = Column(Integer, nullable=False)
    exam_id = Column(Integer, ForeignKey('exams.exam_id'))
    topic_id = Column(Integer, ForeignKey('syllabus.topic_id'))
    exam = relationship("Exam", back_populates="questions")
    syllabus_topic = relationship("Syllabus", back_populates="questions")
    options = relationship("Option", back_populates="question", cascade="all, delete-orphan")
    def __repr__(self): return f"<Question(id={self.question_id}, marks={self.marks})>"

class Option(Base):
    __tablename__ = 'options'
    option_id = Column(Integer, primary_key=True)
    option_text = Column(Text, nullable=False)
    is_correct = Column(Boolean, nullable=False)
    question_id = Column(Integer, ForeignKey('questions.question_id'))
    question = relationship("Question", back_populates="options")
    def __repr__(self): return f"<Option(text='{self.option_text[:20]}...', correct={self.is_correct})>"

# Professor and Publication tables are for later, but part of the schema
class Professor(Base):
    __tablename__ = 'professors'
    professor_id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    current_iit = Column(String(50), nullable=False)
    research_interests = Column(ARRAY(String))
    publications = relationship("Publication", back_populates="author")

class Publication(Base):
    __tablename__ = 'publications'
    publication_id = Column(Integer, primary_key=True)
    title = Column(Text, nullable=False)
    abstract = Column(Text)
    publication_year = Column(Integer)
    professor_id = Column(Integer, ForeignKey('professors.professor_id'))
    author = relationship("Professor", back_populates="publications")

def get_db_url():
    load_dotenv()
    db_url = os.getenv("DATABASE_URL")
    if not db_url: raise ValueError("DATABASE_URL not found in .env file.")
    return db_url

def setup_database():
    engine = create_engine(get_db_url())
    print("Connecting to the database to set up schema...")
    # Base.metadata.drop_all(engine) # KEEP THIS COMMENTED unless you want to wipe
    Base.metadata.create_all(engine)
    print("✅ Database schema is up to date!")

if __name__ == "__main__":
    setup_database()