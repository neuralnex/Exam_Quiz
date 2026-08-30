from datetime import datetime, timedelta
from typing import List, Optional
from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    Float,
    Boolean,
    DateTime,
    ForeignKey,
    JSON,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Course(Base):
    __tablename__ = "courses"

    id = Column(Integer, primary_key=True, autoincrement=True)
    code = Column(String(50), unique=True, nullable=False, index=True)  # e.g., SOE_510
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    documents = relationship("Document", back_populates="course", cascade="all, delete-orphan")
    topics = relationship("Topic", back_populates="course", cascade="all, delete-orphan")
    exams = relationship("ExamModel", back_populates="course", cascade="all, delete-orphan")
    attempts = relationship("ExamAttempt", back_populates="course", cascade="all, delete-orphan")


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    course_id = Column(Integer, ForeignKey("courses.id", ondelete="CASCADE"), nullable=False, index=True)
    filename = Column(String(255), nullable=False)
    file_type = Column(String(20), nullable=False)  # pdf, pptx, png, jpg, etc.
    file_path = Column(String(500), nullable=False)
    file_size_bytes = Column(Integer, default=0)
    chunks_count = Column(Integer, default=0)
    is_indexed = Column(Boolean, default=False)
    indexed_at = Column(DateTime, nullable=True)
    metadata_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    course = relationship("Course", back_populates="documents")

    __table_args__ = (
        UniqueConstraint("course_id", "filename", name="uq_course_document"),
    )


class Topic(Base):
    __tablename__ = "topics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    course_id = Column(Integer, ForeignKey("courses.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    course = relationship("Course", back_populates="topics")

    __table_args__ = (
        UniqueConstraint("course_id", "name", name="uq_course_topic"),
    )


class ExamModel(Base):
    __tablename__ = "exams"

    id = Column(Integer, primary_key=True, autoincrement=True)
    exam_id = Column(String(100), unique=True, nullable=False, index=True)
    course_id = Column(Integer, ForeignKey("courses.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    duration_minutes = Column(Integer, nullable=False, default=45)
    total_marks = Column(Integer, nullable=False, default=0)
    difficulty = Column(String(50), default="Mixed")  # Easy, Medium, Hard, Mixed
    question_type = Column(String(50), default="Multiple Choice")
    is_practice = Column(Boolean, default=False)
    config_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    course = relationship("Course", back_populates="exams")
    questions = relationship("QuestionModel", back_populates="exam", cascade="all, delete-orphan")
    attempts = relationship("ExamAttempt", back_populates="exam", cascade="all, delete-orphan")


class QuestionModel(Base):
    __tablename__ = "questions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    exam_id = Column(Integer, ForeignKey("exams.id", ondelete="CASCADE"), nullable=False, index=True)
    question_uid = Column(String(100), nullable=False, index=True)
    question_text = Column(Text, nullable=False)
    option_a = Column(Text, nullable=False)
    option_b = Column(Text, nullable=False)
    option_c = Column(Text, nullable=False)
    option_d = Column(Text, nullable=False)
    correct_answer = Column(String(10), nullable=False)  # "A", "B", "C", or "D"
    explanation = Column(Text, nullable=False)
    source = Column(String(500), nullable=False)
    topic = Column(String(255), nullable=False)
    difficulty = Column(String(50), default="Medium")
    marks = Column(Integer, default=1)
    order_index = Column(Integer, default=0)

    # Relationships
    exam = relationship("ExamModel", back_populates="questions")


class ExamAttempt(Base):
    __tablename__ = "exam_attempts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    attempt_id = Column(String(100), unique=True, nullable=False, index=True)
    exam_id = Column(Integer, ForeignKey("exams.id", ondelete="CASCADE"), nullable=False, index=True)
    course_id = Column(Integer, ForeignKey("courses.id", ondelete="CASCADE"), nullable=False, index=True)
    student_identifier = Column(String(100), default="Student")
    start_time = Column(DateTime, nullable=False, default=datetime.utcnow)
    end_time_scheduled = Column(
        DateTime,
        nullable=False,
        default=lambda: datetime.utcnow() + timedelta(minutes=45),
    )
    submitted_at = Column(DateTime, nullable=True)
    time_spent_seconds = Column(Integer, default=0)
    duration_seconds = Column(Integer, default=0)
    score = Column(Integer, default=0)
    total_marks = Column(Integer, default=0)
    percentage = Column(Float, default=0.0)
    is_passed = Column(Boolean, default=False)
    is_auto_submitted = Column(Boolean, default=False)
    status = Column(String(50), default="in_progress")  # in_progress, completed, expired
    ai_insights = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    exam = relationship("ExamModel", back_populates="attempts")
    course = relationship("Course", back_populates="attempts")
    answers = relationship("Answer", back_populates="attempt", cascade="all, delete-orphan")
    topic_performances = relationship("TopicPerformance", back_populates="attempt", cascade="all, delete-orphan")


class Answer(Base):
    __tablename__ = "answers"

    id = Column(Integer, primary_key=True, autoincrement=True)
    attempt_id = Column(Integer, ForeignKey("exam_attempts.id", ondelete="CASCADE"), nullable=False, index=True)
    question_uid = Column(String(100), nullable=False)
    selected_option = Column(String(10), nullable=True)  # "A", "B", "C", "D", or None
    is_correct = Column(Boolean, default=False)
    is_marked_review = Column(Boolean, default=False)
    time_spent_seconds = Column(Integer, default=0)

    # Relationships
    attempt = relationship("ExamAttempt", back_populates="answers")


class TopicPerformance(Base):
    __tablename__ = "topic_performances"

    id = Column(Integer, primary_key=True, autoincrement=True)
    attempt_id = Column(Integer, ForeignKey("exam_attempts.id", ondelete="CASCADE"), nullable=False, index=True)
    course_id = Column(Integer, ForeignKey("courses.id", ondelete="CASCADE"), nullable=True, index=True)
    topic_name = Column(String(255), nullable=False)
    total_questions = Column(Integer, default=0)
    correct_questions = Column(Integer, default=0)
    percentage = Column(Float, default=0.0)

    # Relationships
    attempt = relationship("ExamAttempt", back_populates="topic_performances")
