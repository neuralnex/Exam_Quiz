import logging
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from src.config import settings
from src.database import get_db_session
from src.models.entities import ExamModel, QuestionModel, ExamAttempt, Answer, TopicPerformance

logger = logging.getLogger(__name__)


class ExamEngine:
    """
    Manages the lifecycle of an exam attempt: start, submit, score, and analyze.
    """

    def start_exam(self, exam_id: str, student_id: str = "Student") -> Optional[Dict[str, Any]]:
        """Starts a new exam attempt."""
        with get_db_session() as session:
            exam = session.query(ExamModel).filter(ExamModel.exam_id == exam_id).first()
            if not exam:
                return None

            attempt_id_str = f"ATT_{uuid.uuid4().hex[:8].upper()}"
            attempt = ExamAttempt(
                attempt_id=attempt_id_str,
                exam_id=exam.id,
                course_id=exam.course_id,
                student_identifier=student_id,
                start_time=datetime.utcnow(),
                end_time_scheduled=datetime.utcnow() + timedelta(minutes=exam.duration_minutes),
                status="in_progress"
            )
            session.add(attempt)
            session.flush()

            questions = session.query(QuestionModel).filter(
                QuestionModel.exam_id == exam.id
            ).order_by(QuestionModel.order_index).all()

            ui_questions = []
            for q in questions:
                ui_questions.append({
                    "uid": q.question_uid,
                    "text": q.question_text,
                    "options": {
                        "A": q.option_a,
                        "B": q.option_b,
                        "C": q.option_c,
                        "D": q.option_d
                    },
                    "topic": q.topic,
                    "difficulty": q.difficulty
                })

            return {
                "attempt_id": attempt_id_str,
                "exam_id": exam_id,
                "exam_title": exam.title,
                "duration": exam.duration_minutes,
                "questions": ui_questions
            }

    def submit_exam(self, attempt_id_str: str, student_answers: Dict[str, Optional[str]]) -> Optional[Dict[str, Any]]:
        """
        Submits answers, calculates score, and updates attempt status.
        student_answers: { "question_uid": "A/B/C/D" }
        """
        with get_db_session() as session:
            attempt = session.query(ExamAttempt).filter(ExamAttempt.attempt_id == attempt_id_str).first()
            if not attempt:
                return None

            exam = session.query(ExamModel).filter(ExamModel.id == attempt.exam_id).first()
            if not exam:
                return None

            questions = session.query(QuestionModel).filter(QuestionModel.exam_id == exam.id).all()

            topic_stats = {}
            review_questions = []

            for q in questions:
                question_uid = getattr(q, "question_uid", None)
                marks = getattr(q, "marks", 1) or 1
                selected = student_answers.get(question_uid)
                is_correct = (selected == q.correct_answer)
                options = {
                    "A": q.option_a,
                    "B": q.option_b,
                    "C": q.option_c,
                    "D": q.option_d,
                }

                ans = Answer(
                    attempt_id=attempt.id,
                    question_uid=question_uid,
                    selected_option=selected,
                    is_correct=is_correct
                )
                session.add(ans)

                if q.topic not in topic_stats:
                    topic_stats[q.topic] = {"total": 0, "correct": 0}
                topic_stats[q.topic]["total"] += 1
                if is_correct:
                    topic_stats[q.topic]["correct"] += 1

                review_questions.append({
                    "uid": question_uid,
                    "text": q.question_text,
                    "options": options,
                    "selected_answer": selected,
                    "correct_answer": q.correct_answer,
                    "is_correct": is_correct,
                    "explanation": q.explanation,
                    "topic": q.topic,
                    "difficulty": q.difficulty,
                    "marks": marks,
                })

            total_marks = sum(getattr(q, "marks", 1) or 1 for q in questions)
            actual_score = sum(
                getattr(q, "marks", 1) or 1
                for q in questions
                if student_answers.get(getattr(q, "question_uid", None)) == q.correct_answer
            )

            percentage = (actual_score / total_marks * 100) if total_marks > 0 else 0

            attempt.submitted_at = datetime.utcnow()
            attempt.score = actual_score
            attempt.total_marks = total_marks
            attempt.percentage = percentage
            attempt.is_passed = percentage >= settings.PASSING_PERCENTAGE
            attempt.status = "completed"

            for topic, stats in topic_stats.items():
                perf = TopicPerformance(
                    attempt_id=attempt.id,
                    course_id=attempt.course_id,
                    topic_name=topic,
                    total_questions=stats["total"],
                    correct_questions=stats["correct"],
                    percentage=(stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
                )
                session.add(perf)

            return {
                "score": actual_score,
                "total": total_marks,
                "percentage": percentage,
                "is_passed": attempt.is_passed,
                "topic_analysis": topic_stats,
                "questions": review_questions,
            }
