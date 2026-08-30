import logging
import json
import uuid
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, field_validator

from openai import OpenAI

from src.config import settings
from src.database import get_db_session
from src.models.entities import ExamModel, QuestionModel, Course, Topic
from src.vectorstore.pinecone_store import get_vector_store
from src.embeddings.provider import get_embedding_provider

logger = logging.getLogger(__name__)


class MCQ(BaseModel):
    question: str
    option_a: str
    option_b: str
    option_c: str
    option_d: str
    correct_answer: str
    explanation: str
    topic: str
    difficulty: str

    @field_validator("correct_answer")
    @classmethod
    def validate_correct_answer(cls, value: str) -> str:
        if value not in {"A", "B", "C", "D"}:
            raise ValueError("correct_answer must be A, B, C, or D")
        return value


class QuestionGenerator:
    """
    Uses Groq LLM to generate high-quality MCQs based on
    retrieved context from the vector store.
    """

    def __init__(self):
        self.vector_store = get_vector_store()
        self.last_error: Optional[str] = None
        self._init_groq()

    def _init_groq(self):
        self.client = OpenAI(
            api_key=settings.GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
        )
        self.model = settings.GROQ_MODEL

    def _retrieve_context(self, course_code: str, topic: Optional[str] = None, top_k: int = 10) -> str:
        """Retrieves relevant chunks from the vector store to provide context to the LLM."""
        embedder = get_embedding_provider()

        query = f"{course_code} {topic if topic else ''}"
        query_vec = embedder.embed_text(query)

        filter_dict = {"course": course_code}
        if topic:
            filter_dict["topic"] = topic

        results = self.vector_store.search(
            query_vector=query_vec,
            top_k=top_k,
            filter_dict=filter_dict,
            namespace=course_code
        )

        context = "\n\n".join([r["text"] for r in results])
        return context

    def _parse_questions_json(self, content: str) -> List[Dict[str, Any]]:
        """Parse question JSON from a plain or fenced LLM response."""
        content = (content or "").strip()
        if not content:
            raise ValueError("LLM returned an empty response")

        if content.startswith("```"):
            content = content.strip("`").strip()
            if content.lower().startswith("json"):
                content = content[4:].strip()

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            array_start = content.find("[")
            array_end = content.rfind("]")
            if array_start != -1 and array_end != -1 and array_end > array_start:
                data = json.loads(content[array_start:array_end + 1])
            else:
                object_start = content.find("{")
                object_end = content.rfind("}")
                if object_start == -1 or object_end == -1 or object_end <= object_start:
                    raise
                data = json.loads(content[object_start:object_end + 1])

        if isinstance(data, dict):
            for val in data.values():
                if isinstance(val, list):
                    return val
            return [data]

        if isinstance(data, list):
            return data

        raise ValueError("LLM response did not contain question objects")

    def _question_response_schema(self) -> Dict[str, Any]:
        question_schema = {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "option_a": {"type": "string"},
                "option_b": {"type": "string"},
                "option_c": {"type": "string"},
                "option_d": {"type": "string"},
                "correct_answer": {"type": "string", "enum": ["A", "B", "C", "D"]},
                "explanation": {"type": "string"},
                "topic": {"type": "string"},
                "difficulty": {"type": "string"},
            },
            "required": [
                "question",
                "option_a",
                "option_b",
                "option_c",
                "option_d",
                "correct_answer",
                "explanation",
                "topic",
                "difficulty",
            ],
            "additionalProperties": False,
        }
        return {
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "items": question_schema,
                }
            },
            "required": ["questions"],
            "additionalProperties": False,
        }

    def generate_exam(self, course_code: str, title: str, topic: Optional[str] = None, difficulty: str = "Medium", count: int = 15):
        """
        Generates an exam by retrieving context and prompting the LLM.
        """
        self.last_error = None
        context = self._retrieve_context(course_code, topic)
        if not context:
            logger.warning("No context retrieved for question generation.")
            self.last_error = "No indexed course material was found for this course. Index materials first."
            return None

        prompt = f"""
        You are an expert university professor creating a Computer Based Test (CBT) exam.
        Based on the provided course materials, generate {count} high-quality Multiple Choice Questions (MCQs).

        Course: {course_code}
        Topic: {topic if topic else 'General'}
        Difficulty: {difficulty}

        Requirements:
        1. Each question must have exactly 4 options (A, B, C, D).
        2. Only one option must be correct.
        3. Provide a detailed explanation for the correct answer.
        4. Ensure the questions are challenging and test conceptual understanding.
        5. Return only valid JSON, with no markdown or extra text.

        Context from course materials:
        ---
        {context}
        ---

        JSON Format:
        {{
          "questions": [
            {{
              "question": "Question text here?",
              "option_a": "Option A",
              "option_b": "Option B",
              "option_c": "Option C",
              "option_d": "Option D",
              "correct_answer": "A",
              "explanation": "Explanation why A is correct...",
              "topic": "{topic if topic else 'General'}",
              "difficulty": "{difficulty}"
            }}
          ]
        }}
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You create university CBT multiple-choice exams and return only schema-valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "cbt_exam_questions",
                        "strict": True,
                        "schema": self._question_response_schema(),
                    },
                },
            )

            content = response.choices[0].message.content
            questions_data = self._parse_questions_json(content)

            if not questions_data:
                self.last_error = "The model returned no questions."
                logger.error(self.last_error)
                return None

            validated_questions = [MCQ.model_validate(q).model_dump() for q in questions_data]

            with get_db_session() as session:
                course = session.query(Course).filter(Course.code == course_code).first()
                if not course:
                    self.last_error = f"Course {course_code} not found in DB."
                    logger.error(self.last_error)
                    return None

                exam_id_str = f"EXAM_{uuid.uuid4().hex[:8].upper()}"
                exam = ExamModel(
                    exam_id=exam_id_str,
                    course_id=course.id,
                    title=title,
                    duration_minutes=settings.DEFAULT_EXAM_DURATION_MINUTES,
                    difficulty=difficulty,
                    question_type="Multiple Choice"
                )
                session.add(exam)
                session.flush()

                for i, q_data in enumerate(validated_questions[:count]):
                    question = QuestionModel(
                        exam_id=exam.id,
                        question_uid=f"Q_{uuid.uuid4().hex[:8].upper()}",
                        question_text=q_data["question"],
                        option_a=q_data["option_a"],
                        option_b=q_data["option_b"],
                        option_c=q_data["option_c"],
                        option_d=q_data["option_d"],
                        correct_answer=q_data["correct_answer"],
                        explanation=q_data["explanation"],
                        source="AI Generated",
                        topic=q_data.get("topic", topic or "General"),
                        difficulty=q_data.get("difficulty", difficulty),
                        order_index=i
                    )
                    session.add(question)

                logger.info(f"Generated exam {exam_id_str} with {len(validated_questions[:count])} questions.")
                return exam_id_str

        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Error generating exam: {e}")
            return None
