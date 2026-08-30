import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import uuid
from src.config import settings
from src.database import get_db_session, init_db

st.set_page_config(
    page_title=settings.APP_NAME,
    page_icon="🎓",
    layout="wide"
)

from src.models.entities import Course, ExamModel, ExamAttempt, Document, TopicPerformance
from src.services.ingestion import DocumentIngestor
from src.services.generator import QuestionGenerator
from src.services.exam_engine import ExamEngine


@st.cache_resource(show_spinner=False)
def initialize_database():
    try:
        init_db()
        return True
    except Exception as exc:
        return exc


def ensure_database_ready():
    result = initialize_database()
    if result is True:
        return True

    st.error("Database connection failed. Check DATABASE_URL in .env or switch to SQLite for local testing.")
    st.caption(str(result))
    return False


@st.cache_resource(show_spinner=False)
def get_exam_engine():
    return ExamEngine()


@st.cache_resource(show_spinner=False)
def get_document_ingestor():
    return DocumentIngestor()


@st.cache_resource(show_spinner=False)
def get_question_generator():
    return QuestionGenerator()


def get_device_session_id():
    if "device_session_id" not in st.session_state:
        query_sid = st.query_params.get("sid")
        if isinstance(query_sid, list):
            query_sid = query_sid[0] if query_sid else None

        st.session_state.device_session_id = query_sid or f"device_{uuid.uuid4().hex[:12]}"

    if st.query_params.get("sid") != st.session_state.device_session_id:
        st.query_params["sid"] = st.session_state.device_session_id

    return st.session_state.device_session_id

# Custom CSS for a better look
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; }
    .exam-card { padding: 20px; border-radius: 10px; background-color: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
    .review-card { padding: 16px; border-radius: 8px; margin: 14px 0; border: 1px solid #d0d7de; background: #ffffff; }
    .review-card p { margin: 8px 0; }
    .review-header { display: flex; justify-content: space-between; gap: 12px; margin-bottom: 10px; }
    .review-correct { border-color: #2da44e; background: #f0fff4; }
    .review-correct .review-header span { color: #116329; font-weight: 700; }
    .review-wrong { border-color: #cf222e; background: #fff1f1; }
    .review-wrong .review-header span { color: #a40e26; font-weight: 700; }
    </style>
    """, unsafe_allow_html=True)

def main():
    device_session_id = get_device_session_id()
    st.sidebar.title("🎓 CBT Exam Platform")
    st.sidebar.markdown(f"**{settings.APP_NAME}**")
    st.sidebar.markdown(f"Version {settings.APP_VERSION}")
    st.sidebar.caption(f"Session: {device_session_id}")

    menu = ["🏠 Dashboard", "📚 Course Management", "📝 Take Exam", "📊 My Results"]
    choice = st.sidebar.selectbox("Navigation", menu)

    if choice == "🏠 Dashboard":
        show_dashboard()
    elif choice == "📚 Course Management":
        show_course_mgmt()
    elif choice == "📝 Take Exam":
        show_take_exam()
    elif choice == "📊 My Results":
        show_results()

def show_dashboard():
    st.title("University CBT Dashboard")
    if not ensure_database_ready():
        return

    with get_db_session() as session:
        courses = session.query(Course).all()
        total_courses = len(courses)
        total_exams = session.query(ExamModel).count()
        total_attempts = session.query(ExamAttempt).count()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Courses", total_courses)
    col2.metric("Total Exams", total_exams)
    col3.metric("Attempts Taken", total_attempts)

    st.markdown("---")
    st.subheader("Quick Start")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🚀 Start a New Exam"):
            st.info("Please navigate to 'Take Exam' in the sidebar.")
    with col_b:
        if st.button("📚 Manage Materials"):
            st.info("Please navigate to 'Course Management' in the sidebar.")

def show_course_mgmt():
    st.title("Course & Material Management")
    if not ensure_database_ready():
        return

    tab1, tab2, tab3, tab4 = st.tabs(["➕ Add Course", "📁 Index Materials", "🤖 Generate Exam", "🗑️ Delete Course"])

    with tab1:
        st.subheader("Create New Course")
        with st.form("course_form"):
            code = st.text_input("Course Code (e.g., SOE_510)")
            title = st.text_input("Course Title")
            desc = st.text_area("Description")
            submit = st.form_submit_button("Create Course")

            if submit:
                if code and title:
                    with get_db_session() as session:
                        # Simple check
                        exists = session.query(Course).filter(Course.code == code).first()
                        if exists:
                            st.error("Course code already exists!")
                        else:
                            new_course = Course(code=code, title=title, description=desc)
                            session.add(new_course)
                            st.success(f"Course {code} created successfully!")
                            st.rerun()
                else:
                    st.warning("Please fill in required fields.")

    with tab2:
        st.subheader("Index Course Materials")
        with get_db_session() as session:
            courses = session.query(Course).all()
            course_list = {c.code: c.title for c in courses}

        if course_list:
            selected_course = st.selectbox("Select Course", options=list(course_list.keys()), format_func=lambda x: f"{x} - {course_list[x]}", key="index_course_select")

            # Ensure courses directory exists
            course_dir = settings.COURSES_DIR / selected_course
            course_dir.mkdir(parents=True, exist_ok=True)

            st.info(f"Ensure PDF/PPTX/DOCX files are placed in: `{str(course_dir)}`")

            if st.button("🔄 Sync & Index Materials"):
                with st.spinner("Parsing and embedding documents..."):
                    with get_db_session() as session:
                        course = session.query(Course).filter(Course.code == selected_course).first()
                        course_title = course.title if course else "Unknown"
                    ingestor = get_document_ingestor()
                    stats = ingestor.process_course(selected_course, course_title)
                    if stats and stats.get("indexed", 0) > 0:
                        st.success(f"Indexing complete! Indexed {stats['indexed']} of {stats['processed']} file(s).")
                        if stats.get("skipped", 0):
                            st.warning(f"Skipped {stats['skipped']} file(s) because no text could be extracted.")
                    else:
                        st.warning("No files were indexed. If these are scanned PDFs, confirm OCR dependencies are installed.")
        else:
            st.info("No courses available. Please create a course first in the 'Add Course' tab.")

    with tab3:
        st.subheader("AI Exam Generator")
        with get_db_session() as session:
            courses = session.query(Course).all()
            course_list = {c.code: c.title for c in courses}

        if course_list:
            selected_course = st.selectbox("Select Course", options=list(course_list.keys()), format_func=lambda x: f"{x} - {course_list[x]}", key="generate_course_select")

            col1, col2 = st.columns(2)
            with col1:
                exam_title = st.text_input("Exam Title", value="Practice Exam 1")
                difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard", "Mixed"])
            with col2:
                topic = st.text_input("Topic (Optional - leave blank for general)")
                count = st.number_input("Number of Questions", min_value=1, max_value=100, value=15)

            if st.button("🪄 Generate AI Exam"):
                with st.spinner("Groq is crafting your exam..."):
                    generator = get_question_generator()
                    exam_id = generator.generate_exam(
                        course_code=selected_course,
                        title=exam_title,
                        topic=topic if topic else None,
                        difficulty=difficulty,
                        count=int(count)
                    )
                    if exam_id:
                        st.success(f"Exam {exam_id} generated successfully!")
                    else:
                        st.error("Failed to generate exam.")
                        if generator.last_error:
                            st.caption(generator.last_error)
        else:
            st.info("No courses available. Please create a course first in the 'Add Course' tab.")

    with tab4:
        st.subheader("Delete Course")
        with get_db_session() as session:
            courses = session.query(Course).all()
            course_list = {c.code: c.title for c in courses}

        if course_list:
            selected_course = st.selectbox("Select Course to Delete", options=list(course_list.keys()), format_func=lambda x: f"{x} - {course_list[x]}", key="delete_course_select")

            st.warning(f"⚠️ This will permanently delete course **{selected_course} - {course_list[selected_course]}** and all associated data (documents, exams, attempts, etc.)")

            confirm = st.checkbox("I understand this action cannot be undone")

            if st.button("🗑️ Delete Course", type="primary", disabled=not confirm):
                with get_db_session() as session:
                    course = session.query(Course).filter(Course.code == selected_course).first()
                    if course:
                        session.delete(course)
                        session.commit()
                        st.success(f"Course {selected_course} deleted successfully!")
                        st.rerun()
        else:
            st.info("No courses available to delete.")

def show_results():
    st.title("My Performance Analytics")
    if not ensure_database_ready():
        return

    device_session_id = get_device_session_id()
    with get_db_session() as session:
        attempts = (
            session.query(ExamAttempt)
            .filter(ExamAttempt.student_identifier == device_session_id)
            .order_by(ExamAttempt.created_at.desc())
            .all()
        )
        attempt_rows = [{
            "Attempt ID": a.attempt_id,
            "Course": a.course_id,
            "Score": a.score,
            "Total": a.total_marks,
            "Percentage": a.percentage,
            "Status": "Passed" if a.is_passed else "Failed",
            "Date": a.created_at.strftime("%Y-%m-%d %H:%M")
        } for a in attempts]
        attempt_options = [a.attempt_id for a in attempts]

    if not attempts:
        st.info("No exam attempts found yet for this device session.")
        return

    # Table of attempts
    df = pd.DataFrame(attempt_rows)

    st.table(df)

    # Detail view
    selected_att = st.selectbox("View detailed analysis for Attempt ID", options=attempt_options, key="results_attempt_select")
    if selected_att:
        with get_db_session() as session:
            attempt = session.query(ExamAttempt).filter(ExamAttempt.attempt_id == selected_att).first()
            if attempt:
                perf = session.query(TopicPerformance).filter(TopicPerformance.attempt_id == attempt.id).all()

                if perf:
                    perf_df = pd.DataFrame([{
                        "Topic": p.topic_name,
                        "Correct": p.correct_questions,
                        "Total": p.total_questions,
                        "Percentage": p.percentage
                    } for p in perf])

                    fig = px.bar(perf_df, x="Topic", y="Percentage", color="Percentage",
                                 title="Performance by Topic",
                                 color_continuous_scale="RdYlGn")
                    st.plotly_chart(fig)
                else:
                    st.warning("No topic analysis available for this attempt.")

def show_exam_review(results):
    import html

    st.markdown(f"### Score: {results['score']}/{results['total']} ({results['percentage']:.2f}%)")
    if results["is_passed"]:
        st.success("Congratulations! You passed.")
    else:
        st.error("You did not pass this attempt.")

    st.markdown("### Answer Review")
    for i, question in enumerate(results.get("questions", []), start=1):
        status_class = "review-correct" if question["is_correct"] else "review-wrong"
        status_text = "Correct" if question["is_correct"] else "Wrong"
        selected = question.get("selected_answer") or "No answer"
        correct = question["correct_answer"]

        st.markdown(
            f"""
            <div class="review-card {status_class}">
                <div class="review-header">
                    <strong>Question {i}</strong>
                    <span>{status_text}</span>
                </div>
                <p>{html.escape(question["text"])}</p>
                <p><strong>Your answer:</strong> {html.escape(selected)}{": " + html.escape(question["options"].get(selected, "")) if selected in question["options"] else ""}</p>
                <p><strong>Correct answer:</strong> {html.escape(correct)}: {html.escape(question["options"].get(correct, ""))}</p>
                <p><strong>Explanation:</strong> {html.escape(question.get("explanation") or "No explanation available.")}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

def clear_question_state():
    for key in list(st.session_state.keys()):
        if str(key).startswith("q_"):
            del st.session_state[key]

def show_take_exam():
    st.title("Take CBT Exam")
    if not ensure_database_ready():
        return

    if "last_results" in st.session_state:
        show_exam_review(st.session_state.last_results)
        if st.button("Start Another Exam"):
            del st.session_state.last_results
            clear_question_state()
            st.rerun()
        return

    with get_db_session() as session:
        courses = session.query(Course).all()
        course_list = {c.code: c.title for c in courses}

    if course_list:
        selected_course = st.selectbox("Select Course", options=list(course_list.keys()), format_func=lambda x: f"{x} - {course_list[x]}", key="take_course_select")

        if selected_course:
            with get_db_session() as session:
                course = session.query(Course).filter(Course.code == selected_course).first()
                if course:
                    exams = session.query(ExamModel).filter(ExamModel.course_id == course.id).all()
                    exam_list = {e.exam_id: e.title for e in exams}

                    if exam_list:
                        selected_exam = st.selectbox("Select Exam", options=list(exam_list.keys()), format_func=lambda x: f"{x} - {exam_list[x]}", key="take_exam_select")

                        if st.button("🚀 Start Exam"):
                            clear_question_state()
                            engine = get_exam_engine()
                            attempt_data = engine.start_exam(
                                selected_exam,
                                student_id=get_device_session_id(),
                            )
                            if attempt_data:
                                st.session_state.active_attempt = attempt_data
                                st.rerun()
                            else:
                                st.error("Could not start exam. Please try again.")
                    else:
                        st.info("No exams available for this course. Generate one in Course Management.")
                else:
                    st.error("Course not found.")
    else:
        st.info("No courses available. Please create a course first in Course Management.")

    if "active_attempt" in st.session_state:
        attempt = st.session_state.active_attempt
        st.markdown(f"### {attempt['exam_title']}")
        st.info(f"Time Limit: {attempt['duration']} minutes")

        user_answers = {}

        for i, q in enumerate(attempt['questions']):
            st.markdown(f"**Question {i+1}:** {q['text']}")
            # Use a unique key for each radio button
            user_answers[q['uid']] = st.radio(
                f"Select answer for {q['uid']}",
                options=["A", "B", "C", "D"],
                format_func=lambda x: f"{x}: {q['options'][x]}",
                index=None,
                key=f"q_{q['uid']}"
            )
            st.markdown("---")

        if st.button("✅ Submit Exam"):
            with st.spinner("Calculating score..."):
                engine = get_exam_engine()
                results = engine.submit_exam(attempt['attempt_id'], user_answers)
                if not results:
                    st.error("Could not submit exam. Please try again.")
                    return

                st.session_state.last_results = results

                # Clear attempt from state
                del st.session_state.active_attempt
                st.rerun()


if __name__ == "__main__":
    main()
