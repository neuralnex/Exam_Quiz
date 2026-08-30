# AI-Powered University CBT Exam Preparation Platform

A Streamlit app for building course-based CBT practice exams from PDF, PPTX, DOCX, and DOC materials. It indexes course documents into a shared knowledge base, generates multiple-choice exams with Groq, lets multiple users take exams independently, and shows scoring plus answer review.

## Features

- Course management
- Document indexing for PDF, PPTX, DOCX, and DOC files
- OCR fallback for scanned PDFs when Tesseract is installed
- AI-generated multiple-choice exams
- Per-device exam sessions using Streamlit session state and a `sid` URL parameter
- Shared course database, vector store, and knowledge base across users
- Score summary with correct and incorrect answers highlighted
- Heroku-ready `Procfile`

## Requirements

- Python 3.12 recommended
- Groq API key
- Optional Pinecone API key for hosted vector search
- Optional PostgreSQL database for production
- Optional Tesseract OCR binary for scanned PDFs

## Local Setup

Clone the project and enter the directory:

```bash
git clone <https://github.com/neuralnex/Exam_Quiz.git>
cd Exam_Quiz
```

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create your environment file:

```bash
cp .env.example .env
```

Edit `.env` and set at least:

```env
GROQ_API_KEY=your_groq_api_key_here
DATABASE_URL=sqlite:///cbt_exam.db
COURSES_DIR=courses
```

For local testing, SQLite is easiest. For production, use PostgreSQL:

```env
DATABASE_URL=postgresql://user:password@host:5432/dbname
```

## OCR Setup

The Python OCR packages are installed from `requirements.txt`, but scanned PDFs also need the system `tesseract` executable.

On Debian, Ubuntu, or Kali:

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr
```

Without Tesseract, searchable PDFs will still work, but scanned/image-only PDFs may be skipped.

## Course Materials

Put course files inside `courses/<COURSE_CODE>/`.

Example:

```text
courses/
  SOE_506/
    lecture-notes.pdf
    slides.pptx
  SOE_510/
    module-1.docx
```

Then open the app, create the matching course code, and click **Sync & Index Materials**.

## Run Locally

```bash
streamlit run app.py
```

Open the local URL shown by Streamlit, usually:

```text
http://localhost:8501
```

## Basic Workflow

1. Open **Course Management**.
2. Create a course.
3. Place materials in `courses/<COURSE_CODE>/`.
4. Click **Sync & Index Materials**.
5. Generate an exam from the indexed materials.
6. Open **Take Exam** and start an attempt.
7. Submit to see your score and answer review.

## Multiple Users

The app supports multiple users/devices at the app level:

- Each browser/device gets a generated session id.
- The session id is stored in the URL as `?sid=...`.
- Attempts and results are separated by that session id.
- Courses, generated exams, database records, and indexed knowledge base remain shared.

For production traffic, scale process/workers at the platform level.

## Heroku Deployment

This repo includes a `Procfile`:

```Procfile
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false
```

Set config vars in Heroku:

```bash
heroku config:set GROQ_API_KEY=your_groq_api_key_here
heroku config:set DATABASE_URL=your_postgres_url
heroku config:set COURSES_DIR=courses
```

If using Pinecone:

```bash
heroku config:set PINECONE_API_KEY=your_pinecone_api_key_here
heroku config:set PINECONE_INDEX=exam
heroku config:set PINECONE_ENVIRONMENT=us-east-1
```

Deploy:

```bash
git push heroku main
```

Heroku filesystems are ephemeral, so for production you should not rely on uploaded/local course files staying on disk forever. Use persistent storage or include the files in your deployment process.

## Notes

- Do not commit `.env` or real API keys.
- `.vectorstore_data.json` is the local fallback vector store. For production, prefer Pinecone or another persistent vector database.
- The first indexing run can be slow because embedding models may download and initialize.
- If exam generation fails, the app shows the backend error under the failure message.
