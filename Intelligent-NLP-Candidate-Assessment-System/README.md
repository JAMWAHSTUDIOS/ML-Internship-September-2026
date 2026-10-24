# Intelligent NLP Candidate Assessment System

A beginner-friendly Flask + SQLite foundation for job management and candidate applications.

## Current features

- Candidate registration and login
- Admin login through a Flask CLI command
- Role-protected Admin and Candidate dashboards
- Admin job management: create, edit, draft, publish, close, and reopen jobs
- Job fields: title, description, qualification, skills, experience, and deadline
- Admin can add assessment questions, one question per line, for each job
- Published job browsing and job detail pages for candidates
- Candidate resume upload (PDF, DOC, DOCX; up to 5 MB)
- Candidate can view their applications and download their own uploaded resume
- Admin can see recent applications and download submitted resumes
- Database startup creates new tables and adds resume columns to an existing database without deleting records

## Setup

Use Python 3.10 or newer. In a terminal opened inside this project folder:

```powershell
python -m venv .venv
.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000 in your browser.

## Admin account

If you already have an Admin account and database from the previous version, keep your existing `users.db`. Do not run `create-admin` again unless you genuinely need another Admin account.

To create an Admin only when needed:

```powershell
flask --app app create-admin
```

## Admin workflow

1. Log in with your Admin account.
2. Select **Create Job**.
3. Enter the job title, description, qualification, skills, and experience.
4. Add assessment questions, one question per line.
5. Optionally set an application deadline.
6. Save as a draft or publish the job. Publishing requires all three requirement fields and at least one assessment question.
7. Edit jobs or close/reopen published jobs from the dashboard.

## Candidate workflow

1. Register or log in as a candidate.
2. Browse published jobs.
3. Open a job to review its description and requirements.
4. Upload a PDF, DOC, or DOCX resume (maximum 5 MB) and submit the application.
5. Track the application status and download the submitted resume from the dashboard.

## Resume screening and assessment workflow

1. A candidate submits a PDF or DOCX resume for a published job.
2. Groq compares the resume with the requirements stored for that job and saves the skill-gap report.
3. An admin reviews the report. If the model marks the candidate `POTENTIALLY_ELIGIBLE`, the admin can assign the assessment from the application review page.
4. The assigned assessment appears in the candidate dashboard.
5. After submission, Groq returns a reference answer, correctness score, and feedback for each answer. Python separately calculates a cosine similarity score from word-frequency vectors.
6. The admin sees every answer, cosine similarity, Groq correctness score, reference answer, and feedback on the application review page.

Cosine similarity is a text-similarity signal, not a reliable correctness score by itself. Use it alongside the LLM feedback and human review. The current implementation uses a simple word-frequency vector, not a semantic embedding model.

If a candidate already has an application record for a job but has not been assigned an assessment, the application page lets them replace their resume and rerun the analysis instead of creating a duplicate application. Existing `users.db` data is preserved.

## Preserving existing data

1. Back up your existing `users.db`.
2. Copy the updated application files into your existing project folder.
3. Keep `users.db` in that folder.
4. Start the app. `init_db()` creates the new `job_questions` table and adds resume filename columns to `applications` when missing.

Do not delete `users.db`, and do not run `create-admin` if you already have the Admin account you need.

## Development security note

This is a local development starter, not production-ready. Before public deployment, configure a strong `SECRET_KEY`, add CSRF protection, validate uploaded file contents (not just extensions), scan files, and apply production server/security settings.


## Groq resume analysis setup

1. Install dependencies: `python -m pip install -r requirements.txt`.
2. Copy `.env.example` to `.env` in the same folder as `app.py`.
3. Put your Groq API key in `GROQ_API_KEY` inside `.env`. Keep `.env` private.
4. Run the application using `python app.py`.
5. A candidate submits a PDF or DOCX resume for a published job. The app saves the resume under `uploads/resumes/`, reads the job requirements from `users.db`, calls Groq from the Flask server, and stores the analysis in the `resume_analyses` table.
6. Candidates see their analysis on the job detail page. Admins can open an application review from the Recent Applications table and retry failed analyses.
7. Once the analysis marks a candidate potentially eligible, the admin can assign the assessment. The candidate opens it from their dashboard; after submission, scores and feedback are shown to the admin.

The analysis is advisory only. A model/API failure does not delete the application. Scanned PDFs without embedded text need OCR, which is not included.
