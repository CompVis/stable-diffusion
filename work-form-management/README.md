Work Form Management System

A minimal Flask + SQLite app to maintain workflow, due list, priorities and stages.

Quick start (Windows):
1. Create a virtualenv: python -m venv .venv
2. Activate: .\.venv\Scripts\activate
3. Install: pip install -r requirements.txt
4. Run: set FLASK_APP=app.py && flask run

Use the web UI at http://127.0.0.1:5000 to add tasks, edit, mark complete, and view due dates.

Notes:
- Database is stored at work-form-management/instance/tasks.db
- Fields: title, description, status, workflow_stage, due_date, priority
