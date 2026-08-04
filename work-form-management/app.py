from flask import Flask, g, render_template, request, redirect, url_for
import sqlite3
import os
import datetime

BASE_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE_DIR, 'instance', 'tasks.db')
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

app = Flask(__name__)

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DB_PATH)
        db.row_factory = sqlite3.Row
    return db

def init_db():
    db = get_db()
    db.execute('''CREATE TABLE IF NOT EXISTS tasks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        description TEXT,
        status TEXT NOT NULL DEFAULT 'pending',
        workflow_stage TEXT,
        due_date TEXT,
        priority INTEGER DEFAULT 3,
        created_at TEXT,
        updated_at TEXT
    )''')
    db.commit()

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

@app.before_request
def before_request():
    init_db()

@app.route('/')
def index():
    db = get_db()
    rows = db.execute('SELECT * FROM tasks ORDER BY due_date IS NULL, due_date, priority').fetchall()
    tasks = [dict(r) for r in rows]
    return render_template('index.html', tasks=tasks, today=str(datetime.date.today()))

@app.route('/add', methods=['GET','POST'])
def add():
    if request.method == 'POST':
        title = request.form['title'].strip()
        description = request.form.get('description','').strip()
        due_date = request.form.get('due_date') or None
        workflow_stage = request.form.get('workflow_stage','Backlog')
        priority = int(request.form.get('priority',3))
        created = datetime.datetime.utcnow().isoformat()
        db = get_db()
        db.execute('INSERT INTO tasks (title, description, due_date, workflow_stage, priority, created_at, updated_at) VALUES (?,?,?,?,?,?,?)',
                   (title, description, due_date, workflow_stage, priority, created, created))
        db.commit()
        return redirect(url_for('index'))
    return render_template('edit.html', task=None)

@app.route('/edit/<int:task_id>', methods=['GET','POST'])
def edit(task_id):
    db = get_db()
    if request.method == 'POST':
        title = request.form['title'].strip()
        description = request.form.get('description','').strip()
        due_date = request.form.get('due_date') or None
        workflow_stage = request.form.get('workflow_stage','Backlog')
        priority = int(request.form.get('priority',3))
        updated = datetime.datetime.utcnow().isoformat()
        db.execute('UPDATE tasks SET title=?, description=?, due_date=?, workflow_stage=?, priority=?, updated_at=? WHERE id=?',
                   (title, description, due_date, workflow_stage, priority, updated, task_id))
        db.commit()
        return redirect(url_for('index'))
    row = db.execute('SELECT * FROM tasks WHERE id=?', (task_id,)).fetchone()
    if not row:
        return redirect(url_for('index'))
    task = dict(row)
    return render_template('edit.html', task=task)

@app.route('/complete/<int:task_id>')
def complete(task_id):
    db = get_db()
    db.execute("UPDATE tasks SET status='done', updated_at=? WHERE id=?", (datetime.datetime.utcnow().isoformat(), task_id))
    db.commit()
    return redirect(url_for('index'))

@app.route('/delete/<int:task_id>')
def delete(task_id):
    db = get_db()
    db.execute('DELETE FROM tasks WHERE id=?', (task_id,))
    db.commit()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
