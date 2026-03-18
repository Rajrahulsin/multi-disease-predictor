import sqlite3
import json
from datetime import datetime

DB_PATH = "D:/multi-disease-predictor/predictions.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            disease     TEXT,
            input_data  TEXT,
            prediction  TEXT,
            confidence  REAL,
            model_used  TEXT,
            timestamp   TEXT
        )
    """)
    conn.commit()
    conn.close()
    print("✓ Database initialized!")

def save_prediction(disease, input_data, prediction, confidence, model_used):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions (disease, input_data, prediction, confidence, model_used, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        disease,
        json.dumps(input_data),
        prediction,
        round(confidence, 2),
        model_used,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    conn.commit()
    conn.close()

def get_history():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows
