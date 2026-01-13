import sqlite3
import os

DB_PATH = os.path.join("data", "attendance1.db")
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

try:
    cursor.execute("ALTER TABLE faces ADD COLUMN year TEXT")
    print("Column 'year' added successfully to faces table.")
except sqlite3.OperationalError:
    print("Column 'year' already exists.")

conn.commit()
conn.close()