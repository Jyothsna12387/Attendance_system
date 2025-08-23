import sqlite3

conn = sqlite3.connect('data/attendance1.db')
cursor = conn.cursor()

cursor.execute("DROP TABLE IF EXISTS attendance")

cursor.execute('''
    CREATE TABLE attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        reg_no TEXT,
        name TEXT,
        type TEXT NOT NULL,
        time TEXT,
        date TEXT
    )
''')

print("[INFO] Recreated attendance table with 'type' column.")
conn.commit()
conn.close()