import sqlite3

conn = sqlite3.connect("data/attendance1.db")
cursor = conn.cursor()

cursor.execute("PRAGMA table_info(faces)")
columns = cursor.fetchall()

for column in columns:
    print(column)

conn.close()
