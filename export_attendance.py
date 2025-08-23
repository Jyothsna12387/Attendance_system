import sqlite3
import pandas as pd

# Step 1: Connect to your SQLite database
conn = sqlite3.connect("data/attendance1.db")
cursor = conn.cursor()

# Step 2: Read the attendance table into a DataFrame
df = pd.read_sql_query("SELECT * FROM attendance", conn)

# Step 3: Export path (change this to your actual path)
export_path = r"C:\Users\BHAVYA\Downloads\attendance_export.csv"

# Step 4: Save DataFrame to CSV
df.to_csv(export_path, index=False)

print(f"✅ Attendance exported successfully to: {export_path}")

# Step 5: Close the connection
conn.close()
