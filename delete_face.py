import sqlite3

# Connect to the database
conn = sqlite3.connect('data/attendance1.db')
cursor = conn.cursor()

# Ask user for registration number
reg_no = input("Enter the registration number of the student to delete: ")

# Check if student exists
cursor.execute("SELECT * FROM faces WHERE reg_no = ?", (reg_no,))
student = cursor.fetchone()

if student:
    print(f"[INFO] Found student: Reg No = {student[0]}, Name = {student[1]}")
    confirm = input("Are you sure you want to delete this student? (y/n): ").strip().lower()
    if confirm == 'y':
        cursor.execute("DELETE FROM faces WHERE reg_no = ?", (reg_no,))
        conn.commit()
        print(f"[✅ SUCCESS] Student with Reg No {reg_no} deleted successfully.")
    else:
        print("[INFO] Deletion cancelled.")
else:
    print("[❌ ERROR] No student found with that registration number.")

conn.close()
