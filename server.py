from flask import Flask, render_template, redirect, request, url_for, jsonify, session
import subprocess
import sys
import os
import base64
import cv2
import numpy as np
import torch
import sqlite3
from PIL import Image
from datetime import datetime
from facenet_pytorch import MTCNN, InceptionResnetV1

app = Flask(__name__)
app.secret_key = "attendance_secret_key"

# ==========================
# 1) CONFIG & MODELS
# ==========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_EXE = sys.executable 
DB_PATH = os.path.join(BASE_DIR, "data", "attendance1.db")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MTCNN thresholds తగ్గించాను (Look Down పనిచేయడానికి)
mtcnn = MTCNN(
    image_size=160, 
    margin=14, 
    keep_all=False, 
    thresholds=[0.5, 0.6, 0.6], 
    device=device
)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

temp_embeddings = {}

# ==========================
# 2) WEB ROUTES
# ==========================
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/secure_in")
def secure_in():
    subprocess.Popen([PYTHON_EXE, os.path.join(BASE_DIR, "main.py"), "IN"], cwd=BASE_DIR)
    return redirect(url_for('home'))

@app.route("/secure_out")
def secure_out():
    subprocess.Popen([PYTHON_EXE, os.path.join(BASE_DIR, "main.py"), "OUT"], cwd=BASE_DIR)
    return redirect(url_for('home'))

@app.route('/register')
def register_page():
    if session.get('role') != 'faculty':
        return redirect(url_for('login_page'))
    return render_template('register.html')

# ==========================
# 3) API ROUTES
# ==========================

@app.route('/api/check_id', methods=['POST'])
def check_id():
    data = request.json
    reg_no = data.get('reg_no', '').upper()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM faces WHERE reg_no = ?", (reg_no,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return jsonify({"exists": True, "name": row[0]})
    return jsonify({"exists": False})


# ==========================
# POSE DETECTION LOGIC
# ==========================
def verify_pose(landmarks, expected_pose):
    if landmarks is None or len(landmarks) == 0:
        return False, "Face not clear"
        
    lm = landmarks[0] 
    # lm[0]=left_eye, lm[1]=right_eye, lm[2]=nose, lm[3]=mouth_left, lm[4]=mouth_right
    
    nose_x, nose_y = lm[2][0], lm[2][1]
    left_eye_x, left_eye_y = lm[0][0], lm[0][1]
    right_eye_x, right_eye_y = lm[1][0], lm[1][1]
    
    eye_dist = right_eye_x - left_eye_x
    if eye_dist == 0: return False, "Face too far or not clear"
    
    # Horizontal ratio (Left/Right)
    # ముక్కు ఎడమ కంటికి, కుడి కంటికి మధ్య ఎక్కడ ఉందో లెక్కిస్తుంది
    ratio_h = (nose_x - left_eye_x) / eye_dist
    
    detected = "Look Straight"
    if ratio_h < 0.38: 
        detected = "Turn Face Right"
    elif ratio_h > 0.62: 
        detected = "Turn Face Left"
    
    # Vertical Check (Up/Down)
    # కళ్ళు మరియు ముక్కు మధ్య ఉండే నిలువు దూరాన్ని బట్టి
    eye_y_avg = (left_eye_y + right_eye_y) / 2
    vertical_diff = nose_y - eye_y_avg
    
    if vertical_diff < eye_dist * 0.45: 
        detected = "Look Up"
    elif vertical_diff > eye_dist * 0.75: 
        detected = "Look Down"

    return (detected == expected_pose), detected


@app.route('/api/process_web_pose', methods=['POST'])
def process_web_pose():
    try:
        data = request.json
        name, reg_no = data.get('name'), data.get('reg_no').upper()
        # year ఫీల్డ్‌ని ఇక్కడ capture చేస్తున్నాం
        year = data.get('year') 
        expected_pose = data.get('pose')
        img_base64 = data.get('image').split(',')[1]
        
        img_bytes = base64.b64decode(img_base64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb).convert('RGB')

        # Detect face and landmarks
        boxes, probs, landmarks = mtcnn.detect(pil_img, landmarks=True)
        
        if boxes is not None:
            # Step 1: Pose Validation
            is_correct, detected = verify_pose(landmarks, expected_pose)
            if not is_correct:
                return jsonify({"success": False, "message": f"Wrong Pose! Please {expected_pose}"})

            # Step 2: Get Embedding
            face_tensor = mtcnn(pil_img)
            if face_tensor is not None:
                with torch.no_grad():
                    emb = model(face_tensor.unsqueeze(0).to(device)).cpu().numpy().flatten()
                
                if reg_no not in temp_embeddings: temp_embeddings[reg_no] = []
                temp_embeddings[reg_no].append(emb)
                
                if len(temp_embeddings[reg_no]) >= 5:
                    avg_emb = np.mean(temp_embeddings[reg_no], axis=0).astype(np.float32)
                    conn = sqlite3.connect(DB_PATH)
                    cursor = conn.cursor()
                    
                    # De-duplication Check
                    cursor.execute("SELECT reg_no, name, embedding FROM faces")
                    for ex_reg, ex_name, ex_emb_blob in cursor.fetchall():
                        ex_emb = np.frombuffer(ex_emb_blob, dtype=np.float32)
                        if np.linalg.norm(avg_emb - ex_emb) < 0.6:
                            conn.close(); del temp_embeddings[reg_no]
                            return jsonify({"success": False, "message": f"Face already registered as {ex_name}"})

                    # Insert Query లో 'year' కాలమ్‌ని యాడ్ చేశాను
                    cursor.execute("""
                        INSERT INTO faces (reg_no, name, phone, branch, section, year, embedding) 
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (reg_no, name, data.get('phone'), data.get('branch'), data.get('section'), year, avg_emb.tobytes()))
                    
                    conn.commit(); conn.close(); del temp_embeddings[reg_no]
                    return jsonify({"success": True, "completed": True})
                
                return jsonify({"success": True, "completed": False})
        
        return jsonify({"success": False, "message": "Face not detected clearly"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})

# ==========================
# 4) LOGIN & DASHBOARD
# ==========================
@app.route('/login')
def login_page():
    return render_template('login.html')

@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.json
    username = data.get('username', '').upper()
    password = data.get('password')

    if username == "ADMIN" and password == "ADMIN123":
        session['user_name'] = "Admin Faculty"
        session['role'] = 'faculty'
        return jsonify({"success": True, "redirect": url_for('faculty_dashboard')})

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name, reg_no FROM faces WHERE reg_no = ?", (username,))
    student = cursor.fetchone()
    conn.close()

    if student and password == username:
        session['reg_no'] = student[1]
        session['user_name'] = student[0]
        session['role'] = 'student'
        return jsonify({"success": True, "redirect": url_for('student_dashboard')})

    return jsonify({"success": False, "message": "Invalid Register Number or Password"})


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login_page'))


# ==========================
# DASHBOARDS
# ==========================
@app.route('/faculty_dashboard')
def faculty_dashboard():
    if session.get('role') != 'faculty': return redirect(url_for('login_page'))
    today = datetime.now().strftime('%Y-%m-%d')
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(DISTINCT reg_no) FROM attendance WHERE date = ?", (today,))
    today_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM faces")
    total_students = cursor.fetchone()[0]
    cursor.execute("SELECT name, reg_no, time, status FROM attendance WHERE date = ? ORDER BY id DESC LIMIT 5", (today,))
    recent_activities = cursor.fetchall()
    conn.close()
    return render_template('faculty_dashboard.html', user=session.get('user'), today_count=today_count, total_students=total_students, recent_activities=recent_activities)

@app.route('/student_dashboard')
def student_dashboard():
    if 'reg_no' not in session or session.get('role') != 'student':
        return redirect(url_for('login_page'))
    
    reg_no = session['reg_no']
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row 
    cursor = conn.cursor()
    
    cursor.execute("SELECT name, branch, section, phone, year FROM faces WHERE reg_no = ?", (reg_no,))
    student_info = cursor.fetchone()
    
    cursor.execute("SELECT COUNT(DISTINCT date) FROM attendance WHERE reg_no = ? AND status = 'IN'", (reg_no,))
    total_present = cursor.fetchone()[0]
    
    cursor.execute("SELECT date, time, status FROM attendance WHERE reg_no = ? ORDER BY id DESC LIMIT 10", (reg_no,))
    attendance_history = cursor.fetchall()
    conn.close()
    
    if student_info is None:
        return "Student data not found in database."

    return render_template('student_dashboard.html', 
                            info=student_info, 
                            reg_no=reg_no, 
                            total_present=total_present, 
                            history=attendance_history)

@app.route('/api/student_login', methods=['POST'])
def student_login():
    data = request.json
    reg_no = data.get('username', '').upper()
    password = data.get('password')
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM faces WHERE reg_no = ?", (reg_no,))
    user = cursor.fetchone()
    conn.close()
    
    if user and password == reg_no:
        session['reg_no'] = reg_no
        session['user_name'] = user[0]
        session['role'] = 'student'
        return jsonify({"success": True, "redirect": url_for('student_dashboard')})
    
    return jsonify({"success": False, "message": "Invalid Register Number or Password"})


# ==========================
# 5) MANAGE STUDENTS
# ==========================
@app.route('/manage_students')
def manage_students():
    if session.get('role') != 'faculty':
        return redirect(url_for('login_page'))
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT reg_no, name, branch, section, phone, year FROM faces")
    students = cursor.fetchall()
    conn.close()
    return render_template('manage_students.html', students=students)

@app.route('/api/delete_student/<reg_no>', methods=['DELETE'])
def api_delete_student(reg_no):
    if session.get('role') != 'faculty':
        return jsonify({"success": False, "message": "Unauthorized"}), 403
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM faces WHERE reg_no = ?", (reg_no,))
        cursor.execute("DELETE FROM attendance WHERE reg_no = ?", (reg_no,))
        conn.commit()
        conn.close()
        return jsonify({"success": True, "message": "Student deleted successfully"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})
    

# ==========================
# 6) ATTENDANCE REPORTS
# ==========================
@app.route('/attendance_reports')
def attendance_reports():
    if session.get('role') != 'faculty':
        return redirect(url_for('login_page'))
    
    selected_date = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
    f_branch = request.args.get('branch', 'All')
    f_section = request.args.get('section', 'All')
    f_year = request.args.get('year', 'All')
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Present List with Filters
    query_present = """
        SELECT a.reg_no, f.name, f.branch, f.section, a.time, f.year
        FROM attendance a
        JOIN faces f ON a.reg_no = f.reg_no
        WHERE a.date = ? AND a.status = 'IN'
    """
    params = [selected_date]
    if f_branch != 'All': query_present += " AND f.branch = ?"; params.append(f_branch)
    if f_section != 'All': query_present += " AND f.section = ?"; params.append(f_section)
    if f_year != 'All': query_present += " AND f.year = ?"; params.append(f_year)
        
    cursor.execute(query_present, params)
    present_list = cursor.fetchall()
    present_ids = [p[0] for p in present_list]
    
    # Absent List with Filters
    query_absent = "SELECT reg_no, name, branch, section, phone, year FROM faces WHERE 1=1"
    abs_params = []
    if f_branch != 'All': query_absent += " AND branch = ?"; abs_params.append(f_branch)
    if f_section != 'All': query_absent += " AND section = ?"; abs_params.append(f_section)
    if f_year != 'All': query_absent += " AND year = ?"; abs_params.append(f_year)
        
    cursor.execute(query_absent, abs_params)
    all_filtered = cursor.fetchall()
    absent_list = [s for s in all_filtered if s[0] not in present_ids]
    
    conn.close()
    return render_template('reports.html', 
                           present=present_list, 
                           absent=absent_list, 
                           date=selected_date,
                           f_branch=f_branch,
                           f_section=f_section,
                           f_year=f_year)

if __name__ == "__main__":
    app.run(debug=True, port=5000)