from flask import Flask, render_template, redirect, request, url_for, jsonify, session, send_file
import subprocess
import sys
import os
import base64
import mediapipe as mp
import cv2
import numpy as np
import torch
import sqlite3
from PIL import Image
from datetime import datetime
from facenet_pytorch import MTCNN, InceptionResnetV1
import pandas as pd
from io import BytesIO
import math


app = Flask(__name__)

app.secret_key = "attendance_secret_key"


# Mediapipe Face Mesh ని సిద్ధం చేయడం
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=10, refine_landmarks=True)

def calculate_ear(landmarks, eye_indices):
    """కంటి పాయింట్ల ఆధారంగా EAR లెక్కించే లాజిక్"""
    p1 = landmarks[eye_indices[0]]
    p2 = landmarks[eye_indices[1]]
    p3 = landmarks[eye_indices[2]]
    p4 = landmarks[eye_indices[3]]
    p5 = landmarks[eye_indices[4]]
    p6 = landmarks[eye_indices[5]]
    
    # EAR ఫార్ములా
    vertical_1 = np.linalg.norm(np.array([p2.x, p2.y]) - np.array([p6.x, p6.y]))
    vertical_2 = np.linalg.norm(np.array([p3.x, p3.y]) - np.array([p5.x, p5.y]))
    horizontal = np.linalg.norm(np.array([p1.x, p1.y]) - np.array([p4.x, p4.y]))
    
    # ఇక్కడ ear_value ని డిఫైన్ చేసి రిటర్న్ చేస్తున్నాం
    ear_value = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear_value

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
    keep_all=True, 
    thresholds=[0.7, 0.8, 0.8], 
    device=device
)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

temp_embeddings = {}

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        reg_no TEXT,
        name TEXT,
        branch TEXT,    
        section TEXT,   
        year TEXT,      
        subject TEXT,   
        date TEXT,
        time TEXT,
        status TEXT
    )
''')
conn.commit()
conn.close()

# Server memory lo status maintain cheyyadaniki
attendance_status = {
    "is_in_active": False,
    "is_out_active": False
}

@app.route('/api/toggle_attendance/<mode>/<action>')
def toggle_attendance(mode, action):
    if session.get('role') != 'faculty':
        return jsonify({"success": False, "message": "Unauthorized"}), 403
    
    is_active = (action == 'start')
    if mode == 'IN':
        attendance_status["is_in_active"] = is_active
    elif mode == 'OUT':
        attendance_status["is_out_active"] = is_active
        
    return jsonify({"success": True, "status": is_active})
# ==========================
# 2) WEB ROUTES
# ==========================
@app.route('/')
def index():
    return redirect(url_for('login_page'))


@app.route("/secure_in")
def secure_in():
    if session.get('role') != 'faculty':
        return redirect(url_for('login_page'))
    return render_template("secure_in.html")

@app.route("/secure_out")
def secure_out():
    if session.get('role') != 'faculty':
        return redirect(url_for('login_page'))
    return render_template("secure_out.html")

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

    # --- కొత్తగా ఈ లైన్ యాడ్ చేయండి ---
    # కొత్త రిజిస్ట్రేషన్ మొదలవుతుంది కాబట్టి పాత మెమరీని క్లియర్ చేస్తున్నాం
    if reg_no in temp_embeddings:
        del temp_embeddings[reg_no]
        print(f"Cleared temporary memory for {reg_no}")

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
@app.route('/api/clear_temp_registration', methods=['POST'])
def clear_temp_registration():
    data = request.json
    reg_no = data.get('reg_no', '').upper()
    if reg_no in temp_embeddings:
        del temp_embeddings[reg_no]
        print(f"Tab switched/closed: Cleared memory for {reg_no}")
    return jsonify({"success": True})


def verify_pose(landmarks, expected_pose):
    if landmarks is None or len(landmarks) == 0:
        return False, "Face not clear"
        
    lm = landmarks[0] 
    nose_y = lm[2][1]
    left_eye_y = lm[0][1]
    right_eye_y = lm[1][1]
    
    # కళ్ళు మరియు ముక్కు మధ్య ఉండే యావరేజ్ దూరం
    eye_y_avg = (left_eye_y + right_eye_y) / 2
    
    # హారిజాంటల్ దూరం (Eye to Eye distance)
    eye_dist = lm[1][0] - lm[0][0]
    
    # నిలువు దూరం (Vertical difference)
    vertical_diff = nose_y - eye_y_avg
    
    # హారిజాంటల్ రేషియో (Left/Right కోసం)
    ratio_h = (lm[2][0] - lm[0][0]) / eye_dist
    
    detected = "Look Straight"

    # --- Look Down కి ప్రాధాన్యత ఇస్తున్నాం ---
    # కిందికి వంగినప్పుడు vertical_diff పెరుగుతుంది
    if vertical_diff > eye_dist * 0.58: # 0.75 నుండి 0.58 కి తగ్గించాను
        detected = "Look Down"
    elif vertical_diff < eye_dist * 0.38: 
        detected = "Look Up"
    elif ratio_h < 0.40:
        detected = "Turn Face Right"
    elif ratio_h > 0.60:
        detected = "Turn Face Left"

    return (detected == expected_pose), detected

@app.route('/api/process_web_pose', methods=['POST'])
def process_web_pose():
    try:
        data = request.json
        name = data.get('name')
        reg_no = data.get('reg_no', '').strip().upper()
        year = data.get('year')
        expected_pose = data.get('pose')
        img_base64 = data.get('image').split(',')[1]
        
        # Reg No పొడవు చెక్ చేయడం
        if len(reg_no) != 10:
           return jsonify({"success": False, "message": "Invalid Registration Number. Must be 10 characters."})
    
        img_bytes = base64.b64decode(img_base64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb).convert('RGB')

        boxes, probs, landmarks = mtcnn.detect(pil_img, landmarks=True)
        
        if boxes is not None:
            # 1. Pose Validation
            is_correct, detected = verify_pose(landmarks, expected_pose)
            if not is_correct:
                return jsonify({"success": False, "message": f"Wrong Pose! Please {expected_pose}"})

            # 2. Get Embedding
            face_tensor = mtcnn(pil_img)
            if face_tensor is not None:
                with torch.no_grad():
                    emb = model(face_tensor.unsqueeze(0).to(device)).cpu().numpy().flatten()
                
                if reg_no not in temp_embeddings:
                    temp_embeddings[reg_no] = []
                
                temp_embeddings[reg_no].append(emb)
                
                # --- ముఖ్యమైన మార్పు: మొదటి పోజ్ లోనే డూప్లికేట్ చెక్ ---
                if len(temp_embeddings[reg_no]) == 1:
                    conn = sqlite3.connect(DB_PATH)
                    cursor = conn.cursor()
                    cursor.execute("SELECT reg_no, name, embedding FROM faces")
                    rows = cursor.fetchall()
                    
                    for ex_reg, ex_name, ex_emb_blob in rows:
                        if ex_emb_blob:
                            ex_emb = np.frombuffer(ex_emb_blob, dtype=np.float32)
                            dist = np.linalg.norm(emb - ex_emb)
                            
                            # ఒకవేళ ఫేస్ మ్యాచ్ అయితే (Distance < 0.6)
                            if dist < 0.7:
                                conn.close()
                                del temp_embeddings[reg_no] # మెమరీ క్లియర్
                                return jsonify({
                                    "success": False, 
                                    "is_duplicate": True, 
                                    "message": f"Face already registered as {ex_name} ({ex_reg})"
                                })
                    conn.close()

                # --- 5 పోజులు పూర్తయ్యాక సేవ్ చేయడం ---
                if len(temp_embeddings[reg_no]) >= 5:
                    avg_emb = np.mean(temp_embeddings[reg_no], axis=0).astype(np.float32)
                    conn = sqlite3.connect(DB_PATH)
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO faces (reg_no, name, phone, branch, section, year, embedding) 
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (reg_no, name, data.get('phone'), data.get('branch'), data.get('section'), year, avg_emb.tobytes()))
                    conn.commit()
                    conn.close()
                    del temp_embeddings[reg_no]
                    return jsonify({"success": True, "completed": True})
                
                return jsonify({"success": True, "completed": False})
        
        return jsonify({"success": False, "message": "Face not detected clearly"})
    except Exception as e:
        print(f"Error in process_web_pose: {e}")
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


# ===================
# MARK ATTENDANCE
# ===================

user_states = {}

@app.route('/api/mark_attendance', methods=['POST'])
def mark_attendance():
    try:
        data = request.json
        mode = data.get('mode', '').upper().strip()

        subject = str(data.get('subject', 'General')).strip()
        period = str(data.get('period', '')).strip()
        subj = f"{subject} ({period})"

        f_yr = str(data.get('year')).strip()
        f_br = str(data.get('branch')).strip().upper()
        f_sec = str(data.get('section')).strip().upper()

        # -------- Image Decode --------
        img_bytes = base64.b64decode(data.get('image').split(',')[1])
        frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # -------- Face Detection --------
        pil_img = Image.fromarray(rgb)
        boxes, _ = mtcnn.detect(pil_img)
        final_results = []

        if boxes is None:
            return jsonify({"success": True, "results": []})

        mesh_results = face_mesh.process(rgb)
        face_tensors = mtcnn.extract(pil_img, boxes, None)

        if face_tensors is None or len(face_tensors) == 0:
            return jsonify({"success": True, "results": []})

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT reg_no, name, branch, section, year, embedding FROM faces")
        all_students = cursor.fetchall()

        date_str = datetime.now().strftime("%Y-%m-%d")
        curr_time = datetime.now()

        for i in range(len(boxes)):
            box = boxes[i]
            x1, y1, x2, y2 = box
            face_area = (x2 - x1) * (y2 - y1)

            res = {
                "box": box.tolist(),
                "reg_no": "Unknown",
                "status": "Identifying...",
                "color": "white"
            }

            new_emb = model(
                face_tensors[i].unsqueeze(0).to(device)
            ).detach().cpu().numpy().flatten()

            for reg_no, name, br, sec, yr, emb_blob in all_students:
                if np.linalg.norm(
                    new_emb - np.frombuffer(emb_blob, dtype=np.float32)
                ) < 0.70:

                    res["reg_no"] = reg_no

                    # -------- RULE 1: WRONG CLASS --------
                    if str(yr) != f_yr or str(br) != f_br or str(sec) != f_sec:
                        res.update({
                            "status": f"Wrong Class! You belong to {yr}-{br}-{sec}",
                            "color": "orange"
                        })
                        break

                    # -------- RULE 2A: IN MODE DUPLICATE BLOCK (UNCHANGED) --------
                    if mode == "IN":
                        cursor.execute("""
                            SELECT time FROM attendance
                            WHERE reg_no=? AND date=? AND subject=? AND status='IN'
                        """, (reg_no, date_str, subj))

                        already = cursor.fetchone()
                        if already:
                            res.update({
                                "status": f"You already marked at {already[0]}",
                                "color": "orange"
                            })
                            break

                    # -------- RULE 2B: OUT MODE VALIDATION (FIXED) --------
                    if mode == "OUT":

                        # 1) Fetch latest IN of today (ignore subject)
                        cursor.execute("""
                            SELECT time FROM attendance
                            WHERE reg_no=? AND date=? AND status='IN'
                            ORDER BY id DESC LIMIT 1
                        """, (reg_no, date_str))

                        in_row = cursor.fetchone()
                        if not in_row:
                            res.update({
                                "status": "IN not marked yet",
                                "color": "orange"
                            })
                            break

                        in_time_str = in_row[0]

                        # 2) Gap check (50 minutes)
                        in_time = datetime.strptime(
                            f"{date_str} {in_time_str}",
                            "%Y-%m-%d %H:%M:%S"
                        )

                        diff_min = (curr_time - in_time).total_seconds() / 60

                        if diff_min < 15:
                            wait_min = math.ceil(15 - diff_min)
                            if wait_min < 1:
                                wait_min = 1
                            res.update({
                                "status": f"Too Early! Wait {wait_min} minutes",
                                "color": "orange"
                            })
                            break

                        # 3) Already OUT today?
                        cursor.execute("""
                            SELECT time FROM attendance
                            WHERE reg_no=? AND date=? AND status='OUT'
                            ORDER BY id DESC LIMIT 1
                        """, (reg_no, date_str))

                        out_row = cursor.fetchone()
                        if out_row:
                            res.update({
                                "status": f"Already OUT at {out_row[0]}",
                                "color": "orange"
                            })
                            break

                    # -------- INIT USER STATE --------
                    if reg_no not in user_states:
                        user_states[reg_no] = {
                            "blinked": False,
                            "turned": False,
                            "ear_frames": 0,
                            "base_area": face_area,
                            "base_cx": (x1 + x2) / 2,
                            "base_cy": (y1 + y2) / 2
                        }

                    state = user_states[reg_no]

                    # -------- SPOOF CHECK --------
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2

                    area_ratio = abs(face_area - state["base_area"]) / state["base_area"]
                    center_dist = math.hypot(cx - state["base_cx"], cy - state["base_cy"])

                    if area_ratio > 0.20:
                        res.update({"status": "SPOOF: ZOOMING IMAGE", "color": "red"})
                        del user_states[reg_no]
                        break

                    if center_dist > 30 and area_ratio > 0.15:
                        res.update({"status": "SPOOF: MOVING IMAGE", "color": "red"})
                        del user_states[reg_no]
                        break

                    # -------- FACE MESH --------
                    if not mesh_results.multi_face_landmarks:
                        res.update({"status": "FACE NOT CLEAR", "color": "red"})
                        break

                    fl = mesh_results.multi_face_landmarks[i]

                    # -------- BLINK CHECK --------
                    ear = (
                        abs(fl.landmark[145].y - fl.landmark[159].y) +
                        abs(fl.landmark[374].y - fl.landmark[386].y)
                    ) / 2

                    EAR_THRESHOLD = 0.020
                    EAR_FRAMES = 2

                    if not state["blinked"]:
                        if ear < EAR_THRESHOLD:
                            state["ear_frames"] += 1
                        else:
                            state["ear_frames"] = 0

                        if state["ear_frames"] >= EAR_FRAMES:
                            state["blinked"] = True
                            state["ear_frames"] = 0
                            res.update({"status": "BLINK OK! TURN HEAD", "color": "blue"})
                        else:
                            res.update({"status": "STEP 1: BLINK NOW", "color": "yellow"})
                        break

                    # -------- HEAD TURN CHECK --------
                    nose_x = fl.landmark[1].x * w
                    l_eye_x = fl.landmark[33].x * w
                    r_eye_x = fl.landmark[263].x * w
                    turn_ratio = (nose_x - l_eye_x) / (r_eye_x - l_eye_x)

                    if not state["turned"]:
                        if turn_ratio < 0.30 or turn_ratio > 0.70:
                            state["turned"] = True
                            res.update({"status": "HEAD TURN OK!", "color": "blue"})
                        else:
                            res.update({"status": "STEP 2: TURN HEAD", "color": "cyan"})
                        break

                    # -------- FINAL SUCCESS --------
                    t_str = curr_time.strftime("%H:%M:%S")
                    cursor.execute("""
                        INSERT INTO attendance 
                        (reg_no, name, branch, section, year, subject, date, time, status)
                        VALUES (?,?,?,?,?,?,?,?,?)
                    """, (reg_no, name, br, sec, yr, subj, date_str, t_str, mode))

                    res.update({
                        "status": f"{mode} MARKED: {t_str}",
                        "color": "green"
                    })

                    del user_states[reg_no]
                    break

            final_results.append(res)

        conn.commit()
        conn.close()

        return jsonify({"success": True, "results": final_results})

    except Exception as e:
        print("MARK ERROR:", e)
        return jsonify({"success": False, "message": str(e)})




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
    
    f_year = request.args.get('year', '1').strip()
    f_branch = request.args.get('branch', '').upper().strip()
    f_section = request.args.get('section', '').upper().strip()
    f_subject = request.args.get('subject', '').strip()
    f_period = request.args.get('period', '').strip()
    selected_date = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT reg_no, name FROM faces WHERE year=? AND branch=? AND section=?", 
                   (f_year, f_branch, f_section))
    students = cursor.fetchall()
    
    # సబ్జెక్ట్ ఫిల్టర్ లేకుండా ఆ రోజు డేటా అంతా తీసుకోండి
    cursor.execute("SELECT reg_no, status, time, subject FROM attendance WHERE date = ?", (selected_date,))
    raw_attendance = cursor.fetchall()

    attendance_map = {}
    for reg_no, status, at_time, db_subject in raw_attendance:
        if reg_no is None: continue # ఒకవేళ reg_no ఖాళీగా ఉంటే వదిలేయమని చెప్పడం
        r_no = reg_no.upper().strip()
        
        # Subject Match (Case Insensitive)
        if f_subject.lower() not in db_subject.lower():
            continue
            
        if r_no not in attendance_map:
            attendance_map[r_no] = {'in_time': '-', 'out_time': '-'}
        
        # ఇక్కడ అసలు మ్యాజిక్ ఉంది: ఏ Status ఉన్నా సరే చెక్ చేస్తాం
        st_clean = str(status).strip().upper()
        if st_clean == 'IN':
            attendance_map[r_no]['in_time'] = at_time
        elif st_clean == 'OUT':
            attendance_map[r_no]['out_time'] = at_time

    final_report = []
    p_count, a_count = 0, 0
    
    for reg_no, name in students:
        r_no = reg_no.upper().strip()
        times = attendance_map.get(r_no, {'in_time': '-', 'out_time': '-'})
        
        in_t = times['in_time']
        out_t = times['out_time']
        
        # ఇక్కడ చిన్న మార్పు: ఏదో ఒక టైమ్ ఉన్నా Present అని చూపిద్దాం (మీటింగ్ కోసం తాత్కాలికంగా)
        # మీరు రెండూ ఉండాలనుకుంటే (in_t != '-' and out_t != '-') అని ఉంచండి
        is_present = (in_t != '-' and out_t != '-')
        status_val = "Present" if is_present else "Absent"
        
        if status_val == "Present": p_count += 1
        else: a_count += 1
            
        final_report.append({
            'reg_no': reg_no, 'name': name, 'year': f_year, 'branch': f_branch,
            'section': f_section, 'subject': f"{f_subject} ({f_period})", 
            'status': status_val, 'in_time': in_t, 'out_time': out_t
        })
    
    conn.close()
    return render_template('reports.html', report=final_report, present_count=p_count, absent_count=a_count, 
                           f_year=f_year, f_branch=f_branch, f_section=f_section, f_subject=f_subject, f_period=f_period, date=selected_date
                           )

# ==========================
# 7) UPDATE STUDENT INFO
# ==========================
@app.route('/api/update_student', methods=['POST'])  #ee route only faculty edit cheyyadaniki like student name,register number, year,branch,section,phone
def api_update_student():
    if session.get('role') != 'faculty':
        return jsonify({"success": False, "message": "Unauthorized"}), 403
    
    try:
        data = request.json
        reg_no = data.get('reg_no')
        name = data.get('name')
        branch = data.get('branch')
        section = data.get('section')
        phone = data.get('phone')
        year = data.get('year')

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE faces 
            SET name=?, branch=?, section=?, phone=?, year=?
            WHERE reg_no=?
        """, (name, branch, section, phone, year, reg_no))
        
        conn.commit()
        conn.close()
        return jsonify({"success": True, "message": "Student updated successfully"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})
    

@app.route('/edit_student_search')
def edit_student_search():
    if session.get('role') != 'faculty': return redirect(url_for('login_page'))
    return render_template('edit_student_profile.html')

@app.route('/api/get_student_for_edit/<reg_no>')
def get_student_for_edit(reg_no):
    if session.get('role') != 'faculty':
        return jsonify({"success": False, "message": "Unauthorized"}), 403
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        # ఇక్కడ మీ టేబుల్ లో కాలమ్స్ ఆర్డర్ కరెక్ట్ గా ఉందో లేదో చూడండి
        cursor.execute("SELECT reg_no, name, branch, section, phone, year FROM faces WHERE reg_no = ?", (reg_no.upper(),))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return jsonify({
                "success": True, 
                "data": {
                    "reg_no": row[0], "name": row[1], "branch": row[2],
                    "section": row[3], "phone": row[4], "year": row[5]
                }
            })
        return jsonify({"success": False, "message": "Student Not Found"})
    except Exception as e:
        print(f"Server Error: {e}") # ఇది మీ టెర్మినల్ లో కనిపిస్తుంది
        return jsonify({"success": False, "message": str(e)})

# ==========================
# 8) CLEAR ATTENDANCE LOG
# ==========================
#ee route kavalante remove cheseyochu
@app.route('/api/clear_attendance')
def clear_attendance():
    if session.get('role') != 'faculty':
        return jsonify({"success": False, "message": "Unauthorized"}), 403
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM attendance") # మొత్తం అటెండెన్స్ రికార్డులు క్లియర్ అవుతాయి
        conn.commit()
        conn.close()
        return "Attendance Log Cleared Successfully! Now go back and remark."
    except Exception as e:
        return str(e)


# ==========================
# 8) DOWNLOAD REPORT AS EXCEL
# ==========================

@app.route('/download_report')
def download_report():
    if session.get('role') != 'faculty':
        return redirect(url_for('login_page'))
    
    # 1. ఫిల్టర్స్ తీసుకోవడం
    f_year = request.args.get('year', '1').strip()
    f_branch = request.args.get('branch', '').upper().strip()
    f_section = request.args.get('section', '').upper().strip()
    f_subject = request.args.get('subject', '').strip()
    f_period = request.args.get('period', '').strip()
    selected_date = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 2. డేటా జనరేషన్ (మనం ఇంతకుముందు రాసిన లాజిక్)
    cursor.execute("SELECT reg_no, name FROM faces WHERE year=? AND branch=? AND section=?", (f_year, f_branch, f_section))
    students = cursor.fetchall()
    
    cursor.execute("SELECT reg_no, status, time, subject FROM attendance WHERE date = ?", (selected_date,))
    raw_attendance = cursor.fetchall()

    attendance_map = {}
    for reg_no, status, at_time, db_subject in raw_attendance:
        r_no = reg_no.upper().strip()
        if f_subject.lower() not in db_subject.lower(): continue
        if f_period and f_period not in db_subject: continue
        if r_no not in attendance_map: attendance_map[r_no] = {'in_time': '-', 'out_time': '-'}
        st_clean = str(status).strip().upper()
        if st_clean == 'IN': attendance_map[r_no]['in_time'] = at_time
        elif st_clean == 'OUT': attendance_map[r_no]['out_time'] = at_time

    excel_data = []
    for reg_no, name in students:
        r_no = reg_no.upper().strip()
        times = attendance_map.get(r_no, {'in_time': '-', 'out_time': '-'})
        excel_data.append({
            'Roll No': reg_no,
            'Student Name': name,
            'Status': "Present" if times['in_time'] != '-' and times['out_time'] != '-' else "Absent",
            'IN Time': times['in_time'],
            'OUT Time': times['out_time']
        })
    conn.close()

    # 3. Excel ఫార్మాటింగ్ (Filters + Data)
    df = pd.DataFrame(excel_data)
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # ముందుగా ఫిల్టర్ వివరాలను ఒక చిన్న షీట్ లాగా తయారు చేయడం
        filter_info = [
            ['Attendance Report'],
            ['Date:', selected_date],
            ['Subject:', f"{f_subject} ({f_period})"],
            ['Year:', f"{f_year} Year"],
            ['Branch & Section:', f"{f_branch} - {f_section}"],
            [] # ఒక ఖాళీ వరుస
        ]
        filter_df = pd.DataFrame(filter_info)
        
        # ఫిల్టర్లను మొదట రాయడం (Row 0 నుండి)
        filter_df.to_excel(writer, index=False, header=False, sheet_name='Attendance')
        
        # అసలు డేటాను ఫిల్టర్ల కింద (Row 7 నుండి) రాయడం
        df.to_excel(writer, index=False, startrow=7, sheet_name='Attendance')
    
    output.seek(0)
    filename = f"Report_{f_subject}_{selected_date}.xlsx"
    return send_file(output, as_attachment=True, download_name=filename, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                        )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)