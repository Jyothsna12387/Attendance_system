import os
import cv2
import torch
import numpy as np
import sqlite3
import time
import sys
import threading
import queue
import winsound  
from datetime import datetime
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity
import mediapipe as mp
import pyttsx3

# ================= VOICE SYSTEM =================
voice_queue = queue.Queue()
is_speaking = False 

def voice_worker():
    global is_speaking
    v_engine = pyttsx3.init()
    v_engine.setProperty('rate', 145)
    while True:
        msg = voice_queue.get()
        if msg is None: break
        try:
            is_speaking = True
            v_engine.say(msg)
            v_engine.runAndWait()
            is_speaking = False
        except: 
            is_speaking = False
        voice_queue.task_done()

threading.Thread(target=voice_worker, daemon=True).start()

def speak(text):
    voice_queue.put(text)

def play_success_beep():
    winsound.Beep(1000, 200)

# ================= CONFIG & MODELS =================
RECOGNITION_THRESHOLD = 0.65
EAR_THRESHOLD = 0.24 

if len(sys.argv) < 2:
    print("Usage: python main.py IN|OUT")
    sys.exit()

ATTENDANCE_MODE = sys.argv[1].upper()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(keep_all=True, device=device) 
model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=5)

# ================= DB LOAD =================
conn = sqlite3.connect("data/attendance1.db", check_same_thread=False)
c = conn.cursor()
c.execute("SELECT name, reg_no, embedding FROM faces")
rows = c.fetchall()
known_names, known_regs, known_embeds = [], [], []
for n, r, e in rows:
    known_names.append(n); known_regs.append(r); known_embeds.append(np.frombuffer(e, dtype=np.float32))
known_embeds = np.array(known_embeds)

blink_data = {} 

def get_ear(landmarks, h, w):
    def dist(p1, p2):
        return np.linalg.norm(np.array([landmarks[p1].x*w, landmarks[p1].y*h]) - 
                              np.array([landmarks[p2].x*w, landmarks[p2].y*h]))
    le = (dist(159, 145) + dist(158, 144)) / (2.0 * dist(33, 133))
    re = (dist(386, 374) + dist(385, 373)) / (2.0 * dist(362, 263))
    return (le + re) / 2.0

# ================= CAMERA LOOP =================
cap = cv2.VideoCapture(0)
print(f"--- System Started: {ATTENDANCE_MODE} Mode ---")

while True:
    ret, frame = cap.read()
    if not ret: break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, _ = mtcnn.detect(rgb)
    mesh_results = face_mesh.process(rgb)
    all_lms = mesh_results.multi_face_landmarks if mesh_results.multi_face_landmarks else []

    faces_in_frame = 0
    marked_in_frame = 0

    if boxes is not None:
        faces_in_frame = len(boxes)
        faces = mtcnn.extract(rgb, boxes, save_path=None)
        
        if faces is not None:
            with torch.no_grad():
                embeddings = model(faces.to(device)).cpu().numpy()

            for i, box in enumerate(boxes):
                sims = cosine_similarity([embeddings[i]], known_embeds)[0]
                idx = np.argmax(sims)
                x1, y1, x2, y2 = map(int, box)
                
                if sims[idx] > RECOGNITION_THRESHOLD:
                    name, reg = known_names[idx], known_regs[idx]
                    if reg not in blink_data:
                        blink_data[reg] = {"closed": False, "marked": False}

                    if blink_data[reg]["marked"]:
                        marked_in_frame += 1
                        color = (0, 255, 0)
                        status_text = "Status: Marked"
                    else:
                        color = (0, 255, 255)
                        status_text = "Blink to Mark"
                        current_ear = 0.35
                        for lms in all_lms:
                            cx, cy = lms.landmark[1].x * w, lms.landmark[1].y * h
                            if x1 < cx < x2 and y1 < cy < y2:
                                current_ear = get_ear(lms.landmark, h, w)
                                break

                        if current_ear < EAR_THRESHOLD:
                            blink_data[reg]["closed"] = True
                        elif current_ear > (EAR_THRESHOLD + 0.02) and blink_data[reg]["closed"]:
                            d, t_now = datetime.now().strftime("%Y-%m-%d"), datetime.now().strftime("%H:%M:%S")
                            c.execute("INSERT INTO attendance (name, reg_no, status, date, time) VALUES (?,?,?,?,?)",
                                      (name, reg, ATTENDANCE_MODE, d, t_now))
                            conn.commit()
                            
                            success_msg = f"Hello {name}, your attendance has been marked"
                            print(f"[SUCCESS] {success_msg}")
                            threading.Thread(target=play_success_beep, daemon=True).start()
                            
                            blink_data[reg]["marked"] = True
                            speak(success_msg)
                            blink_data[reg]["closed"] = False
                            # Mark ayina frame lone color green avvali
                            color = (0, 255, 0)
                            status_text = "Status: Marked"
                            marked_in_frame += 1

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.putText(frame, status_text, (x1, y2+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, "Unknown", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Attendance System", frame)

    # ================= AUTO CLOSE LOGIC =================
    if faces_in_frame > 0 and marked_in_frame == faces_in_frame:
        # Voice purthiga ayyevaraku wait chesthu frame ni refresh chesthunnam
        if voice_queue.empty() and not is_speaking:
            print("\nAttendance and Voice both completed. Closing...")
            time.sleep(1) 
            break

    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()
conn.close()