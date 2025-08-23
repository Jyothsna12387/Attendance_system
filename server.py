# In your server.py file

from flask import Flask, request, send_from_directory
from flask_cors import CORS
import numpy as np
import cv2
import torch
import sqlite3
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import os

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize FaceNet models
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def load_known_embeddings():
    if not os.path.exists('data/attendance1.db'):
        print("[ERROR] Database file 'data/attendance1.db' not found.")
        return np.array([]), [], []

    conn = sqlite3.connect('data/attendance1.db')
    c = conn.cursor()
    c.execute("SELECT name, reg_no, embedding FROM faces")
    data = c.fetchall()
    conn.close()

    known_embeddings = []
    known_names = []
    known_regs = []

    for name, reg_no, embedding_blob in data:
        embedding = np.frombuffer(embedding_blob, dtype=np.float32)
        known_embeddings.append(embedding)
        known_names.append(name)
        known_regs.append(reg_no)

    return np.array(known_embeddings), known_names, known_regs

known_embeddings, known_names, known_regs = load_known_embeddings()

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

@app.route('/')
def home():
    return send_from_directory(app.static_folder, 'in_cam.html')

@app.route('/in_cam.html')
def serve_in_cam():
    return send_from_directory(app.static_folder, 'in_cam.html')

@app.route('/out_cam.html')
def serve_out_cam():
    return send_from_directory(app.static_folder, 'out_cam.html')

def process_face_and_return_data(image_file):
    try:
        image_data = image_file.read()
        image_np = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(img)

        recognized_reg_no = "N/A"
        recognized_name = "Unknown"
        box_coords = None
        box_color = 'red'

        if boxes is not None and len(boxes) > 0:
            box_coords = boxes[0].tolist()

            face = mtcnn(img)
            if face is not None:
                face_embedding = model(face.unsqueeze(0).to(device)).detach().cpu().numpy()

                if len(known_embeddings) > 0:
                    similarities = cosine_similarity(face_embedding, known_embeddings)[0]
                    best_match_idx = np.argmax(similarities)

                    if similarities[best_match_idx] > 0.7:
                        recognized_name = known_names[best_match_idx]
                        recognized_reg_no = known_regs[best_match_idx]
                        box_color = 'green'

        return {
            "reg_no": recognized_reg_no,
            "name": recognized_name,
            "box_coords": box_coords,
            "box_color": box_color
        }
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

@app.route('/mark_in', methods=['POST'])
def mark_in():
    data = process_face_and_return_data(request.files['image'])
    if data is None or data['box_color'] == 'red':
        message = "Face not recognized."
        result = {
            "status": "success",
            "message": message,
            "recognized_id": data['reg_no'] if data else "N/A",
            "box_coords": data['box_coords'] if data else None,
            "box_color": 'red'
        }
        return result

    conn = sqlite3.connect('data/attendance1.db')
    c = conn.cursor()
    date_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H:%M:%S")

    c.execute("SELECT type FROM attendance WHERE reg_no = ? AND date = ? ORDER BY time DESC LIMIT 1", (data['reg_no'], date_str))
    last_record = c.fetchone()

    if not last_record or last_record[0] == 'out':
        c.execute("INSERT INTO attendance (reg_no, name, type, time, date) VALUES (?, ?, ?, ?, ?)",
                  (data['reg_no'], data['name'], 'in', time_str, date_str))
        conn.commit()
        message = f"IN attendance marked for {data['reg_no']}."
        print(f"[INFO] IN attendance marked for {data['reg_no']}.")
    else:
        message = f"{data['reg_no']} has not checked out yet."
        data['box_color'] = 'red'
        print(f"[INFO] {data['reg_no']} has not checked out yet.")
    
    conn.close()
    
    result = {
        "status": "success",
        "message": message,
        "recognized_id": data['reg_no'],
        "box_coords": data['box_coords'],
        "box_color": data['box_color']
    }
    return result

@app.route('/mark_out', methods=['POST'])
def mark_out():
    data = process_face_and_return_data(request.files['image'])
    if data is None or data['box_color'] == 'red':
        message = "Face not recognized."
        result = {
            "status": "success",
            "message": message,
            "recognized_id": data['reg_no'] if data else "N/A",
            "box_coords": data['box_coords'] if data else None,
            "box_color": 'red'
        }
        return result

    conn = sqlite3.connect('data/attendance1.db')
    c = conn.cursor()
    date_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H:%M:%S")

    c.execute("SELECT type FROM attendance WHERE reg_no = ? AND date = ? ORDER BY time DESC LIMIT 1", (data['reg_no'], date_str))
    last_record = c.fetchone()
    
    if last_record and last_record[0] == 'in':
        c.execute("INSERT INTO attendance (reg_no, name, type, time, date) VALUES (?, ?, ?, ?, ?)",
                  (data['reg_no'], data['name'], 'out', time_str, date_str))
        conn.commit()
        message = f"OUT attendance marked for {data['reg_no']}."
        print(f"[INFO] OUT attendance marked for {data['reg_no']}.")
    else:
        message = f"{data['reg_no']} has not checked in."
        data['box_color'] = 'red'
        print(f"[INFO] {data['reg_no']} has not checked in.")
    
    conn.close()

    result = {
        "status": "success",
        "message": message,
        "recognized_id": data['reg_no'],
        "box_coords": data['box_coords'],
        "box_color": data['box_color']
    }
    return result

if __name__ == '__main__':
    app.run(debug=True)