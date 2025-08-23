import os
import cv2
import torch
import numpy as np
import sqlite3
from facenet_pytorch import MTCNN, InceptionResnetV1
from datetime import datetime
import pyttsx3
from sklearn.metrics.pairwise import cosine_similarity

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize FaceNet models
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Initialize TTS
engine = pyttsx3.init()

# Load database
conn = sqlite3.connect('data/attendance1.db')
c = conn.cursor()

# Load face embeddings from 'faces' table
c.execute("SELECT name, reg_no, embedding FROM faces")
data = c.fetchall()
known_embeddings = []
known_names = []
known_regs = []

for name, reg_no, embedding_blob in data:
    embedding = np.frombuffer(embedding_blob, dtype=np.float32)
    known_embeddings.append(embedding)
    known_names.append(name)
    known_regs.append(reg_no)

# Convert embeddings to NumPy array
known_embeddings = np.array(known_embeddings)

# Start webcam
cap = cv2.VideoCapture(0)
recognized = False

print("[INFO] Showing webcam. It will auto-exit after marking attendance.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face = mtcnn(img)

    if face is not None:
        face_embedding = model(face.unsqueeze(0).to(device)).detach().cpu().numpy()

        # Compare with known embeddings
        similarities = cosine_similarity(face_embedding, known_embeddings)[0]
        best_match_idx = np.argmax(similarities)
        if similarities[best_match_idx] > 0.6:  # threshold
            name = known_names[best_match_idx]
            reg_no = known_regs[best_match_idx]

            # Check if already marked
            c.execute("SELECT * FROM attendance WHERE reg_no = ?", (reg_no,))
            if c.fetchone():
                print(f"[INFO] Hello {name}, your attendance has already been marked.")
                engine.say(f"Hello {name}, your attendance is already marked.")
                engine.runAndWait()
                break
            else:
                # Mark attendance
                now = datetime.now()
                time_str = now.strftime("%H:%M:%S")
                date_str = now.strftime("%Y-%m-%d")

                c.execute("INSERT INTO attendance (name, reg_no, time, date, status) VALUES (?, ?, ?, ?, ?)",
                          (name, reg_no, time_str, date_str, 'Present'))
                conn.commit()

                print(f"[INFO] Hello {name}, your attendance has been marked.")
                engine.say(f"Hello {name}, your attendance has been marked.")
                engine.runAndWait()
                break
        else:
            # Unknown face
            cv2.putText(frame, "Unknown", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Smart Attendance", frame)

    # Remove 'press q to quit' condition, exit only on recognition
    if cv2.waitKey(1) == 27:  # optional ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
conn.close()
