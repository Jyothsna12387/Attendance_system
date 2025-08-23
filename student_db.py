import os
import cv2
import torch
import numpy as np
import sqlite3
from facenet_pytorch import MTCNN, InceptionResnetV1
from datetime import datetime

# Initialize FaceNet model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, keep_all=False, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Connect to database
conn = sqlite3.connect('data/attendance1.db')
cursor = conn.cursor()

# Input student details
reg_no = input("Enter Registration Number: ")
name = input("Enter Student Name: ")

# Check for duplicate reg_no
cursor.execute("SELECT * FROM faces WHERE reg_no = ?", (reg_no,))
if cursor.fetchone():
    print("[ERROR] Registration number already exists.")
    conn.close()
    exit()

# Load existing face embeddings
cursor.execute("SELECT embedding FROM faces")
existing_embeddings = [np.frombuffer(row[0], dtype=np.float32) for row in cursor.fetchall()]

# Start video capture
cap = cv2.VideoCapture(0)
print(f"[INFO] Capturing 50 face images for {name}. Please stay in front of the camera.")

captured_embeddings = []
frame_count = 0
max_images = 50
checked_similarity_once = False
proceed_with_registration = True

while frame_count < max_images:
    ret, frame = cap.read()
    if not ret:
        continue

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face = mtcnn(img)

    if face is not None:
        with torch.no_grad():
            embedding = model(face.unsqueeze(0).to(device)).cpu().numpy().flatten()

        # 🔁 Check for similar face only once
        if not checked_similarity_once:
            for emb in existing_embeddings:
                dist = np.linalg.norm(embedding - emb)
                if dist < 0.7:
                    print("[⚠️ WARNING] This face appears similar to a previously registered student.")

                    # ✅ Ask user up to 3 times
                    attempts = 0
                    while attempts < 3:
                        response = input("Proceed with registration? (y/n): ").strip().lower()
                        if response == 'y':
                            proceed_with_registration = True
                            break
                        elif response == 'n':
                            proceed_with_registration = False
                            break
                        else:
                            print("[INFO] Please enter 'y' or 'n'.")
                            attempts += 1

                    if not proceed_with_registration:
                        print("[❌ CANCELED] Registration aborted by user.")
                        cap.release()
                        cv2.destroyAllWindows()
                        conn.close()
                        exit()
                    else:
                        print("[✅ CONTINUING] Proceeding with registration.")
                    break  # stop checking after one similar match
            checked_similarity_once = True

        if proceed_with_registration:
            captured_embeddings.append(embedding)
            frame_count += 1
            print(f"[INFO] Captured image {frame_count}/{max_images}")

    # Show webcam window
    cv2.imshow("Registration Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if len(captured_embeddings) < 10:
    print("[ERROR] Not enough faces captured. Try again.")
    conn.close()
    exit()

# Average embeddings and insert into DB
avg_embedding = np.mean(captured_embeddings, axis=0).astype(np.float32).tobytes()
cursor.execute("INSERT INTO faces (reg_no, name, embedding) VALUES (?, ?, ?)", (reg_no, name, avg_embedding))
conn.commit()
conn.close()

print(f"[✅ SUCCESS] Registered {name} with {frame_count} face samples.")
