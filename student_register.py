import os
import cv2
import torch
import numpy as np
import sqlite3
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox
from facenet_pytorch import MTCNN, InceptionResnetV1
import sys

# ==========================
# 1) CONFIG & MODELS
# ==========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# MTCNN: ఫేస్ డిటెక్షన్ కోసం
mtcnn = MTCNN(image_size=160, margin=0, keep_all=False, device=device)
# InceptionResnet: ఫేస్ ఎంబెడ్డింగ్స్ (Biometric data) కోసం
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# డేటాబేస్ ఫోల్డర్ చెక్ చేయడం
if not os.path.exists(os.path.join(BASE_DIR, "data")):
    os.makedirs(os.path.join(BASE_DIR, "data"))

DB_PATH = os.path.join(BASE_DIR, "data", "attendance1.db")
POSES = ["Look straight", "Turn face LEFT", "Turn face RIGHT", "Look UP", "Look DOWN"]

# Database Initialization
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # ఫోన్ కాలమ్ (phone) యాడ్ చేయబడింది
    cursor.execute('''CREATE TABLE IF NOT EXISTS faces 
                     (reg_no TEXT PRIMARY KEY, name TEXT, phone TEXT, embedding BLOB)''')
    conn.commit()
    conn.close()

init_db()

class StudentRegistrationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Student Enrollment Portal")
        self.root.geometry("1000x750")
        self.root.configure(bg="#F0F2F5")

        self.cap = None
        self.captured_embeddings = []
        self.current_pose_idx = 0
        
        self.main_container = tk.Frame(self.root, bg="#F0F2F5")
        self.main_container.pack(expand=True, fill="both")

        # సర్వర్ నుండి డేటా వస్తే (Name, Reg, Phone)
        if len(sys.argv) > 3:
            self.student_name = sys.argv[1]
            self.student_reg = sys.argv[2]
            self.student_phone = sys.argv[3]
            self.setup_camera_page()
        else:
            self.show_form_page()

    # ==========================
    # 2) REGISTRATION FORM
    # ==========================
    def show_form_page(self):
        self.form_frame = tk.Frame(self.main_container, bg="white", padx=40, pady=30, 
                                 bd=0, highlightthickness=1, highlightbackground="#DCDFE6")
        self.form_frame.place(relx=0.5, rely=0.5, anchor="center")

        tk.Label(self.form_frame, text="Student Registration", font=("Segoe UI", 22, "bold"), 
                 bg="white", fg="#1A73E8").pack(pady=(0, 20))

        def add_input(label):
            tk.Label(self.form_frame, text=label, font=("Segoe UI", 10, "bold"), bg="white", fg="#606266").pack(anchor="w", pady=(5, 0))
            entry = tk.Entry(self.form_frame, font=("Segoe UI", 12), width=35, bd=0, 
                             highlightthickness=1, highlightbackground="#DCDFE6")
            entry.pack(pady=5, ipady=8)
            return entry

        self.name_ent = add_input("Full Name")
        self.reg_ent = add_input("Registration Number (UPPERCASE)")
        self.phone_ent = add_input("Phone Number (for Alerts)")
        self.pass_ent = add_input("Security Password (Syncs with Reg No)")

        self.reg_ent.bind("<KeyRelease>", lambda e: self.sync_password())

        btn_frame = tk.Frame(self.form_frame, bg="white")
        btn_frame.pack(pady=25, fill="x")

        tk.Button(btn_frame, text="Start Capture", command=self.validate_and_switch, 
                  bg="#1A73E8", fg="white", font=("Segoe UI", 11, "bold"), 
                  bd=0, cursor="hand2", width=15, height=2).pack(side="left", padx=5)

        tk.Button(btn_frame, text="Cancel", command=self.root.destroy, 
                  bg="#F56C6C", fg="white", font=("Segoe UI", 11, "bold"), 
                  bd=0, cursor="hand2", width=10, height=2).pack(side="left", padx=5)

    def sync_password(self):
        val = self.reg_ent.get().upper()
        self.pass_ent.delete(0, tk.END)
        self.pass_ent.insert(0, val)

    def validate_and_switch(self):
        self.student_name = self.name_ent.get().strip()
        self.student_reg = self.reg_ent.get().strip().upper()
        self.student_phone = self.phone_ent.get().strip()

        if not self.student_name or not self.student_reg or not self.student_phone:
            messagebox.showerror("Error", "All fields are mandatory!")
            return

        if len(self.student_phone) < 10 or not self.student_phone.isdigit():
            messagebox.showerror("Error", "Enter a valid 10-digit phone number!")
            return

        # Duplicate Check
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM faces WHERE reg_no = ?", (self.student_reg,))
        if cursor.fetchone():
            messagebox.showwarning("Warning", "This Registration Number is already enrolled!")
            conn.close()
            return
        conn.close()

        self.form_frame.destroy()
        self.setup_camera_page()

    # ==========================
    # 3) CAMERA CAPTURE
    # ==========================
    def setup_camera_page(self):
        self.camera_frame = tk.Frame(self.main_container, bg="#F0F2F5")
        self.camera_frame.pack(expand=True, fill="both")

        tk.Label(self.camera_frame, text=f"Registering: {self.student_name}", font=("Segoe UI", 16, "bold"), 
                 bg="#F0F2F5", fg="#333").pack(pady=10)

        self.video_label = tk.Label(self.camera_frame, bg="black", width=640, height=480)
        self.video_label.pack(pady=5)

        self.info_box = tk.Frame(self.camera_frame, bg="#E8F0FE", padx=20, pady=10)
        self.info_box.pack(pady=15)
        
        self.pose_instr = tk.Label(self.info_box, text="Initializing Camera...", font=("Segoe UI", 14, "bold"), 
                                   bg="#E8F0FE", fg="#1967D2")
        self.pose_instr.pack()

        self.cap = cv2.VideoCapture(0)
        self.capture_loop()

    def capture_loop(self):
        if self.current_pose_idx >= len(POSES):
            self.finalize_registration()
            return

        ret, frame = self.cap.read()
        if not ret: return

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # UI Instructions
        pose_goal = POSES[self.current_pose_idx]
        self.pose_instr.config(text=f"ACTION: {pose_goal.upper()}")

        # Face Detection
        boxes, _ = mtcnn.detect(rgb)

        if boxes is not None:
            for box in boxes:
                b = box.astype(int)
                cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (26, 115, 232), 2)
            
            # Face Capture Logic
            face_tensor = mtcnn(rgb)
            if face_tensor is not None:
                with torch.no_grad():
                    emb = model(face_tensor.unsqueeze(0).to(device)).cpu().numpy().flatten()
                self.captured_embeddings.append(emb)
                self.current_pose_idx += 1

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((640, 480))
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
        self.root.after(30, self.capture_loop)

    def finalize_registration(self):
        # సగటు ఎంబెడ్డింగ్ తీసుకోవడం (Accuracy కోసం)
        avg_emb = np.mean(self.captured_embeddings, axis=0).astype(np.float32).tobytes()
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        # నేమ్, రెగ్, ఫోన్ మరియు ఎంబెడ్డింగ్ సేవ్ చేయడం
        cursor.execute("INSERT INTO faces (reg_no, name, phone, embedding) VALUES (?, ?, ?, ?)", 
                      (self.student_reg, self.student_name, self.student_phone, avg_emb))
        conn.commit()
        conn.close()

        if self.cap: self.cap.release()
        
        messagebox.showinfo("Success", f"Student {self.student_name} registered successfully with Phone: {self.student_phone}")
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = StudentRegistrationApp(root)
    root.mainloop()