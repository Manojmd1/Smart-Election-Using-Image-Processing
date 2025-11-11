import cv2
import pickle
import numpy as np
import os
import tkinter as tk
from tkinter import messagebox
import subprocess

class FaceCaptureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Add Your Adhar")
        self.root.geometry("600x600")  # Increased the window size

        # Set background image
        self.background_image = tk.PhotoImage(file="image1.png")
        self.background_label = tk.Label(root, image=self.background_image)
        self.background_label.pack(fill="both", expand=True)

        # Create a frame to hold the GUI fields
        self.frame = tk.Frame(root, bg="#413839")
        self.frame.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.4, relheight=0.4)  # Increased the frame size

        self.aadhar_label = tk.Label(self.frame, text="Enter your aadhar number", font=("Arial", 14, "bold"), fg="#ff0000", bg="#ffffff")  # blue text
        self.aadhar_label.pack(pady=10)

        self.aadhar_entry = tk.Entry(self.frame, width=40, font=("Arial", 12), fg="#333", bg="#ffffff")  # dark gray text
        self.aadhar_entry.pack(pady=10)

        self.capture_button = tk.Button(self.frame, text="Capture Face", command=self.capture_face, font=("Arial", 12, "bold"), fg="#fff", bg="#0000ff")  # green button
        self.capture_button.pack(pady=10)

        self.run_button = tk.Button(self.frame, text="Vote Here", command=self.run_script, font=("Arial", 12, "bold"), fg="#fff", bg="#ff0000")  # green button
        self.run_button.pack(pady=10)

        self.status_label = tk.Label(self.frame, text="", font=("Arial", 12), fg="#666", bg="#413839")  # dark gray text
        self.status_label.pack(pady=10)

    def capture_face(self):
        aadhar_number = self.aadhar_entry.get()
        if not aadhar_number.isdigit() or len(aadhar_number) != 12:
            messagebox.showerror("Error", "Enter Valid aadhar number")
            return

        video = cv2.VideoCapture(0)
        facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces_data = []

        i = 0
        framesTotal = 51
        captureAfterFrame = 2

        while True:
            ret, frame = video.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = facedetect.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                crop_img = frame[y:y+h, x:x+w]
                resized_img = cv2.resize(crop_img, (50, 50))
                if len(faces_data) <= framesTotal and i % captureAfterFrame == 0:
                    faces_data.append(resized_img)
                i += 1
                cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)

            cv2.imshow('frame', frame)
            k = cv2.waitKey(1)
            if k == ord('q') or len(faces_data) >= framesTotal:
                break

        video.release()
        cv2.destroyAllWindows()

        faces_data = np.asarray(faces_data)
        faces_data = faces_data.reshape((framesTotal, -1))

        if 'names.pkl' not in os.listdir('data/'):
            names = [int(aadhar_number)] * framesTotal
            with open('data/names.pkl', 'wb') as f:
                pickle.dump(names, f)
        else:
            with open('data/names.pkl', 'rb') as f:
                names = pickle.load(f)
            names = names + [int(aadhar_number)] * framesTotal
            with open('data/names.pkl', 'wb') as f:
                pickle.dump(names, f)

        if 'faces_data.pkl' not in os.listdir('data/'):
            with open('data/faces_data.pkl', 'wb') as f:
                pickle.dump(faces_data, f)
        else:
            with open('data/faces_data.pkl', 'rb') as f:
                faces = pickle.load(f)
            faces = np.append(faces, faces_data, axis=0)
            with open('data/faces_data.pkl', 'wb') as f:
                pickle.dump(faces, f)

        self.status_label.config(text="Face captured successfully!", fg="#0000ff")  # green text

    def run_script(self):
        # Run another Python file
        script_path = "give_vote.py"
        subprocess.run(["python", script_path])

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceCaptureApp(root)
    root.mainloop()