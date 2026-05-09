import tkinter as tk
from tkinter import filedialog, Label, Button
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO
import torch

class ObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Object Detector GUI")
        self.root.geometry("1600x1000")
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Running inference on: {self.device}")

        # Load the best trained model
        self.model = YOLO('best_obstacle_model.pt')
        
        self.class_thresholds = {
            'person': (60.0, 40.0),
            'car': (30.0, 20.0),
            'animal': (70.0, 50.0),
        }
        self.default_thresh = (15.0, 5.0)

        self.video_label = Label(root)
        self.video_label.pack(pady=20)

        self.btn_frame = tk.Frame(root)
        self.btn_frame.pack(pady=10)

        self.btn_camera = Button(self.btn_frame, text="Open Camera", command=self.start_camera, width=20, height=2, bg="#4CAF50", fg="white")
        self.btn_camera.grid(row=0, column=0, padx=10)

        self.btn_upload = Button(self.btn_frame, text="Upload Video", command=self.upload_video, width=20, height=2, bg="#2196F3", fg="white")
        self.btn_upload.grid(row=0, column=1, padx=10)

        self.btn_stop = Button(self.btn_frame, text="Stop", command=self.stop_video, width=20, height=2, bg="#f44336", fg="white")
        self.btn_stop.grid(row=0, column=2, padx=10)
        self.cap = None
        self.is_running = False
        self.frame_count = 0

    def start_camera(self):
        self.stop_video()
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            return
        self.is_running = True
        self.update_frame()

    def upload_video(self):
        self.stop_video()
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov;*.mkv")])
        if file_path:
            self.cap = cv2.VideoCapture(file_path)
            self.is_running = True
            self.update_frame()

    def stop_video(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.video_label.config(image='')

    def update_frame(self):
        if self.is_running and self.cap:
            ret, frame = self.cap.read()
            if ret:
                # Perform inference using the full image size and selected device
                max_dim = max(frame.shape[:2])
                results = self.model(frame, verbose=False, conf=0.25, imgsz=max_dim, device=self.device) 
                annotated_frame = results[0].plot()

                # Distance estimation
                if results[0].boxes:
                    frame_height = frame.shape[0]
                    frame_width = frame.shape[1]
                    frame_area = frame_width * frame_height
                    class_names = results[0].names

                    for box in results[0].boxes:
                        x1, y1, x2, y2 = box.xyxy[0]
                        cls_id = int(box.cls[0])
                        cls_name = class_names.get(cls_id, 'unknown')
                        
                        box_width = float(x2 - x1)
                        box_height = float(y2 - y1)
                        box_area = box_width * box_height
                        occupancy_pct = (box_area / frame_area) * 100
                        
                        danger_t, caution_t = self.class_thresholds.get(cls_name, self.default_thresh)
                        
                        if occupancy_pct > danger_t:
                            cv2.putText(annotated_frame, f"DANGER {occupancy_pct:.1f}%", (int(x1), int(y1) - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        elif occupancy_pct > caution_t:
                            cv2.putText(annotated_frame, f"CAUTION {occupancy_pct:.1f}%", (int(x1), int(y1) - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                        else:
                            cv2.putText(annotated_frame, f"SAFE {occupancy_pct:.1f}%", (int(x1), int(y1) - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                # Resize frame to fit the GUI window
                h, w = rgb_image.shape[:2]
                max_width = 1580
                max_height = 850  # Limit height to keep buttons visible
                scale = min(max_width / w, max_height / h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                rgb_image = cv2.resize(rgb_image, (new_w, new_h))

                img = Image.fromarray(rgb_image)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
                self.root.after(1, self.update_frame)
            else:
                self.stop_video()

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.mainloop()
