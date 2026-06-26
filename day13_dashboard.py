"""
Day 13: Web-Based Face Mask Detection Dashboard
- Flask backend captures webcam feed and performs face mask detection
- Streams MJPEG live video on /video_feed
- Exposes /stats API for detection metrics
- Supports /start, /stop, and /save_log controls
"""

from collections import deque
from pathlib import Path
import time
import threading
import csv

import cv2
import numpy as np
from flask import Flask, Response, jsonify, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


app = Flask(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "model"

FACE_PROTOTXT = MODEL_DIR / "deploy.prototxt"
FACE_MODEL = MODEL_DIR / "res10_300x300_ssd_iter_140000.caffemodel"
MASK_MODEL = MODEL_DIR / "mask_detector.h5"
LOG_FILE = PROJECT_ROOT / "dashboard_log.csv"


class DashboardCamera:
    def __init__(self):
        self.face_net = cv2.dnn.readNetFromCaffe(str(FACE_PROTOTXT), str(FACE_MODEL))
        self.mask_model = load_model(str(MASK_MODEL))

        self.capture = None
        self.thread = None
        self.running = False
        self.lock = threading.Lock()

        self.last_frame = None
        self.total_detections = 0
        self.violations = 0
        self.start_time = None
        self.recent_alerts = deque(maxlen=10)

    def start(self):
        with self.lock:
            if self.running:
                return True, "Camera already running"

            self.capture = cv2.VideoCapture(0)
            if not self.capture.isOpened():
                self.capture = None
                return False, "Unable to open webcam"

            self.running = True
            self.start_time = time.time()
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()
            return True, "Camera started"

    def stop(self):
        with self.lock:
            if not self.running:
                return False, "Camera is not running"

            self.running = False

        if self.thread is not None:
            self.thread.join(timeout=2.0)
            self.thread = None

        with self.lock:
            if self.capture is not None:
                self.capture.release()
                self.capture = None

        return True, "Camera stopped"

    def _capture_loop(self):
        while True:
            with self.lock:
                if not self.running or self.capture is None:
                    break
                capture = self.capture

            ret, frame = capture.read()
            if not ret:
                time.sleep(0.1)
                continue

            annotated = self._annotate_frame(frame)
            _, encoded = cv2.imencode('.jpg', annotated)
            with self.lock:
                self.last_frame = encoded.tobytes()

            time.sleep(0.02)

    def _annotate_frame(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104.0, 177.0, 123.0], swapRB=False, crop=False)
        self.face_net.setInput(blob)
        detections = self.face_net.forward()

        frame_with_boxes = frame.copy()
        frame_faces = 0
        frame_violations = 0

        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < 0.5:
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            startX, startY, endX, endY = box.astype('int')
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)

            if endX <= startX or endY <= startY:
                continue

            face = frame[startY:endY, startX:endX]
            if face.size == 0:
                continue

            face_blob = cv2.resize(face, (224, 224)).astype('float32')
            face_blob = preprocess_input(face_blob)
            face_blob = np.expand_dims(face_blob, axis=0)

            preds = self.mask_model.predict(face_blob, verbose=0)
            class_idx = int(np.argmax(preds[0]))
            class_label = "With Mask" if class_idx == 0 else "Without Mask"
            class_color = (0, 255, 0) if class_idx == 0 else (0, 0, 255)
            score = float(preds[0][class_idx])

            frame_faces += 1
            self.total_detections += 1
            if class_idx == 1:
                frame_violations += 1
                self.violations += 1
                alert = f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {class_label} ({score:.1%})"
                with self.lock:
                    self.recent_alerts.appendleft(alert)

            cv2.rectangle(frame_with_boxes, (startX, startY), (endX, endY), class_color, 2)
            label_text = f"{class_label}: {score:.1%}"
            text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame_with_boxes, (startX, startY - text_size[1] - 10),
                          (startX + text_size[0] + 10, startY), class_color, -1)
            cv2.putText(frame_with_boxes, label_text, (startX + 5, startY - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if frame_faces > 0:
            border_color = (0, 255, 0) if frame_violations == 0 else (0, 0, 255)
            cv2.rectangle(frame_with_boxes, (0, 0), (w - 1, h - 1), border_color, 8)

        return frame_with_boxes

    def get_stats(self):
        with self.lock:
            total = self.total_detections
            violations = self.violations
            alerts = list(self.recent_alerts)
            running = self.running
            elapsed = time.time() - self.start_time if self.start_time and running else 0.0

        compliance_rate = 100.0
        if total > 0:
            compliance_rate = max(0.0, 100.0 * (1.0 - violations / float(total)))

        return {
            "total_detections": total,
            "violations": violations,
            "compliance_rate": round(compliance_rate, 1),
            "session_duration": round(elapsed, 1),
            "recent_alerts": alerts,
            "camera_running": running
        }

    def get_frame(self):
        with self.lock:
            return self.last_frame

    def save_log(self):
        stats = self.get_stats()
        header = ["timestamp", "total_detections", "violations", "compliance_rate", "session_duration"]
        write_header = not LOG_FILE.exists()
        with open(LOG_FILE, "a", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            if write_header:
                writer.writerow(header)
            writer.writerow([
                time.strftime('%Y-%m-%d %H:%M:%S'),
                stats["total_detections"],
                stats["violations"],
                stats["compliance_rate"],
                stats["session_duration"]
            ])

        return True


camera = DashboardCamera()


def generate_frames():
    while True:
        frame = camera.get_frame()
        if frame is None:
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Camera stopped or no feed", (24, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            _, jpg = cv2.imencode('.jpg', placeholder)
            frame = jpg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.1)


@app.route('/')
def index():
    return send_from_directory(PROJECT_ROOT, 'day13_dashboard.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stats')
def stats():
    return jsonify(camera.get_stats())


@app.route('/start')
def start_camera():
    success, message = camera.start()
    return jsonify({"success": success, "message": message})


@app.route('/stop')
def stop_camera():
    success, message = camera.stop()
    return jsonify({"success": success, "message": message})


@app.route('/save_log')
def save_log():
    camera.save_log()
    return jsonify({"success": True, "message": "Log saved"})


if __name__ == '__main__':
    if not FACE_PROTOTXT.exists() or not FACE_MODEL.exists() or not MASK_MODEL.exists():
        print("ERROR: Required model files are missing in the model directory.")
        print(f"Expected: {FACE_PROTOTXT}, {FACE_MODEL}, {MASK_MODEL}")
    else:
        app.run(host='0.0.0.0', port=5000, threaded=True)
