"""
Day 11: Real-Time Mask Detection with Logging

- CSV Logging: logs each no-mask detection to `alerts/violations_log.csv`
- Snapshot Capture: saves cropped face images to `alerts/snapshots/` as `violation_YYYYMMDD_HHMMSS.jpg`
- Session Summary: tracks total faces, violations, compliance; saved to `alerts/session_summary.txt` on quit
"""

from pathlib import Path
import cv2
import numpy as np
import time
import csv
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


class RealtimeMaskLogger:
    def __init__(self, prototxt_path, caffemodel_path, mask_model_path, alerts_dir: Path, cooldown=5.0):
        self.face_net = cv2.dnn.readNetFromCaffe(str(prototxt_path), str(caffemodel_path))
        self.mask_model = load_model(str(mask_model_path))

        self.class_names = ["With Mask", "Without Mask"]
        self.colors = {0: (0, 255, 0), 1: (0, 0, 255)}

        self.alerts_dir = Path(alerts_dir)
        self.snapshots_dir = self.alerts_dir / "snapshots"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

        self.log_csv = self.alerts_dir / "violations_log.csv"
        if not self.log_csv.exists():
            with open(self.log_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'confidence', 'alert_triggered', 'snapshot'])

        # Session stats
        self.total_faces = 0
        self.total_violations = 0

        # Alert cooldown
        self.cooldown = float(cooldown)
        self.last_alert_time = 0.0

        # Try to init pygame for optional audible alerts
        self.pygame_available = False
        try:
            import pygame
            self.pygame = pygame
            sr = 44100
            try:
                pygame.mixer.init(frequency=sr, size=-16, channels=1)
            except Exception:
                pygame.mixer.pre_init(sr, -16, 1)
                pygame.mixer.init()
            self.alarm_sound = self._generate_alarm_sound(duration=0.6, freq=880, sr=sr)
            self.pygame_available = True
        except Exception:
            self.pygame_available = False

    def _generate_alarm_sound(self, duration=0.6, freq=880, sr=44100):
        try:
            t = np.linspace(0, duration, int(sr * duration), False)
            tone = 0.5 * np.sin(2 * np.pi * freq * t)
            audio = (tone * 32767).astype(np.int16)
            try:
                sound = self.pygame.sndarray.make_sound(audio)
            except Exception:
                stereo = np.column_stack([audio, audio])
                sound = self.pygame.sndarray.make_sound(stereo)
            return sound
        except Exception:
            return None

    def detect_faces(self, frame, confidence_threshold=0.5):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104.0, 177.0, 123.0], swapRB=False, crop=False)
        self.face_net.setInput(blob)
        detections = self.face_net.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < confidence_threshold:
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            startX, startY, endX, endY = box.astype(int)
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)
            faces.append((startX, startY, endX, endY))

        return faces

    def predict_mask(self, face_roi):
        face_resized = cv2.resize(face_roi, (224, 224))
        face_blob = preprocess_input(face_resized.astype('float32'))
        face_blob = np.expand_dims(face_blob, axis=0)
        preds = self.mask_model.predict(face_blob, verbose=0)
        class_idx = int(np.argmax(preds[0]))
        confidence = float(preds[0][class_idx])
        return class_idx, confidence

    def log_violation(self, confidence, alert_triggered, snapshot_path):
        timestamp = datetime.now().isoformat(sep=' ', timespec='seconds')
        with open(self.log_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, f"{confidence:.6f}", int(bool(alert_triggered)), str(snapshot_path)])

    def save_snapshot(self, face_roi):
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"violation_{ts}.jpg"
        path = self.snapshots_dir / filename
        # Save image (BGR) as-is
        cv2.imwrite(str(path), face_roi)
        return path

    def run(self, camera_index=0, confidence_threshold=0.5):
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("ERROR: Could not open webcam")
            return

        frame_count = 0
        start_time = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                faces = self.detect_faces(frame, confidence_threshold)
                self.total_faces += len(faces)

                # Draw boxes and handle violations
                annotated = frame.copy()
                now = time.time()
                for (startX, startY, endX, endY) in faces:
                    face_roi = frame[startY:endY, startX:endX]
                    if face_roi.size == 0:
                        continue

                    class_idx, confidence = self.predict_mask(face_roi)
                    color = self.colors[class_idx]
                    label = self.class_names[class_idx]

                    # Draw bounding box and label
                    cv2.rectangle(annotated, (startX, startY), (endX, endY), color, 2)
                    text = f"{label}: {confidence:.1%}"
                    tsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(annotated, (startX, startY - tsize[1] - 8), (startX + tsize[0] + 8, startY), color, -1)
                    cv2.putText(annotated, text, (startX + 4, startY - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    if class_idx == 1:
                        # violation
                        self.total_violations += 1

                        # Determine whether an audible alert is triggered based on cooldown
                        alert_triggered = False
                        if (now - self.last_alert_time) >= self.cooldown:
                            alert_triggered = True
                            self.last_alert_time = now
                            if self.pygame_available and self.alarm_sound is not None:
                                try:
                                    self.alarm_sound.play()
                                except Exception:
                                    pass

                        # Save snapshot and log
                        snapshot_path = self.save_snapshot(face_roi)
                        self.log_violation(confidence, alert_triggered, snapshot_path)

                # Overlay stats
                frame_count += 1
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0.0
                cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(annotated, f"Faces: {len(faces)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(annotated, f"Total Faces: {self.total_faces}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)
                cv2.putText(annotated, f"Violations: {self.total_violations}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

                cv2.imshow('Realtime Mask Detection - Logging', annotated)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    # Save and print session summary
                    self.save_session_summary()
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            if self.pygame_available:
                try:
                    self.pygame.mixer.quit()
                except Exception:
                    pass

    def save_session_summary(self):
        summary = {}
        summary['total_faces'] = self.total_faces
        summary['total_violations'] = self.total_violations
        if self.total_faces > 0:
            compliance = 100.0 * max(0.0, (self.total_faces - self.total_violations) / self.total_faces)
        else:
            compliance = 0.0
        summary['compliance_percent'] = compliance
        summary['timestamp'] = datetime.now().isoformat(sep=' ', timespec='seconds')

        text = (
            f"Session Summary - {summary['timestamp']}\n"
            f"Total Faces Detected: {summary['total_faces']}\n"
            f"Total Violations: {summary['total_violations']}\n"
            f"Compliance: {summary['compliance_percent']:.2f}%\n"
        )

        print(text)

        out_path = self.alerts_dir / 'session_summary.txt'
        with open(out_path, 'w') as f:
            f.write(text)


def main():
    project_root = Path(__file__).resolve().parent
    model_dir = project_root / 'model'
    alerts_dir = project_root / 'alerts'

    prototxt_path = model_dir / 'deploy.prototxt'
    caffemodel_path = model_dir / 'res10_300x300_ssd_iter_140000.caffemodel'
    mask_model_path = model_dir / 'mask_detector.h5'

    for path in [prototxt_path, caffemodel_path, mask_model_path]:
        if not path.exists():
            print(f"ERROR: Model file not found: {path}")
            return

    logger = RealtimeMaskLogger(prototxt_path, caffemodel_path, mask_model_path, alerts_dir, cooldown=5.0)
    logger.run(camera_index=0, confidence_threshold=0.5)


if __name__ == '__main__':
    print('=' * 60)
    print('REAL-TIME FACE MASK DETECTION WITH LOGGING')
    print('=' * 60 + '\n')
    main()
