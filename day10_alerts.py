"""
Day 10: Real-Time Face Mask Detection with Alerts
- Visual flashing banner when no mask detected
- Audio alarm (pygame) with generated tone fallback
- 5-second cooldown between audible alerts
- Alert counter displayed on screen
"""

from pathlib import Path
import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


class RealtimeMaskAlertDetector:
    def __init__(self, prototxt_path, caffemodel_path, mask_model_path, cooldown=5.0):
        # Load face detector
        self.face_net = cv2.dnn.readNetFromCaffe(str(prototxt_path), str(caffemodel_path))

        # Load mask classifier
        self.mask_model = load_model(str(mask_model_path))

        self.class_names = ["With Mask", "Without Mask"]
        self.colors = {0: (0, 255, 0), 1: (0, 0, 255)}

        # Alert state
        self.cooldown = float(cooldown)
        self.last_alert_time = 0.0
        self.alert_count = 0

        # Try to initialize pygame for audio alerts and generate a tone sound
        self.pygame_available = False
        try:
            import pygame
            self.pygame = pygame
            # Configure mixer with a known sample rate
            sr = 44100
            try:
                pygame.mixer.init(frequency=sr, size=-16, channels=1)
            except Exception:
                # If pre-init is needed, call pre_init then init
                pygame.mixer.pre_init(sr, -16, 1)
                pygame.mixer.init()

            self.alarm_sound = self._generate_alarm_sound(duration=0.8, freq=880, sr=sr)
            self.pygame_available = True
        except Exception:
            self.pygame_available = False

    def _generate_alarm_sound(self, duration=0.8, freq=880, sr=44100):
        # Generate a simple sine wave tone and return a pygame Sound
        try:
            t = np.linspace(0, duration, int(sr * duration), False)
            tone = 0.5 * np.sin(2 * np.pi * freq * t)
            audio = (tone * 32767).astype(np.int16)
            # Try mono first
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
            startX, startY, endX, endY = box.astype("int")
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)
            faces.append((startX, startY, endX, endY))

        return faces

    def predict_mask(self, face_roi):
        face_resized = cv2.resize(face_roi, (224, 224))
        face_blob = preprocess_input(face_resized.astype("float32"))
        face_blob = np.expand_dims(face_blob, axis=0)
        preds = self.mask_model.predict(face_blob, verbose=0)
        class_idx = int(np.argmax(preds[0]))
        confidence = float(preds[0][class_idx])
        return class_idx, confidence

    def draw_predictions(self, frame, faces):
        frame_copy = frame.copy()
        any_no_mask = False

        for (startX, startY, endX, endY) in faces:
            face_roi = frame[startY:endY, startX:endX]
            if face_roi.size == 0:
                continue

            class_idx, confidence = self.predict_mask(face_roi)
            if class_idx == 1:
                any_no_mask = True

            label = self.class_names[class_idx]
            color = self.colors[class_idx]
            text = f"{label}: {confidence:.1%}"

            cv2.rectangle(frame_copy, (startX, startY), (endX, endY), color, 2)
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame_copy, (startX, startY - text_size[1] - 10), (startX + text_size[0] + 10, startY), color, -1)
            cv2.putText(frame_copy, text, (startX + 5, startY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame_copy, any_no_mask

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
                frame_annotated, any_no_mask = self.draw_predictions(frame, faces)

                now = time.time()

                # If any face without mask, show flashing banner (visual alert)
                if any_no_mask:
                    # Flashing at ~2Hz
                    if int(now * 2) % 2 == 0:
                        h, w = frame_annotated.shape[:2]
                        cv2.rectangle(frame_annotated, (0, 0), (w, 60), (0, 0, 255), -1)
                        cv2.putText(frame_annotated, 'ALERT: No Mask Detected!', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

                    # Audible alert + counter only if cooldown elapsed
                    if (now - self.last_alert_time) >= self.cooldown:
                        self.alert_count += 1
                        self.last_alert_time = now
                        if self.pygame_available and self.alarm_sound is not None:
                            try:
                                self.alarm_sound.play()
                            except Exception:
                                pass

                # Overlay FPS and counts
                frame_count += 1
                elapsed = now - start_time
                current_fps = frame_count / elapsed if elapsed > 0 else 0.0
                cv2.putText(frame_annotated, f"FPS: {current_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame_annotated, f"Faces: {len(faces)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame_annotated, f"Alerts: {self.alert_count}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                cv2.imshow('Realtime Mask Detection - Alerts', frame_annotated)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            if self.pygame_available:
                try:
                    self.pygame.mixer.quit()
                except Exception:
                    pass


def main():
    project_root = Path(__file__).resolve().parent
    model_dir = project_root / "model"

    prototxt_path = model_dir / "deploy.prototxt"
    caffemodel_path = model_dir / "res10_300x300_ssd_iter_140000.caffemodel"
    mask_model_path = model_dir / "mask_detector.h5"

    for path in [prototxt_path, caffemodel_path, mask_model_path]:
        if not path.exists():
            print(f"ERROR: Model file not found: {path}")
            return

    detector = RealtimeMaskAlertDetector(prototxt_path, caffemodel_path, mask_model_path, cooldown=5.0)
    detector.run(camera_index=0, confidence_threshold=0.5)


if __name__ == '__main__':
    print("=" * 60)
    print("REAL-TIME FACE MASK DETECTION WITH ALERTS")
    print("=" * 60 + "\n")
    main()
