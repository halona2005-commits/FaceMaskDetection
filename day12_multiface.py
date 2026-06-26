"""
Day 12: Real-Time Multi-Face Mask Detection
- Detect multiple faces per frame
- Predict mask/no-mask independently for every detected face
- Draw individual bounding boxes and labels per face
- Trigger alert when any face is without a mask
- Display real-time counter: Faces Detected, With Mask, Without Mask
- Highlight frame border RED on any violation, GREEN when all faces comply
"""

from pathlib import Path
import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


class MultiFaceMaskDetector:
    """Real-time multi-face mask detection system."""

    def __init__(self, prototxt_path, caffemodel_path, mask_model_path):
        print("Initializing Multi-Face Mask Detection...")
        self.face_net = cv2.dnn.readNetFromCaffe(str(prototxt_path), str(caffemodel_path))
        self.mask_model = load_model(str(mask_model_path))

        self.class_names = ["With Mask", "Without Mask"]
        self.colors = {
            0: (0, 255, 0),   # Green
            1: (0, 0, 255)    # Red
        }

    def detect_faces(self, frame, confidence_threshold=0.5):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame,
            1.0,
            (300, 300),
            [104.0, 177.0, 123.0],
            swapRB=False,
            crop=False
        )
        self.face_net.setInput(blob)
        detections = self.face_net.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < confidence_threshold:
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            startX, startY, endX, endY = box.astype('int')
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)

            if endX <= startX or endY <= startY:
                continue

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

    def draw_predictions(self, frame, faces):
        annotated = frame.copy()
        total_faces = len(faces)
        with_mask = 0
        without_mask = 0
        any_violation = False

        for (startX, startY, endX, endY) in faces:
            face_roi = frame[startY:endY, startX:endX]
            if face_roi.size == 0:
                continue

            class_idx, confidence = self.predict_mask(face_roi)
            label = self.class_names[class_idx]
            color = self.colors[class_idx]
            text = f"{label}: {confidence:.1%}"

            if class_idx == 0:
                with_mask += 1
            else:
                without_mask += 1
                any_violation = True

            cv2.rectangle(annotated, (startX, startY), (endX, endY), color, 2)

            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(
                annotated,
                (startX, startY - text_size[1] - 12),
                (startX + text_size[0] + 10, startY),
                color,
                -1
            )
            cv2.putText(
                annotated,
                text,
                (startX + 5, startY - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        return annotated, total_faces, with_mask, without_mask, any_violation

    def annotate_frame(self, frame, total_faces, with_mask, without_mask, any_violation, fps):
        annotated = frame.copy()
        border_color = (0, 255, 0) if not any_violation else (0, 0, 255)
        border_thickness = 8
        h, w = annotated.shape[:2]
        cv2.rectangle(annotated, (0, 0), (w - 1, h - 1), border_color, border_thickness)

        status_text = (
            f"Faces Detected: {total_faces} | With Mask: {with_mask} | Without Mask: {without_mask}"
        )
        status_color = border_color

        cv2.putText(
            annotated,
            status_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            status_color,
            2,
            cv2.LINE_AA
        )

        if any_violation:
            alert_text = "ALERT: Mask violation detected!"
            cv2.putText(
                annotated,
                alert_text,
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )

        cv2.putText(
            annotated,
            f"FPS: {fps:.1f}",
            (10, annotated.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        return annotated

    def run(self, camera_index=0, confidence_threshold=0.5):
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("ERROR: Could not open webcam")
            return

        frame_count = 0
        start_time = time.time()

        print("Press 'Q' to quit")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("ERROR: Could not read frame")
                    break

                faces = self.detect_faces(frame, confidence_threshold)
                annotated, total_faces, with_mask, without_mask, any_violation = self.draw_predictions(frame, faces)

                frame_count += 1
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0.0

                annotated = self.annotate_frame(annotated, total_faces, with_mask, without_mask, any_violation, fps)
                cv2.imshow('Day12 Multi-Face Mask Detection', annotated)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()


def main():
    project_root = Path(__file__).resolve().parent
    model_dir = project_root / 'model'

    prototxt_path = model_dir / 'deploy.prototxt'
    caffemodel_path = model_dir / 'res10_300x300_ssd_iter_140000.caffemodel'
    mask_model_path = model_dir / 'mask_detector.h5'

    for path in [prototxt_path, caffemodel_path, mask_model_path]:
        if not path.exists():
            print(f"ERROR: Required model file not found: {path}")
            return

    detector = MultiFaceMaskDetector(prototxt_path, caffemodel_path, mask_model_path)
    detector.run(camera_index=0, confidence_threshold=0.5)


if __name__ == '__main__':
    main()
