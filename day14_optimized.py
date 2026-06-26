import cv2
import time
import numpy as np
from pathlib import Path
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Paths
project_root = Path(__file__).resolve().parent
model_dir    = project_root / "model"
FACE_PROTO   = str(model_dir / "deploy.prototxt")
FACE_MODEL   = str(model_dir / "res10_300x300_ssd_iter_140000.caffemodel")
MASK_MODEL   = str(model_dir / "mask_detector.h5")

# Settings
CONFIDENCE_THRESHOLD = 0.5
MIN_FACE_SIZE        = 30
SKIP_FRAMES          = 2      # process every 2nd frame
RESIZE_SCALE         = 0.5   # resize frame to 50%
FPS_BUFFER           = 30     # rolling average buffer

print("="*55)
print("  OPTIMIZED FACE MASK DETECTION - DAY 14")
print("="*55)

# Load models
print("\nLoading models...")
face_net = cv2.dnn.readNet(FACE_PROTO, FACE_MODEL)
mask_net = load_model(MASK_MODEL)
print("✅ Models loaded!")

# Class labels
LABELS = ["without_mask", "with_mask"]
COLORS = {"with_mask": (0, 255, 0), "without_mask": (0, 0, 255)}


def equalize_lighting(frame):
    """Apply histogram equalization for poor lighting."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)
    # Merge back to BGR for detection
    frame_eq = cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2BGR)
    return frame_eq


def detect_faces(frame, confidence_threshold=0.5):
    """Detect faces using OpenCV DNN."""
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        frame, 1.0, (300, 300),
        (104.0, 177.0, 123.0)
    )
    face_net.setInput(blob)
    detections = face_net.forward()

    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < confidence_threshold:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        startX, startY, endX, endY = box.astype("int")

        # Clamp to frame
        startX = max(0, startX)
        startY = max(0, startY)
        endX   = min(w - 1, endX)
        endY   = min(h - 1, endY)

        # Filter small faces
        face_w = endX - startX
        face_h = endY - startY
        if face_w < MIN_FACE_SIZE or face_h < MIN_FACE_SIZE:
            continue

        faces.append({
            "box": (startX, startY, endX, endY),
            "confidence": float(confidence)
        })

    return faces


def predict_mask(face_img):
    """Predict mask/no-mask for a face crop."""
    face = cv2.resize(face_img, (224, 224))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = preprocess_input(face)
    face = np.expand_dims(face, axis=0)
    preds = mask_net.predict(face, verbose=0)[0]
    label = LABELS[np.argmax(preds)]
    confidence = float(np.max(preds))
    return label, confidence


def draw_border(frame, color, thickness=15):
    """Draw colored border around frame."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, h), color, thickness)
    return frame


def run_detection():
    """Main optimized detection loop."""
    cap = None
    frame_count   = 0
    fps_buffer    = deque(maxlen=FPS_BUFFER)
    last_frame    = None
    last_faces    = []
    prev_time     = time.time()

    # Stats
    total_faces      = 0
    total_violations = 0
    start_time       = time.time()

    print("\n✅ Starting optimized webcam detection...")
    print("Press Q to quit\n")

    while True:
        # Auto reconnect if webcam disconnects
        if cap is None or not cap.isOpened():
            print("Opening webcam...")
            try:
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    print("❌ Webcam not found! Retrying in 2s...")
                    time.sleep(2)
                    continue
                print("✅ Webcam connected!")
            except Exception as e:
                print(f"Webcam error: {e}. Retrying...")
                time.sleep(2)
                continue

        try:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("⚠️ Frame read failed. Reconnecting...")
                cap.release()
                cap = None
                continue

            frame_count += 1

            # FPS calculation
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time + 1e-6)
            prev_time = curr_time
            fps_buffer.append(fps)
            avg_fps = sum(fps_buffer) / len(fps_buffer)

            # ── OPTIMIZATION 1: Skip alternate frames ──
            if frame_count % SKIP_FRAMES == 0:

                # ── OPTIMIZATION 2: Resize frame ──
                small = cv2.resize(
                    frame, None,
                    fx=RESIZE_SCALE,
                    fy=RESIZE_SCALE
                )

                # ── OPTIMIZATION 3: Histogram equalization ──
                small_eq = equalize_lighting(small)

                # Detect faces on small frame
                faces_small = detect_faces(small_eq, CONFIDENCE_THRESHOLD)

                # Scale boxes back to original size
                last_faces = []
                for f in faces_small:
                    x1, y1, x2, y2 = f["box"]
                    last_faces.append({
                        "box": (
                            int(x1 / RESIZE_SCALE),
                            int(y1 / RESIZE_SCALE),
                            int(x2 / RESIZE_SCALE),
                            int(y2 / RESIZE_SCALE)
                        ),
                        "confidence": f["confidence"]
                    })

            # Use last detected faces for non-processed frames
            has_violation = False
            frame_faces   = 0
            frame_masks   = 0
            frame_no_mask = 0

            for face_info in last_faces:
                x1, y1, x2, y2 = face_info["box"]

                # Crop face from original frame
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue

                # Predict
                label, conf = predict_mask(face_crop)
                color = COLORS[label]

                # Draw box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                text = f"{label}: {conf*100:.1f}%"
                cv2.putText(
                    frame, text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2
                )

                frame_faces += 1
                if label == "without_mask":
                    has_violation = True
                    frame_no_mask += 1
                else:
                    frame_masks += 1

            # Update stats
            total_faces      += frame_faces
            total_violations += frame_no_mask

            # ── OPTIMIZATION 4: Frame border color ──
            border_color = (0, 0, 255) if has_violation else (0, 255, 0)
            draw_border(frame, border_color)

            # Alert banner
            if has_violation:
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 40),
                              (0, 0, 255), -1)
                cv2.putText(
                    frame,
                    "⚠ ALERT: No Mask Detected!",
                    (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2
                )

            # HUD overlay
            session_time = time.time() - start_time
            compliance = (
                ((total_faces - total_violations) / total_faces * 100)
                if total_faces > 0 else 100.0
            )

            hud_lines = [
                f"FPS: {avg_fps:.1f}",
                f"Faces: {frame_faces}",
                f"With Mask: {frame_masks}",
                f"No Mask: {frame_no_mask}",
                f"Compliance: {compliance:.1f}%",
                f"Session: {int(session_time)}s",
                f"Frame: {frame_count}",
                "Press Q to quit"
            ]

            y_start = 55
            for i, line in enumerate(hud_lines):
                cv2.putText(
                    frame, line,
                    (10, y_start + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1,
                    cv2.LINE_AA
                )

            cv2.imshow("Day 14 - Optimized Face Mask Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            print(f"Error: {e}")
            continue

    # Cleanup
    if cap:
        cap.release()
    cv2.destroyAllWindows()

    # Benchmark summary
    session_time = time.time() - start_time
    avg_fps_final = sum(fps_buffer) / len(fps_buffer) if fps_buffer else 0
    compliance = (
        ((total_faces - total_violations) / total_faces * 100)
        if total_faces > 0 else 100.0
    )

    print("\n" + "="*55)
    print("  BENCHMARK SUMMARY")
    print("="*55)
    print(f"  Total Frames Processed : {frame_count}")
    print(f"  Session Duration       : {session_time:.1f}s")
    print(f"  Average FPS            : {avg_fps_final:.1f}")
    print(f"  Baseline FPS (Day 9)   : ~10.7")
    improvement = ((avg_fps_final - 10.7) / 10.7) * 100
    print(f"  FPS Improvement        : {improvement:.1f}%")
    print(f"  Total Faces Detected   : {total_faces}")
    print(f"  Total Violations       : {total_violations}")
    print(f"  Compliance Rate        : {compliance:.1f}%")
    print("="*55)
    print("✅ Day 14 Complete!")


if __name__ == "__main__":
    run_detection()