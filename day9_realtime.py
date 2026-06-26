"""
Day 9: Real-Time Face Mask Detection using Webcam
- Load both face detector (DNN) and mask classifier (CNN)
- Process webcam frames in real-time
- Detect faces and classify mask/no-mask
- Display with color-coded bounding boxes (GREEN=Mask, RED=No Mask)
- Show FPS counter
"""

from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import time


class RealtimeMaskDetector:
    """Real-time face mask detection system"""
    
    def __init__(self, prototxt_path, caffemodel_path, mask_model_path):
        """
        Initialize the real-time mask detection system
        
        Args:
            prototxt_path: Path to deploy.prototxt
            caffemodel_path: Path to caffemodel
            mask_model_path: Path to mask_detector.h5
        """
        print("Initializing Real-Time Mask Detection System...")
        print("-" * 60)
        
        # Load face detector (DNN)
        print("Loading face detector model...")
        self.face_net = cv2.dnn.readNetFromCaffe(str(prototxt_path), str(caffemodel_path))
        print("✓ Face detector loaded")
        
        # Load mask classifier
        print("Loading mask classification model...")
        self.mask_model = load_model(str(mask_model_path))
        print("✓ Mask classifier loaded")
        
        self.class_names = ["With Mask", "Without Mask"]
        self.colors = {
            0: (0, 255, 0),   # Green for With Mask
            1: (0, 0, 255)    # Red for Without Mask
        }
        
        print("-" * 60)
        print("System ready!\n")
    
    def detect_faces(self, frame, confidence_threshold=0.5):
        """
        Detect faces in frame using DNN
        
        Args:
            frame: Input frame (BGR)
            confidence_threshold: Minimum confidence for detection
        
        Returns:
            List of face bounding boxes [(x1, y1, x2, y2), ...]
        """
        h, w = frame.shape[:2]
        
        # Prepare blob
        blob = cv2.dnn.blobFromImage(
            frame, 1.0, (300, 300),
            [104.0, 177.0, 123.0],
            swapRB=False,
            crop=False
        )
        
        # Detect
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        
        # Parse detections
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence < confidence_threshold:
                continue
            
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            startX, startY, endX, endY = box.astype("int")
            
            # Bound to frame
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)
            
            faces.append((startX, startY, endX, endY))
        
        return faces
    
    def predict_mask(self, face_roi):
        """
        Predict if face has mask or not
        
        Args:
            face_roi: Cropped face region
        
        Returns:
            Tuple of (class_idx, confidence)
        """
        # Resize to model input size
        face_resized = cv2.resize(face_roi, (224, 224))
        
        # Normalize using MobileNetV2 preprocessing
        face_blob = preprocess_input(face_resized.astype("float32"))
        face_blob = np.expand_dims(face_blob, axis=0)
        
        # Predict
        preds = self.mask_model.predict(face_blob, verbose=0)
        class_idx = np.argmax(preds[0])
        confidence = preds[0][class_idx]
        
        return class_idx, confidence
    
    def draw_predictions(self, frame, faces):
        """
        Draw face detections with predictions on frame
        
        Args:
            frame: Input frame (BGR)
            faces: List of face bounding boxes
        
        Returns:
            Annotated frame
        """
        frame_copy = frame.copy()
        
        for (startX, startY, endX, endY) in faces:
            # Extract face region
            face_roi = frame[startY:endY, startX:endX]
            
            if face_roi.size == 0:
                continue
            
            # Predict mask/no-mask
            class_idx, confidence = self.predict_mask(face_roi)
            label = self.class_names[class_idx]
            color = self.colors[class_idx]
            
            # Prepare label
            text = f"{label}: {confidence:.1%}"
            
            # Draw bounding box
            cv2.rectangle(frame_copy, (startX, startY), (endX, endY), color, 2)
            
            # Get text size for background
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Draw background for text
            cv2.rectangle(
                frame_copy,
                (startX, startY - text_size[1] - 10),
                (startX + text_size[0] + 10, startY),
                color,
                -1
            )
            
            # Draw text
            cv2.putText(
                frame_copy,
                text,
                (startX + 5, startY - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        
        return frame_copy
    
    def run(self, camera_index=0, confidence_threshold=0.5):
        """
        Run real-time mask detection
        
        Args:
            camera_index: Webcam index (0 for default)
            confidence_threshold: Face detection confidence threshold
        """
        # Open webcam
        print(f"Opening webcam (index={camera_index})...")
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print("ERROR: Could not open webcam")
            return
        
        # Get webcam properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Webcam opened: {frame_width}x{frame_height} @ {fps:.1f}fps")
        print("Press 'Q' to quit\n")
        
        # FPS tracking
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    print("ERROR: Failed to read frame from webcam")
                    break
                
                # Detect faces
                faces = self.detect_faces(frame, confidence_threshold)
                
                # Draw predictions
                frame_annotated = self.draw_predictions(frame, faces)
                
                # Calculate and display FPS
                frame_count += 1
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed if elapsed > 0 else 0
                
                fps_text = f"FPS: {current_fps:.1f}"
                cv2.putText(
                    frame_annotated,
                    fps_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                
                # Display face count
                info_text = f"Faces: {len(faces)}"
                cv2.putText(
                    frame_annotated,
                    info_text,
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                
                # Show frame
                cv2.imshow('Real-Time Face Mask Detection', frame_annotated)
                
                # Check for quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    print("\nQuitting...")
                    break
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            
            print("Webcam closed")
            print(f"\nTotal frames processed: {frame_count}")
            print(f"Average FPS: {frame_count / (time.time() - start_time):.1f}")


def main():
    """Main entry point"""
    project_root = Path(__file__).resolve().parent
    model_dir = project_root / "model"
    
    # Paths
    prototxt_path = model_dir / "deploy.prototxt"
    caffemodel_path = model_dir / "res10_300x300_ssd_iter_140000.caffemodel"
    mask_model_path = model_dir / "mask_detector.h5"
    
    # Verify files exist
    for path in [prototxt_path, caffemodel_path, mask_model_path]:
        if not path.exists():
            print(f"ERROR: Model file not found: {path}")
            return
    
    # Initialize detector
    detector = RealtimeMaskDetector(
        prototxt_path,
        caffemodel_path,
        mask_model_path
    )
    
    # Run real-time detection
    detector.run(
        camera_index=0,
        confidence_threshold=0.5
    )


if __name__ == '__main__':
    print("="*60)
    print("REAL-TIME FACE MASK DETECTION")
    print("="*60 + "\n")
    
    main()
