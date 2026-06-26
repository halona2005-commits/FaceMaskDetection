"""
Day 8: Face Detection using OpenCV DNN
- Load OpenCV's pre-trained face detector (SSD MobileNet)
- Detect faces in images with confidence threshold filtering
- Test on sample images and visualize detections
"""

from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class FaceDetector:
    """Face detector using OpenCV DNN module"""
    
    def __init__(self, prototxt_path, model_path):
        """
        Initialize the face detector with pre-trained model
        
        Args:
            prototxt_path: Path to deploy.prototxt
            model_path: Path to caffemodel
        """
        print("Loading face detector model...")
        self.net = cv2.dnn.readNetFromCaffe(str(prototxt_path), str(model_path))
        print("Face detector model loaded successfully!")
        
    def detect_faces(self, frame, confidence_threshold=0.5):
        """
        Detect faces in a given frame/image
        
        Args:
            frame: Input image (BGR format from OpenCV)
            confidence_threshold: Minimum confidence to consider detection (0-1)
        
        Returns:
            detections: List of dicts with keys:
                - 'box': (startX, startY, endX, endY)
                - 'confidence': Confidence score
        """
        h, w = frame.shape[:2]
        
        # Prepare blob for network input (300x300 required by model)
        blob = cv2.dnn.blobFromImage(
            frame, 1.0, (300, 300),
            [104.0, 177.0, 123.0],  # Mean subtraction values
            swapRB=False,
            crop=False
        )
        
        # Set input and run forward pass
        self.net.setInput(blob)
        detections = self.net.forward()
        
        # Process detections
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            # Filter weak detections
            if confidence < confidence_threshold:
                continue
            
            # Extract face bounding box coordinates
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            startX, startY, endX, endY = box.astype("int")
            
            # Ensure coordinates are within frame bounds
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)
            
            faces.append({
                'box': (startX, startY, endX, endY),
                'confidence': confidence
            })
        
        return faces
    
    def draw_detections(self, frame, detections):
        """
        Draw bounding boxes and confidence scores on frame
        
        Args:
            frame: Input image
            detections: List of detection dicts
        
        Returns:
            frame: Image with drawn bounding boxes
        """
        frame_copy = frame.copy()
        
        for detection in detections:
            startX, startY, endX, endY = detection['box']
            confidence = detection['confidence']
            
            # Draw green bounding box
            cv2.rectangle(frame_copy, (startX, startY), (endX, endY), (0, 255, 0), 2)
            
            # Draw confidence score
            text = f"{confidence:.2%}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Background for text
            cv2.rectangle(
                frame_copy,
                (startX, startY - text_size[1] - 10),
                (startX + text_size[0] + 10, startY),
                (0, 255, 0),
                -1
            )
            
            # Text
            cv2.putText(
                frame_copy,
                text,
                (startX + 5, startY - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )
        
        return frame_copy


def load_sample_images(data_dir, num_samples=5):
    """
    Load sample images from dataset
    
    Args:
        data_dir: Path to data directory
        num_samples: Number of samples to load
    
    Returns:
        List of (image_path, image) tuples
    """
    sample_images = []
    
    # Get images from both with_mask and without_mask folders
    mask_dir = data_dir / "with_mask"
    no_mask_dir = data_dir / "without_mask"
    
    # Load samples from each folder
    samples_per_class = num_samples // 2
    
    for folder, folder_name in [(mask_dir, "with_mask"), (no_mask_dir, "without_mask")]:
        if folder.exists():
            images = list(folder.glob("*.jpg"))[:samples_per_class]
            for img_path in images:
                img = cv2.imread(str(img_path))
                if img is not None:
                    sample_images.append((img_path, img))
    
    return sample_images


def test_face_detector():
    """Test face detector on sample images"""
    project_root = Path(__file__).resolve().parent
    model_dir = project_root / "model"
    data_dir = project_root / "data"
    
    # Initialize detector
    prototxt_path = model_dir / "deploy.prototxt"
    model_path = model_dir / "res10_300x300_ssd_iter_140000.caffemodel"
    
    detector = FaceDetector(prototxt_path, model_path)
    
    # Load sample images
    print("\nLoading sample images...")
    sample_images = load_sample_images(data_dir, num_samples=5)
    print(f"Loaded {len(sample_images)} sample images")
    
    if not sample_images:
        print("ERROR: Could not load sample images")
        return
    
    # Create figure for visualization
    num_images = len(sample_images)
    fig, axes = plt.subplots(
        (num_images + 1) // 2, 2,
        figsize=(14, 5 * ((num_images + 1) // 2))
    )
    
    if num_images == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Process each image
    print("\n" + "="*60)
    print("FACE DETECTION RESULTS")
    print("="*60)
    
    for idx, (img_path, frame) in enumerate(sample_images):
        print(f"\n[{idx+1}/{num_images}] Processing: {img_path.name}")
        
        # Detect faces
        detections = detector.detect_faces(frame, confidence_threshold=0.5)
        print(f"  Detected {len(detections)} face(s)")
        
        # Print detection details
        for det_idx, detection in enumerate(detections):
            startX, startY, endX, endY = detection['box']
            conf = detection['confidence']
            width = endX - startX
            height = endY - startY
            print(f"    Face {det_idx+1}: Confidence={conf:.4f}, "
                  f"Box=({startX},{startY},{endX},{endY}), "
                  f"Size={width}x{height}px")
        
        # Draw detections
        frame_with_boxes = detector.draw_detections(frame, detections)
        
        # Convert BGR to RGB for matplotlib
        frame_rgb = cv2.cvtColor(frame_with_boxes, cv2.COLOR_BGR2RGB)
        
        # Display
        ax = axes[idx]
        ax.imshow(frame_rgb)
        ax.set_title(
            f"{img_path.parent.name}\n"
            f"Detected: {len(detections)} face(s)",
            fontsize=11,
            fontweight='bold'
        )
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(num_images, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('OpenCV DNN Face Detection Results (Confidence Threshold: 0.5)',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('face_detection_results.png', dpi=150, bbox_inches='tight')
    print("\n" + "="*60)
    print("Face detection results saved as 'face_detection_results.png'")
    print("="*60)
    plt.show()


def demo_single_image():
    """Demo function for single image face detection"""
    project_root = Path(__file__).resolve().parent
    model_dir = project_root / "model"
    
    # Initialize detector
    prototxt_path = model_dir / "deploy.prototxt"
    model_path = model_dir / "res10_300x300_ssd_iter_140000.caffemodel"
    
    detector = FaceDetector(prototxt_path, model_path)
    
    # Example usage
    print("\n" + "="*60)
    print("FACE DETECTOR API EXAMPLE")
    print("="*60)
    print("""
Usage:
    from day8_face_detector import FaceDetector
    
    # Initialize detector
    detector = FaceDetector(
        'model/deploy.prototxt',
        'model/res10_300x300_ssd_iter_140000.caffemodel'
    )
    
    # Load image
    frame = cv2.imread('path/to/image.jpg')
    
    # Detect faces
    detections = detector.detect_faces(frame, confidence_threshold=0.5)
    
    # Draw detections
    result = detector.draw_detections(frame, detections)
    cv2.imshow('Faces', result)
    cv2.waitKey(0)
    
    # Each detection contains:
    # {
    #     'box': (startX, startY, endX, endY),
    #     'confidence': float between 0 and 1
    # }
    """)
    print("="*60)


if __name__ == '__main__':
    # Run face detection test
    test_face_detector()
    
    # Show API example
    demo_single_image()
