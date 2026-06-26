"""
Day 9: Real-Time Mask Detection - Quick Start Guide
"""

# BASIC USAGE
# ===========
# Simply run the script:
#
#   python day9_realtime.py
#
# The script will:
#   1. Initialize both models (face detector + mask classifier)
#   2. Open your default webcam (index 0)
#   3. Process frames in real-time
#   4. Draw bounding boxes:
#      - GREEN box with "Mask: X%" = Face with mask detected
#      - RED box with "No Mask: X%" = Face without mask detected
#   5. Display FPS counter in top-left
#   6. Press 'Q' to quit


# SYSTEM REQUIREMENTS
# ===================
# - Webcam/camera connected to computer
# - OpenCV (cv2) with video capture support
# - TensorFlow/Keras
# - CUDA (optional, for GPU acceleration)


# SCRIPT FEATURES
# ===============
# 1. Real-Time Processing
#    - Detects faces using OpenCV DNN (SSD MobileNet)
#    - Classifies mask/no-mask for each face
#    - Displays results at webcam's native FPS
#
# 2. Visual Feedback
#    - Color-coded bounding boxes (GREEN/RED)
#    - Confidence scores displayed on boxes
#    - FPS counter shows processing speed
#    - Face count shows number of detected faces
#
# 3. Performance Optimized
#    - Efficient DNN-based face detection
#    - Pre-trained MobileNetV2 classifier (lightweight)
#    - Batch prediction if multiple faces present


# CLASS: RealtimeMaskDetector
# ===========================
#
# Methods:
#   - __init__(prototxt_path, caffemodel_path, mask_model_path)
#     Initialize with model file paths
#
#   - detect_faces(frame, confidence_threshold=0.5)
#     Detect face regions in a frame
#     Returns: List of bounding boxes (x1, y1, x2, y2)
#
#   - predict_mask(face_roi)
#     Classify single face as mask/no-mask
#     Returns: (class_idx, confidence)
#
#   - draw_predictions(frame, faces)
#     Draw detections and predictions on frame
#     Returns: Annotated frame
#
#   - run(camera_index=0, confidence_threshold=0.5)
#     Main real-time detection loop


# ADVANCED USAGE
# ==============
#
# Use different camera:
#   detector.run(camera_index=1)  # Use second camera
#
# Adjust face detection confidence:
#   detector.run(confidence_threshold=0.7)  # More strict detection


# KEYBOARD CONTROLS
# =================
# Q / q     - Quit the application
# (other keys can be added in future versions)


# OUTPUT INFORMATION
# ==================
# Console prints:
#   - Model loading status
#   - Webcam resolution and FPS
#   - Total frames processed
#   - Average FPS after exit
#
# On-screen display:
#   - FPS counter (top-left)
#   - Face count (below FPS)
#   - Bounding boxes with confidence scores


# TROUBLESHOOTING
# ===============
#
# "ERROR: Could not open webcam"
#   - Check if camera is connected
#   - Try different camera index: detector.run(camera_index=1)
#   - Check if camera is not used by another application
#
# Slow performance / Low FPS
#   - Close unnecessary applications
#   - GPU might help if available
#   - Reduce frame resolution if supported
#
# Model loading errors
#   - Ensure all model files exist in model/ folder:
#     * deploy.prototxt
#     * res10_300x300_ssd_iter_140000.caffemodel
#     * mask_detector.h5


# EXAMPLE OUTPUT
# ==============
#
# ============================================================
# REAL-TIME FACE MASK DETECTION
# ============================================================
#
# Initializing Real-Time Mask Detection System...
# ────────────────────────────────────────────────────────────
# Loading face detector model...
# ✓ Face detector loaded
# Loading mask classification model...
# ✓ Mask classifier loaded
# ────────────────────────────────────────────────────────────
# System ready!
#
# Opening webcam (index=0)...
# Webcam opened: 640x480 @ 30.0fps
# Press 'Q' to quit
#
# [... running, detecting faces in real-time ...]
#
# Quitting...
# Webcam closed
#
# Total frames processed: 542
# Average FPS: 28.3


# NEXT STEPS
# ==========
# 1. Run: python day9_realtime.py
# 2. Position yourself in front of webcam
# 3. Test with/without mask
# 4. Observe color changes and confidence scores
# 5. Press Q to exit


# FILES USED
# ==========
# day9_realtime.py              - This script
# model/deploy.prototxt        - Face detector architecture
# model/res10_300x300_ssd_...  - Face detector weights
# model/mask_detector.h5       - Mask classifier model
