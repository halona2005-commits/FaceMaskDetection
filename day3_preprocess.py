import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from pathlib import Path
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

project_root = Path(__file__).resolve().parent
DATASET_PATH = project_root / "data"
DATA_PATH = project_root / "data"
DATA_PATH.mkdir(exist_ok=True)

IMAGE_SIZE = (224, 224)
data = []
labels = []

print("Loading images...")

# Load with_mask → label 1
for img_name in os.listdir(DATASET_PATH / "with_mask"):
    img_path = str(DATASET_PATH / "with_mask" / img_name)
    img = cv2.imread(img_path)
    if img is None:
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMAGE_SIZE)
    img = preprocess_input(img)  # ← MobileNetV2 preprocessing
    data.append(img)
    labels.append(1)

print(f"With mask: {labels.count(1)}")

# Load without_mask → label 0
for img_name in os.listdir(DATASET_PATH / "without_mask"):
    img_path = str(DATASET_PATH / "without_mask" / img_name)
    img = cv2.imread(img_path)
    if img is None:
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMAGE_SIZE)
    img = preprocess_input(img)  # ← MobileNetV2 preprocessing
    data.append(img)
    labels.append(0)

print(f"Without mask: {labels.count(0)}")

data   = np.array(data, dtype="float32")
labels = np.array(labels)

print(f"Data shape: {data.shape}")
print(f"Pixel range: {data.min():.2f} to {data.max():.2f}")

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42, stratify=labels
)

np.save(DATA_PATH / "X_train.npy", X_train)
np.save(DATA_PATH / "X_test.npy",  X_test)
np.save(DATA_PATH / "y_train.npy", y_train)
np.save(DATA_PATH / "y_test.npy",  y_test)

print("✅ Saved successfully!")