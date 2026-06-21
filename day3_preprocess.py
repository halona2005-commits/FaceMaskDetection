from pathlib import Path
import cv2
import numpy as np
from sklearn.model_selection import train_test_split


def get_image_paths(directory: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted([p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in exts])


def load_and_preprocess(path: Path, size=(224, 224)):
    img = cv2.imread(str(path))
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    img = img.astype("float32") / 255.0
    return img


def main():
    project_root = Path(__file__).resolve().parent
    data_root = project_root / "data"
    classes = ["with_mask", "without_mask"]

    X = []
    y_labels = []
    skipped = 0

    for cls in classes:
        cls_dir = data_root / cls
        if not cls_dir.exists():
            print(f"Warning: directory not found: {cls_dir}")
            continue
        paths = get_image_paths(cls_dir)
        for p in paths:
            img = load_and_preprocess(p)
            if img is None:
                skipped += 1
                continue
            X.append(img)
            y_labels.append(cls)
            if len(X) % 500 == 0:
                print(f"Loaded {len(X)} images so far...")

    if not X:
        print("No images were loaded. Exiting.")
        return

    X = np.stack(X, axis=0)

    # Map labels manually: with_mask -> 1, without_mask -> 0
    y = np.array([1 if lbl == "with_mask" else 0 for lbl in y_labels], dtype=np.int64)
    # reshape to (n_samples, 1) for consistency with binary labels
    y = y.reshape(-1, 1)
    # Flatten y for train_test_split stratify
    y_flat = y.ravel()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y_flat
    )

    # Ensure data folder exists
    data_root.mkdir(parents=True, exist_ok=True)

    np.save(data_root / "X_train.npy", X_train)
    np.save(data_root / "X_test.npy", X_test)
    np.save(data_root / "y_train.npy", y_train)
    np.save(data_root / "y_test.npy", y_test)

    print(f"Processed images: {len(X)}; skipped: {skipped}")
    print(f"Saved: {data_root / 'X_train.npy'}, {data_root / 'X_test.npy'}, {data_root / 'y_train.npy'}, {data_root / 'y_test.npy'}")


if __name__ == "__main__":
    main()
