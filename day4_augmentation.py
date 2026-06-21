import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import random
import cv2

# Prefer tensorflow's ImageDataGenerator, fall back to keras if needed
try:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
except Exception:
    try:
        from keras.preprocessing.image import ImageDataGenerator
    except Exception:
        ImageDataGenerator = None


def load_data(data_dir: Path):
    X_train = np.load(data_dir / 'X_train.npy')
    y_train = np.load(data_dir / 'y_train.npy')
    return X_train, y_train


def display_augmented_samples(datagen, sample, n=10, cols=5):
    aug_images = []
    it = datagen.flow(np.expand_dims(sample, 0), batch_size=1)
    for i in range(n):
        batch = next(it)
        img = batch[0]
        img = np.clip(img, 0.0, 1.0)
        aug_images.append(img)

    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()
    for i in range(len(axes)):
        axes[i].axis('off')
        if i < len(aug_images):
            axes[i].imshow(aug_images[i])
    plt.tight_layout()
    plt.show()


def augment_image_fallback(image, rotation_range=20, width_shift_range=0.2,
                           height_shift_range=0.2, shear_range=0.2,
                           zoom_range=0.2, horizontal_flip=True):
    h, w = image.shape[:2]
    img = (image * 255.0).astype('uint8')

    # Rotation
    angle = random.uniform(-rotation_range, rotation_range)
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # Shifts
    tx = random.uniform(-width_shift_range, width_shift_range) * w
    ty = random.uniform(-height_shift_range, height_shift_range) * h
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # Shear
    shear = random.uniform(-shear_range, shear_range)
    M = np.array([[1, shear, 0], [0, 1, 0]], dtype=np.float32)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # Zoom
    zx = 1 + random.uniform(-zoom_range, zoom_range)
    zy = zx
    nh, nw = int(h * zy), int(w * zx)
    img_zoom = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    # center crop or pad back to original size
    if nw >= w and nh >= h:
        x1 = (nw - w) // 2
        y1 = (nh - h) // 2
        img = img_zoom[y1:y1 + h, x1:x1 + w]
    else:
        top = (h - nh) // 2
        left = (w - nw) // 2
        canvas = np.zeros((h, w, 3), dtype=img_zoom.dtype)
        canvas[top:top + nh, left:left + nw] = img_zoom
        img = canvas

    # Horizontal flip
    if horizontal_flip and random.random() < 0.5:
        img = cv2.flip(img, 1)

    img = img.astype('float32') / 255.0
    return img


def display_augmented_list(aug_images, cols=5):
    n = len(aug_images)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()
    for i in range(len(axes)):
        axes[i].axis('off')
        if i < n:
            axes[i].imshow(np.clip(aug_images[i], 0.0, 1.0))
    plt.tight_layout()
    plt.show()


def main():
    project_root = Path(__file__).resolve().parent
    data_dir = project_root / 'data'

    if not (data_dir / 'X_train.npy').exists():
        print('Missing X_train.npy in data/. Run day3_preprocess.py first.')
        return

    X_train, y_train = load_data(data_dir)
    print('Loaded X_train:', X_train.shape, 'y_train:', y_train.shape)

    # If Keras ImageDataGenerator is available, use it; otherwise use fallback
    if ImageDataGenerator is not None:
        train_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        val_datagen = ImageDataGenerator(rescale=1.0)

        sample_img = X_train[0]
        display_augmented_samples(train_datagen, sample_img, n=10, cols=5)
        print('Total augmented samples generated for display:', 10)
    else:
        print('Keras not available — using OpenCV-based fallback augmenter.')
        sample_img = X_train[0]
        aug_images = [augment_image_fallback(sample_img) for _ in range(10)]
        display_augmented_list(aug_images, cols=5)
        print('Total augmented samples generated for display (fallback):', len(aug_images))


if __name__ == '__main__':
    main()
