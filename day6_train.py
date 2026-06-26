from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from day5_model import build_model
from tensorflow.keras.utils import to_categorical

try:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
except Exception as e:
    print('TensorFlow/Keras not available or failed to import:', e)
    raise


def load_data():
    project_root = Path(__file__).resolve().parent
    data_dir = project_root / "data"

    print("Loading preprocessed data...")
    X_train = np.load(data_dir / "X_train.npy")
    y_train = np.load(data_dir / "y_train.npy")
    X_test  = np.load(data_dir / "X_test.npy")
    y_test  = np.load(data_dir / "y_test.npy")

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape:  {X_test.shape}")
    print(f"y_test shape:  {y_test.shape}")

    return X_train, y_train, X_test, y_test


def normalize_data(X_train, X_test):
    X_train = X_train.astype("float32") / 255.0
    X_test  = X_test.astype("float32")  / 255.0
    return X_train, X_test


def create_data_augmentation():
    return ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest"
    )


def create_callbacks(model_dir):
    return [
        ModelCheckpoint(
            filepath=str(model_dir / "mask_detector.h5"),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=7,
            verbose=1,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]


def plot_training_history(history, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history["accuracy"],     label="Training Accuracy",   linewidth=2)
    axes[0].plot(history.history["val_accuracy"], label="Validation Accuracy", linewidth=2)
    axes[0].set_title("Model Accuracy", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
    axes[0].legend(loc="lower right"); axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["loss"],     label="Training Loss",   linewidth=2)
    axes[1].plot(history.history["val_loss"], label="Validation Loss", linewidth=2)
    axes[1].set_title("Model Loss", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].legend(loc="upper right"); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
    print(f"Training plot saved to: {output_path}")
    plt.close()


def main():
    project_root = Path(__file__).resolve().parent
    model_dir = project_root / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    X_train, y_train, X_test, y_test = load_data()

    # One-hot encode labels
    y_train = to_categorical(y_train, num_classes=2)
    y_test  = to_categorical(y_test,  num_classes=2)
    print(f"y_train after encoding: {y_train.shape}")
    print(f"y_test  after encoding: {y_test.shape}")

    # Build model
    print("\nBuilding model...")
    model = build_model(input_shape=(224, 224, 3))
    model.summary()

    # Augmentation + callbacks
    print("\nCreating data augmentation...")
    augmentation = create_data_augmentation()
    print("Creating callbacks...")
    callbacks = create_callbacks(model_dir)

    # Train
    print("\nStarting model training...")
    print("Training with batch_size=32, epochs=20")
    history = model.fit(
        augmentation.flow(X_train, y_train, batch_size=32),
        validation_data=(X_test, y_test),
        epochs=20,
        callbacks=callbacks,
        verbose=1
    )

    # Plot
    print("\nPlotting training history...")
    plot_training_history(history, model_dir / "training_plot.png")

    # Evaluate
    print("\nEvaluating model on test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

    print("\n" + "="*50)
    print("TRAINING COMPLETED")
    print("="*50)
    print(f"Final Training Accuracy:   {history.history['accuracy'][-1]:.4f}")
    print(f"Final Training Loss:       {history.history['loss'][-1]:.4f}")
    print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"Final Validation Loss:     {history.history['val_loss'][-1]:.4f}")
    print(f"Test Accuracy:             {test_accuracy:.4f}")
    print(f"Test Loss:                 {test_loss:.4f}")
    print("="*50)
    print(f"Best model saved to: {model_dir / 'mask_detector.h5'}")


if __name__ == '__main__':
    main()