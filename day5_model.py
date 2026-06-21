from pathlib import Path

try:
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.layers import AveragePooling2D, Flatten, Dense, Dropout, Input
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.utils import plot_model
except Exception as e:
    print('TensorFlow/Keras not available or failed to import:', e)
    print('This script requires TensorFlow. Install with: pip install tensorflow')
    raise


def build_model(input_shape=(224, 224, 3)):
    baseModel = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)

    # Freeze base model layers
    for layer in baseModel.layers:
        layer.trainable = False

    # Construct head
    head = baseModel.output
    head = AveragePooling2D(pool_size=(7, 7))(head)
    head = Flatten(name="flatten")(head)
    head = Dense(128, activation="relu")(head)
    head = Dropout(0.5)(head)
    head = Dense(2, activation="softmax")(head)

    model = Model(inputs=baseModel.input, outputs=head)

    # Compile
    opt = Adam(learning_rate=1e-4)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"]) 

    return model


def main():
    project_root = Path(__file__).resolve().parent
    model_dir = project_root / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    print("Building model...")
    model = build_model()

    print('\nModel Summary:\n')
    model.summary()

    out_path = model_dir / "model_architecture.png"
    try:
        plot_model(model, to_file=str(out_path), show_shapes=True, show_layer_names=True)
        print(f"Saved model architecture image to: {out_path}")
    except Exception as e:
        print("plot_model failed (ensure pydot and graphviz are installed):", e)
        print("To install: pip install pydot; then install graphviz system package and add to PATH.")


if __name__ == '__main__':
    main()
