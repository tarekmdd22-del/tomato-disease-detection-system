"""
Train a tomato leaf disease classifier using transfer learning
(MobileNetV2 backbone, pretrained on ImageNet).

Usage:
    python src/train_model.py --data data/train --epochs 15

Expected data layout:
    data/train/
        Healthy/*.jpg
        Late_Blight/*.jpg
        Early_Blight/*.jpg
        ... (one folder per class)
"""
import argparse
import json
import os
import sys

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(os.path.dirname(__file__))
from preprocessing import load_datasets, load_datasets_from_train_val, build_augmentation_layer, IMG_SIZE

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def build_model(num_classes, img_size=IMG_SIZE, fine_tune_at=None):
    """MobileNetV2 backbone (frozen, or partially fine-tuned) + custom classification head."""
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=img_size + (3,),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = fine_tune_at is not None
    if fine_tune_at is not None:
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

    inputs = tf.keras.Input(shape=img_size + (3,))
    x = build_augmentation_layer()(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    return model, base_model


def main(data_dir, val_dir=None, epochs=15, fine_tune_epochs=5, batch_size=32):
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    if val_dir:
        print(f"Loading images from {data_dir} (train) and {val_dir} (val) ...")
        train_ds, val_ds, class_names = load_datasets_from_train_val(data_dir, val_dir, batch_size=batch_size)
    else:
        print(f"Loading images from {data_dir} (auto 80/20 split) ...")
        train_ds, val_ds, class_names = load_datasets(data_dir, batch_size=batch_size)
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")

    with open(os.path.join(MODELS_DIR, "class_names.json"), "w") as f:
        json.dump(class_names, f, indent=2)

    # --- Phase 1: train the classification head only (backbone frozen) ---
    model, base_model = build_model(num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(MODELS_DIR, "tomato_disease_model.keras"),
            save_best_only=True, monitor="val_accuracy",
        ),
    ]

    print("\n=== Phase 1: training classification head (backbone frozen) ===")
    history1 = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)

    # --- Phase 2: fine-tune the top layers of the backbone ---
    print("\n=== Phase 2: fine-tuning top backbone layers ===")
    base_model.trainable = True
    fine_tune_at = len(base_model.layers) - 30  # unfreeze last 30 layers
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    history2 = model.fit(train_ds, validation_data=val_ds, epochs=fine_tune_epochs, callbacks=callbacks)

    model.save(os.path.join(MODELS_DIR, "tomato_disease_model.keras"))

    # --- Evaluation ---
    print("\n=== Evaluating on validation set ===")
    y_true, y_pred = [], []
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    cm = confusion_matrix(y_true, y_pred).tolist()

    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"Final validation accuracy: {val_acc:.4f}")

    results = {
        "val_accuracy": float(val_acc),
        "val_loss": float(val_loss),
        "classification_report": report,
        "confusion_matrix": cm,
        "class_names": class_names,
    }
    with open(os.path.join(OUTPUTS_DIR, "training_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved model to {MODELS_DIR}/tomato_disease_model.keras")
    print(f"Saved evaluation results to {OUTPUTS_DIR}/training_results.json")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to training data folder (one subfolder per class)")
    parser.add_argument("--val_data", type=str, default=None, help="Path to a separate validation folder (one subfolder per class). If omitted, --data is auto-split 80/20.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--fine_tune_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    main(args.data, val_dir=args.val_data, epochs=args.epochs, fine_tune_epochs=args.fine_tune_epochs, batch_size=args.batch_size)
