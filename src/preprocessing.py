"""
Image preprocessing & augmentation utilities for the Tomato Disease
Detection System.
"""
import tensorflow as tf

IMG_SIZE = (224, 224)   # matches MobileNetV2 / most transfer-learning backbones
BATCH_SIZE = 32


def load_datasets(data_dir, img_size=IMG_SIZE, batch_size=BATCH_SIZE, val_split=0.2, seed=42):
    """
    Loads images from a directory structured as:
        data_dir/
            ClassName1/*.jpg
            ClassName2/*.jpg
            ...
    Returns (train_ds, val_ds, class_names).
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="training",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="validation",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
    )
    class_names = train_ds.class_names

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=autotune)
    val_ds = val_ds.cache().prefetch(buffer_size=autotune)

    return train_ds, val_ds, class_names


def load_datasets_from_train_val(train_dir, val_dir, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    """
    Use this when the dataset already ships with separate train/ and val/
    folders (each containing one subfolder per class), instead of a single
    folder that needs an automatic split.
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
    )
    class_names = train_ds.class_names

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=autotune)
    val_ds = val_ds.cache().prefetch(buffer_size=autotune)

    return train_ds, val_ds, class_names


def build_augmentation_layer():
    """Light augmentation appropriate for leaf images (no vertical flips - leaves have a canonical up/down)."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ], name="augmentation")


def preprocess_single_image(image_bytes, img_size=IMG_SIZE):
    """Decode raw image bytes (as received by the API) into a model-ready batch of 1."""
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.image.resize(image, img_size)
    image = tf.expand_dims(image, axis=0)
    return image
