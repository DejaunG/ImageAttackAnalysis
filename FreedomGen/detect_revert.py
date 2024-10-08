import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import os
import logging
import sys

# Set up logging
logging.basicConfig(filename='detect_revert.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)


def load_custom_dataset(train_dir, val_dir, img_size=(224, 224), batch_size=8):
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2
    )

    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, validation_generator


def create_model(num_classes):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers:
        layer.trainable = False
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(base_model.input, x)
    return model


def train_model(model, train_generator, validation_generator, epochs=2):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    steps_per_epoch = min(len(train_generator), 10)  # Limit steps per epoch
    validation_steps = min(len(validation_generator), 5)  # Limit validation steps

    try:
        history = model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=validation_steps
        )
    except Exception as e:
        logging.warning(f"Training interrupted: {str(e)}")
        history = None

    return history


def load_and_preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = preprocess_input(img_array[np.newaxis, ...])
    return img_array


def revert_image(image):
    # Convert to PIL Image
    img_pil = Image.fromarray(np.uint8(image[0] * 127.5 + 127.5))

    # Apply a slight Gaussian blur to reduce noise
    img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=0.5))

    # Enhance sharpness
    enhancer = ImageEnhance.Sharpness(img_pil)
    img_pil = enhancer.enhance(1.5)

    # Enhance contrast slightly
    enhancer = ImageEnhance.Contrast(img_pil)
    img_pil = enhancer.enhance(1.2)

    # Convert back to numpy array and normalize
    reverted_image = np.array(img_pil, dtype=np.float32)
    reverted_image = (reverted_image - 127.5) / 127.5
    reverted_image = np.expand_dims(reverted_image, axis=0)

    return reverted_image


def detect_and_revert(model, image):
    # Predict the class of the original image
    prediction = model.predict(image)
    class_index = np.argmax(prediction)

    # Assuming class 0 is "normal" and class 1 is "adversarial"
    is_adversarial = class_index == 1
    confidence = prediction[0][class_index]

    # Revert the image
    reverted_image = revert_image(image)

    # Predict the class of the reverted image
    reverted_prediction = model.predict(reverted_image)
    reverted_class_index = np.argmax(reverted_prediction)
    reverted_confidence = reverted_prediction[0][reverted_class_index]

    return is_adversarial, confidence, reverted_image, reverted_class_index == 0, reverted_confidence


def main(train_dir, val_dir, uploaded_image_path):
    try:
        logging.info("Loading custom dataset...")
        train_generator, validation_generator = load_custom_dataset(train_dir, val_dir)

        num_classes = len(train_generator.class_indices)
        logging.info(f"Number of classes: {num_classes}")

        logging.info("Creating and training model...")
        model = create_model(num_classes)
        train_model(model, train_generator, validation_generator)

        logging.info("Processing uploaded image...")
        image = load_and_preprocess_image(uploaded_image_path)

        logging.info("Detecting and reverting tampering...")
        is_adversarial, confidence, reverted_image, is_reverted_normal, reverted_confidence = detect_and_revert(model,
                                                                                                                image)
        # Log and print the results
        original_result = "Adversarial" if is_adversarial else "Normal"
        reverted_result = "Normal" if is_reverted_normal else "Adversarial"
        logging.info(f"Original image classification: {original_result} (Confidence: {confidence:.2f})")
        logging.info(f"Reverted image classification: {reverted_result} (Confidence: {reverted_confidence:.2f})")
        print(f"Original image classification: {original_result} (Confidence: {confidence:.2f})")
        print(f"Reverted image classification: {reverted_result} (Confidence: {reverted_confidence:.2f})")

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image[0] / 2 + 0.5)  # Denormalize
        plt.title(f"Original Image\n{original_result} (Conf: {confidence:.2f})")
        plt.subplot(1, 2, 2)
        plt.imshow(reverted_image[0] / 2 + 0.5)  # Denormalize
        plt.title(f"Reverted Image\n{reverted_result} (Conf: {reverted_confidence:.2f})")
        plt.savefig('generatedimages/detected_reverted.png')
        plt.close()

        logging.info("Detection and reversion completed.")
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python detect_revert.py <train_dir> <val_dir> <uploaded_image_path>")
        sys.exit(1)

    train_dir = sys.argv[1]
    val_dir = sys.argv[2]
    uploaded_image_path = sys.argv[3]
    main(train_dir, val_dir, uploaded_image_path)