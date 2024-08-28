import scipy
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
import matplotlib.pyplot as plt
import numpy as np
import os
import logging
import sys

# Set up logging
logging.basicConfig(filename='fgsm_adv.log', level=logging.INFO,
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

def train_model(model, train_generator, validation_generator, epochs=20):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}

    for epoch in range(epochs):
        logging.info(f"Epoch {epoch + 1}/{epochs}")

        # Train on batches
        train_loss = []
        train_accuracy = []
        for _ in range(len(train_generator)):
            x_batch, y_batch = next(train_generator)
            metrics = model.train_on_batch(x_batch, y_batch)
            train_loss.append(metrics[0])
            train_accuracy.append(metrics[1])

        # Validate
        val_loss = []
        val_accuracy = []
        for _ in range(len(validation_generator)):
            x_batch, y_batch = next(validation_generator)
            metrics = model.test_on_batch(x_batch, y_batch)
            val_loss.append(metrics[0])
            val_accuracy.append(metrics[1])

        # Log metrics
        train_loss = np.mean(train_loss)
        train_accuracy = np.mean(train_accuracy)
        val_loss = np.mean(val_loss)
        val_accuracy = np.mean(val_accuracy)

        logging.info(f"train_loss: {train_loss:.4f}, train_accuracy: {train_accuracy:.4f}")
        logging.info(f"val_loss: {val_loss:.4f}, val_accuracy: {val_accuracy:.4f}")

        history['accuracy'].append(train_accuracy)
        history['val_accuracy'].append(val_accuracy)
        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)

    return history

def generate_adversarial_example(model, image, epsilon):
    image_tensor = tf.convert_to_tensor(image)
    with tf.GradientTape() as tape:
        tape.watch(image_tensor)
        prediction = model(image_tensor)
        loss = tf.keras.losses.categorical_crossentropy(prediction, prediction)
    gradient = tape.gradient(loss, image_tensor)
    signed_grad = tf.sign(gradient)
    adversarial_example = image_tensor + epsilon * signed_grad
    return tf.clip_by_value(adversarial_example, -1, 1)

def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('generatedimages/training_history.png')
    plt.close()

def process_single_image(image_path, model, epsilon_values=[0, 0.01, 0.1, 0.15, 0.2]):
    logging.info(f"Processing image: {image_path}")
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = preprocess_input(img_array[np.newaxis, ...])

    original_pred = model.predict(img_array)
    class_names = ['fresh', 'non-fresh']  # Dataset Class Names
    original_label = class_names[np.argmax(original_pred)]
    original_prob = np.max(original_pred)

    adversarial_examples = []
    for epsilon in epsilon_values:
        adv_x = generate_adversarial_example(model, img_array, epsilon)
        adversarial_examples.append(adv_x)

        # Save individual adversarial image
        plt.figure(figsize=(5, 5))
        plt.imshow(adv_x[0] / 2 + 0.5)  # Denormalize
        adv_pred = model.predict(adv_x)
        adv_label = class_names[np.argmax(adv_pred)]
        adv_prob = np.max(adv_pred)
        plt.title(f"Epsilon {epsilon}: {adv_label} ({adv_prob:.2f})")
        plt.axis('off')
        plt.savefig(f'generatedimages/adversarial_epsilon_{epsilon}.png', bbox_inches='tight')
        plt.close()

    plt.figure(figsize=(20, 10))
    plt.subplot(2, 3, 1)
    plt.imshow(img)
    plt.title(f"Original: {original_label} ({original_prob:.2f})")

    for i, (epsilon, adv_x) in enumerate(zip(epsilon_values, adversarial_examples)):
        plt.subplot(2, 3, i + 2)
        plt.imshow(adv_x[0] / 2 + 0.5)  # Denormalize
        adv_pred = model.predict(adv_x)
        adv_label = class_names[np.argmax(adv_pred)]
        adv_prob = np.max(adv_pred)
        plt.title(f"Epsilon {epsilon}: {adv_label} ({adv_prob:.2f})")

    plt.tight_layout()
    plt.savefig('generatedimages/adversarial_examples.png')
    plt.close()
    logging.info("Adversarial examples generated and saved.")

def main(train_dir, val_dir, uploaded_image_path):
    try:
        logging.info("Loading custom dataset...")
        train_generator, validation_generator = load_custom_dataset(train_dir, val_dir)

        num_classes = len(train_generator.class_indices)
        logging.info(f"Number of classes: {num_classes}")

        logging.info("Creating and training model...")
        model = create_model(num_classes)
        history = train_model(model, train_generator, validation_generator)

        logging.info("Generating and saving training plots...")
        plot_training_history(history)

        if os.path.exists(uploaded_image_path):
            logging.info("Processing uploaded image...")
            process_single_image(uploaded_image_path, model)
        else:
            logging.warning(f"Uploaded image not found at {uploaded_image_path}")

        logging.info("Script completed successfully.")
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python fgsm_adv.py <train_dir> <val_dir> <uploaded_image_path>")
        sys.exit(1)

    train_dir = sys.argv[1]
    val_dir = sys.argv[2]
    uploaded_image_path = sys.argv[3]
    main(train_dir, val_dir, uploaded_image_path)