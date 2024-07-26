import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import logging
import sys
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Set up logging
logging.basicConfig(filename='fgsm_adv.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize lists to store metrics
train_accuracy = []
val_accuracy = []
train_loss = []
val_loss = []


def process_single_image(image_path, model, epsilon_values=[0, 0.01, 0.1, 0.15, 0.2]):
    logging.info(f"Processing image: {image_path}")
    # Load and preprocess the image
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array[np.newaxis, ...])

    # Get the original prediction
    original_pred = model.predict(img_array)
    original_label = decode_predictions(original_pred)[0][0]

    # Generate adversarial examples
    adversarial_examples = []
    for epsilon in epsilon_values:
        adv_x = generate_adversarial_example(model, img_array, epsilon)
        adversarial_examples.append(adv_x)

    # Plot and save the results
    plt.figure(figsize=(20, 10))
    plt.subplot(2, 3, 1)
    plt.imshow(img)
    plt.title(f"Original: {original_label[1]}")

    for i, (epsilon, adv_x) in enumerate(zip(epsilon_values, adversarial_examples)):
        plt.subplot(2, 3, i + 2)
        plt.imshow(adv_x[0] / 2 + 0.5)  # Denormalize
        adv_pred = model.predict(adv_x)
        adv_label = decode_predictions(adv_pred)[0][0]
        plt.title(f"Epsilon {epsilon}: {adv_label[1]}")

    plt.tight_layout()
    plt.savefig('generatedimages/adversarial_examples.png')
    plt.close()
    logging.info("Adversarial examples generated and saved.")


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


def create_adversarial_pattern(input_image, input_label):
    input_image = tf.convert_to_tensor(input_image, dtype=tf.float32)
    input_label = tf.convert_to_tensor(input_label, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = model(input_image)
        loss = loss_object(input_label, prediction)
    gradient = tape.gradient(loss, input_image)
    signed_grad = tf.sign(gradient)
    return signed_grad


def generate_adversarial_examples(x, y, epsilon):
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.float32)
    x_adv = create_adversarial_pattern(x, y)
    x_adv = tf.clip_by_value(x + epsilon * tf.sign(x_adv), -1, 1)
    return x_adv


def adversarial_train(epochs=2, epsilon=0.01, num_samples=5000):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Use only a subset of the data
    x_train = x_train[:num_samples]
    y_train = y_train[:num_samples]
    x_test = x_test[:1000]
    y_test = y_test[:1000]

    x_train = preprocess_input(x_train.astype('float32'))
    x_test = preprocess_input(x_test.astype('float32'))
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    for epoch in range(epochs):
        logging.info(f"Epoch {epoch + 1}/{epochs}")

        # Train on batches
        batch_train_loss = []
        batch_train_accuracy = []
        for i in range(0, len(x_train), 32):
            if i % 320 == 0:  # Log every 10 batches
                logging.info(f"Processing batch {i // 32 + 1}/{len(x_train) // 32}")

            x_batch = x_train[i:i + 32]
            y_batch = y_train[i:i + 32]

            # Generate adversarial examples
            adv_x = generate_adversarial_examples(x_batch, y_batch, epsilon)

            # Train on clean and adversarial examples
            history = model.fit(
                tf.concat([x_batch, adv_x], axis=0),
                tf.concat([y_batch, y_batch], axis=0),
                epochs=1,
                verbose=0
            )
            batch_train_loss.append(history.history['loss'][0])
            batch_train_accuracy.append(history.history['accuracy'][0])

        # Evaluate on validation set
        val_loss_value, val_accuracy_value = model.evaluate(x_test, y_test, verbose=0)

        # Store metrics
        train_accuracy.append(np.mean(batch_train_accuracy))
        train_loss.append(np.mean(batch_train_loss))
        val_accuracy.append(val_accuracy_value)
        val_loss.append(val_loss_value)

        logging.info(f"Train Accuracy: {train_accuracy[-1]:.4f}, Validation Accuracy: {val_accuracy[-1]:.4f}")
        logging.info(f"Train Loss: {train_loss[-1]:.4f}, Validation Loss: {val_loss[-1]:.4f}")


def plot_metrics():
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_accuracy, label='Training Accuracy')
    plt.plot(val_accuracy, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('generatedimages/training_validation_plots.png')
    plt.close()


def main(uploaded_image_path):
    try:
        logging.info("Loading MobileNetV2 model...")
        global model, loss_object

        # Load the base model for adversarial example generation
        base_model = MobileNetV2(weights='imagenet')

        # Process the uploaded image
        if os.path.exists(uploaded_image_path):
            process_single_image(uploaded_image_path, base_model)
        else:
            logging.warning(f"Uploaded image not found at {uploaded_image_path}")

        # Create a new model for adversarial training
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
        x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        x = tf.keras.layers.Dense(10, activation='softmax')(x)
        model = tf.keras.Model(base_model.input, x)

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        loss_object = tf.keras.losses.CategoricalCrossentropy()

        logging.info("Starting adversarial training...")
        adversarial_train(epochs=2, epsilon=0.01, num_samples=5000)

        logging.info("Generating and saving plots...")
        plot_metrics()

        logging.info("Script completed successfully.")
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        if len(sys.argv) > 1:
            uploaded_image_path = sys.argv[1]
            main(uploaded_image_path)
        else:
            logging.error("No image path provided")
            sys.exit(1)
    except Exception as e:
        logging.error(f"Script failed: {str(e)}")
        sys.exit(1)