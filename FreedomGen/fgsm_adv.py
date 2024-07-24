import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import logging
import sys
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# Check for GPU availability
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("GPU is available and enabled")
else:
    print("No GPU found. Running on CPU")

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


def preprocess(image):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (32, 32))
    image = preprocess_input(image)
    return image


def get_imagenet_label(probs):
    return decode_predictions(probs, top=1)[0][0]


def create_adversarial_pattern(input_image, input_label):
    input_image = tf.convert_to_tensor(input_image)
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = model(input_image)
        loss = loss_object(input_label, prediction)
    gradient = tape.gradient(loss, input_image)
    signed_grad = tf.sign(gradient)
    return signed_grad


def generate_adversarial_examples(x, y, epsilon):
    x = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)
    x_adv = create_adversarial_pattern(x, y)
    x_adv = tf.clip_by_value(x + epsilon * tf.sign(x_adv), -1, 1)
    return x_adv


def adversarial_train(epochs=5, epsilon=0.01):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = preprocess_input(x_train.astype('float32'))
    x_test = preprocess_input(x_test.astype('float32'))
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    with tf.device('/GPU:0'):
        for epoch in range(epochs):
            logging.info(f"Epoch {epoch + 1}/{epochs}")

            # Train on batches
            batch_train_loss = []
            batch_train_accuracy = []
            for i in range(0, len(x_train), 32):
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


def main():
    try:
        logging.info("Loading ResNet50 model...")
        global model, loss_object
        with tf.device('/GPU:0'):
            base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
            x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
            x = tf.keras.layers.Dense(10, activation='softmax')(x)
            model = tf.keras.Model(base_model.input, x)

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        loss_object = tf.keras.losses.CategoricalCrossentropy()

        logging.info("Starting adversarial training...")
        adversarial_train(epochs=5, epsilon=0.01)

        logging.info("Generating and saving plots...")
        plot_metrics()

        logging.info("Script completed successfully.")
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Script failed: {str(e)}")
        sys.exit(1)