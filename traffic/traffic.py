import cv2
import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow import keras
from keras import layers

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():
    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()
    # model = build_model_with_inception()
    # model = load_model("model_for_img_recg")
    model.summary()
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test, y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """

    images = []
    labels = []
    for i in range(NUM_CATEGORIES):
        subdir_path = os.path.join(data_dir, f'{i}')

        files = os.listdir(subdir_path)
        for file in files:
            image_path = os.path.join(subdir_path, file)
            if os.path.isfile(image_path):
                uniform_image_array = cv2.resize(cv2.imread(image_path), (IMG_WIDTH, IMG_HEIGHT))
                images.append(uniform_image_array)
                labels.append(i)
    return images, labels


def load_model(filename) -> keras.models:
    return keras.models.load_model(filename)


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    if tf.config.list_physical_devices('GPU'):
        print("TensorFlow is using the gpu")

    # Creation of a convolutional neural network
    model = keras.models.Sequential()
    model.add(keras.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3)))

    # Convolution
    model.add(keras.layers.Conv2D(16, (3, 3), activation="relu"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(keras.layers.Conv2D(32, (3, 3), activation="relu"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(keras.layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # model.add(keras.layers.Conv2D(32, (3, 3), activation="relu"))
    # model.add(keras.layers.Conv2D(32, (3, 3), activation="relu"))

    # Pooling
    # model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # Flattening
    model.add(keras.layers.Flatten())

    # Hidden layers
    model.add(keras.layers.Dense(512, activation="relu"))

    # Dropout
    # model.add(keras.layers.Dropout(0.3))

    # Output layer
    model.add(keras.layers.Dense(NUM_CATEGORIES, activation="softmax"))

    # Compile the model
    model.compile(optimizer="adam",
                  loss=keras.losses.categorical_crossentropy,
                  metrics=["accuracy"])
    return model


def build_model_with_inception():
    from keras.applications.inception_v3 import InceptionV3
    inception_model = InceptionV3(include_top=False,
                                  weights=None,
                                  input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
    inception_model.load_weights("inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5")

    for layer in inception_model.layers:
        layer.trainable = False

    # Get the bottleneck layer
    last_layer = inception_model.get_layer('mixed7')
    last_output = last_layer.output

    x = keras.layers.Flatten()(last_output)
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(NUM_CATEGORIES, activation="softmax")(x)

    model = keras.Model(inception_model.input, x)
    print(inception_model.layers[0].input_shape)
    # Compile the model
    model.compile(optimizer="adam",
                  loss=keras.losses.categorical_crossentropy,
                  metrics=["accuracy"])
    return model


if __name__ == "__main__":
    main()
