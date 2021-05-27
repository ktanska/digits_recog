import cv2
import argparse
import os
from PIL import Image
import PIL.ImageOps
import keras
import numpy as np
from keras import layers
import tensorflow as tf


def false_positives(conf_matrix):
    sums = np.sum(conf_matrix, axis=0)
    fp = np.subtract(sums, np.diagonal(conf_matrix))
    return fp


def false_negatives(conf_matrix):
    sums = np.sum(conf_matrix, axis=1)
    fn = np.subtract(sums, np.diagonal(conf_matrix))
    return fn


def recall(conf_matrix):
    tp = np.diagonal(conf_matrix)
    fn = false_negatives(conf_matrix)
    return (tp / (tp + fn))


def precision(conf_matrix):
    tp = np.diagonal(conf_matrix)
    fp = false_positives(conf_matrix)
    return (tp / (tp + fp))


def train():
    print("Rozpoczecie funkcji trenujacej na bazie mnist")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    num_classes = 10
    input_shape = (28, 28, 1)

    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    print("Data type for input: ", x_train.dtype)
    print("Tensor shape for input: ", x_train.shape)

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.summary()

    batch_size = 128
    epochs = 15

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

    score = model.evaluate(x_test, y_test, verbose=0)
    test_examples = y_test.shape[0]
    y = model.predict(x_test[:test_examples])
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

    model.save('new_model.h5', save_format='h5')

    detections = tf.math.argmax(y[:test_examples], 1)
    true_values = tf.math.argmax(y_test[:test_examples], 1)
    cm = tf.math.confusion_matrix(true_values, detections)
    print("Confusion Matrix: ")
    print(cm.numpy())

    # Define functions to count TP, FP, FN, recall and precision for each class

    print("Recall: ", 100 * recall(cm))
    print("Precision: ", 100 * precision(cm))
    print("FP: ", false_positives(cm))
    print("FN: ", false_negatives(cm))

    File_object = open(r"wyniki_dla_bazy_mnist.txt", "a")
    File_object.write("Macierz pomylek \n")
    File_object.write(str(cm) + "\n")
    File_object.write("Recall\n")
    File_object.write(str(100 * recall(cm)) + "\n")
    File_object.write("Precision \n")
    File_object.write(str(100 * precision(cm))+ "\n")
    File_object.write("FP " + str(false_positives(cm)) + "\n")
    File_object.write("FN " + str(false_negatives(cm)) + "\n")
    File_object.close()

if __name__ == '__main__':
    train()