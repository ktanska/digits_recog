import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import argparse
import os
from PIL import Image
import PIL.ImageOps

def load_images_to_data(image_label, image_directory, features_data, label_data):
    list_of_files = os.listdir(image_directory)
    for file in list_of_files:
        image_label = file[0]
        image_file_name = os.path.join(image_directory, file)
        if ".jpg" in image_file_name:
            print(image_file_name)
            img = Image.open(image_file_name).convert("L")
            img = np.resize(img, (28,28,1))
            im2arr = np.array(img)
            im2arr = im2arr.reshape(1,28,28,1)
            features_data = np.append(features_data, im2arr, axis=0)
            label_data = np.append(label_data, [image_label], axis=0)
    return features_data, label_data

def train():
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
    epochs = 1

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

    score = model.evaluate(x_test, y_test, verbose=0)
    test_examples = y_test.shape[0]

    for i in range(0,10):
        for j in range(0,4):
            nazwa = 'skany/Kinia/' + str(i) + "." + str(j) + ".jpg"
            img = Image.open(nazwa).convert("L")
            inverted_image = PIL.ImageOps.invert(img)
            inverted_image.save(nazwa.replace('jpg','png'))
            img = Image.open(nazwa.replace('jpg','png')).convert("L")
            img = np.resize(img, (28, 28, 1))
            im2arr = np.array(img)
            im2arr = im2arr.reshape(1, 28, 28, 1)
            y = model.predict_classes(im2arr)
            print(nazwa)
            print(y)
            print("Test loss:", score[0])
            print("Test accuracy:", score[1])

    # detections = tf.math.argmax(y[:test_examples], 1)
    # true_values = tf.math.argmax(y_test[:test_examples], 1)
    # cm = tf.math.confusion_matrix(true_values, detections)
    # print("Confusion Matrix: ")
    # print(cm.numpy())
    #
    # # Define functions to count TP, FP, FN, recall and precision for each class
    #
    # def false_positives(conf_matrix):
    #     sums = np.sum(conf_matrix, axis=0)
    #     fp = np.subtract(sums, np.diagonal(conf_matrix))
    #     return fp
    #
    # def false_negatives(conf_matrix):
    #     sums = np.sum(conf_matrix, axis=1)
    #     fn = np.subtract(sums, np.diagonal(conf_matrix))
    #     return fn
    #
    # def recall(conf_matrix):
    #     tp = np.diagonal(conf_matrix)
    #     fn = false_negatives(conf_matrix)
    #     return (tp / (tp + fn))
    #
    # def precision(conf_matrix):
    #     tp = np.diagonal(conf_matrix)
    #     fp = false_positives(conf_matrix)
    #     return (tp / (tp + fp))
    #
    # print("Recall: ", 100 * recall(cm))
    # print("Precision: ", 100 * precision(cm))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--train', '-t', default='False', help='train mode')
    args = parser.parse_args()
    print(args.train)
    #if args.train == "True":
    train()
