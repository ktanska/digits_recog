import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import argparse
from PIL import Image
import PIL.ImageOps
import torch.optim
import torch.utils.data
import itertools

def train():
    predictions = []
    references = []
    model = keras.models.load_model("mnist_model.h5")
    for i in range(0,10):
        for j in range(1,4):
            nazwa = 'skany/skany/Kinia/' + str(i) + "." + str(j) + ".jpg"
            img = Image.open(nazwa).convert("L")
            inverted_image = PIL.ImageOps.invert(img)
            inverted_image.save(nazwa.replace('jpg','png'))
            img = Image.open(nazwa.replace('jpg','png')).convert("L")
            img = np.resize(img, (28, 28, 1))
            im2arr = np.array(img)
            im2arr = im2arr.reshape(1, 28, 28, 1)
            y = model.predict(im2arr).argmax(axis=1)
            print(nazwa)
            print(y)
            predictions.append(i)
            references.append(y)

    for i in range(0,10):
        for j in range(1,3):
            nazwa = 'skany/skany/Ola/' + str(i) + "." + str(j) + ".jpg"
            img = Image.open(nazwa).convert("L")
            inverted_image = PIL.ImageOps.invert(img)
            inverted_image.save(nazwa.replace('jpg','png'))
            img = Image.open(nazwa.replace('jpg','png')).convert("L")
            img = np.resize(img, (28, 28, 1))
            im2arr = np.array(img)
            im2arr = im2arr.reshape(1, 28, 28, 1)
            y = model.predict(im2arr).argmax(axis=1)
            print(nazwa)
            print(y)
            predictions.append(i)
            references.append(y)

    for i in range(0,10):
        for j in range(1,8):
            nazwa = 'skany/skany/Julka/' + str(i) + "." + str(j) + ".JPG"
            img = Image.open(nazwa).convert("L")
            inverted_image = PIL.ImageOps.invert(img)
            inverted_image.save(nazwa.replace('JPG','png'))
            img = Image.open(nazwa.replace('JPG','png')).convert("L")
            img = np.resize(img, (28, 28, 1))
            im2arr = np.array(img)
            im2arr = im2arr.reshape(1, 28, 28, 1)
            y = model.predict(im2arr).argmax(axis=1)
            print(nazwa)
            print(y)
            predictions.append(i)
            references.append(y)

    test = []
    for i in range(0, len(references)):
        obj = [references[i], predictions[i]]
        test.append(obj)
    cmt = torch.zeros(10, 10, dtype=torch.int64)
    print(test)
    for p in test:
        tl, pl = p[0], p[1]
        cmt[tl, pl] = cmt[tl, pl] + 1
    print(cmt)
    File_object = open(r"macierz_pomylek_dla_naszych_danych.txt", "a")
    File_object.write(str(cmt) + "\n")
    File_object.close()
    classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    normalize = False
    cm = np.array(cmt)
    title = 'Confusion matrix'
    cmap = plt.cm.Blues
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(test)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    name_fig = "macierz_pomylek_dla_naszych_danych.png"
    plt.savefig(name_fig)
    print(name_fig)
    plt.clf()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--train', '-t', default='False', help='train mode')
    args = parser.parse_args()
    print(args.train)
    #if args.train == "True":
    train()
