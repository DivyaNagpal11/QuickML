import numpy as np
import pandas as pd
from .spark_database import *
import os
import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from PIL import Image
import cv2
import keract
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import tensorflow as tf
from tensorflow import Graph, Session
from keras.models import load_model


def get_categories_path(project_id):
    directory = r"mlp/static/Resources/" + str(project_id)
    files = os.listdir(directory)
    category = []
    for name in files:
        for sub_file in os.listdir(os.path.join(directory, name)):
            path = str(project_id) + "/" + name + "/" + sub_file
            category.append(path)
    return category


def visualizing(project_id):
    path = []
    path = get_categories_path(project_id)
    count = len(path)
    sub_path = []
    for val in range(count):
        location = r"mlp/static/Resources/" + path[val]
        for a in os.listdir(location):
            sub_path.append(path[val] + "/" + a)
            break
    return sub_path


def create_img_labels(project_id):
    x = 0
    img = []
    labels = []
    category = get_categories_path(project_id)
    for category_path in category:
        for a in os.listdir(r"mlp/static/Resources/" + category_path):
            try:
                image = cv2.imread(r"mlp/static/Resources/" + category_path + "/" + a)
                img_to_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
                img_to_yuv[:, :, 0] = cv2.equalizeHist(img_to_yuv[:, :, 0])
                hist_equalization_result = cv2.cvtColor(img_to_yuv, cv2.COLOR_YUV2RGB)
                image_from_array = Image.fromarray(hist_equalization_result, 'RGB')
                size_image = image_from_array.resize((50, 50))
                img.append(np.array(size_image))
                labels.append(x)
            except AttributeError:
                raise Exception()
        x += 1
    img = np.array(img)
    labels = np.array(labels)
    len_data = len(img)
    s = np.arange(img.shape[0])
    np.random.shuffle(s)
    img = img[s]
    labels = labels[s]
    (x_train, x_test) = img[int(0.1 * len_data):], img[: int(0.1 * len_data)]
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    (y_train, y_test) = labels[int(0.1 * len_data):], labels[:int(0.1 * len_data)]
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    filename = "train_" + str(project_id) + ".pickle"
    directory = r"mlp/SaveTrainDataModels/" + str(project_id)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(Path('mlp/SaveTrainDataModels/' + str(project_id) + '/' + filename), 'wb') as handle:
        pickle.dump([x_train, x_test, y_train, y_test], handle)


def display_activations(project_id, activations, save=False):
    import math
    for layer_name, acts in activations.items():
        if acts.shape[0] != 1:
            print('-> Skipped. First dimension is not 1.')
            continue
        if len(acts.shape) <= 2:
            print('-> Skipped. 2D Activations.')
            continue
        nrows = int(math.sqrt(acts.shape[-1]) - 0.001) + 1  # best square fit for the given number
        ncols = int(math.ceil(acts.shape[-1] / nrows))
        fig, axes = plt.subplots(nrows, ncols, figsize=(8, 8))
        fig.suptitle(layer_name)
        for i in range(nrows * ncols):
            if i < acts.shape[-1]:
                img = acts[0, :, :, i]
                hmap = axes.flat[i].imshow(img)
            axes.flat[i].axis('off')
        fig.subplots_adjust(right=0.8)
        cbar = fig.add_axes([0.85, 0.15, 0.03, 0.7])
        fig.colorbar(hmap, cax=cbar)
        img_name = layer_name.split('/')[0]
    directory = r"mlp/static/CnnImages/" + str(project_id)
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig("mlp/static/CnnImages/" + str(project_id) + '/' + img_name + ".png", bbox_inches='tight')
    return img_name


def convolution_function(project_id, index_no, layer_name, parameters):
    filename = "train_" + str(project_id) + ".pickle"
    directory = r"mlp/SaveTrainDataModels/" + str(project_id) + "/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(directory + filename, 'rb') as f:
        x_train, x_test, y_train, y_test = pickle.load(f)
    graph = Graph()
    tf.reset_default_graph()
    with graph.as_default():
        sess = Session()
        with sess.as_default():
            for i in range(len(index_no)):
                if layer_name[i] == "conv2d":
                    para_split = parameters[i].split(',')
                    if i == 0:
                        model = Sequential()
                        model.add(Conv2D(filters=int(para_split[0]), kernel_size=int(para_split[1]), strides=int(para_split[2]),
                                         padding=para_split[3], activation=para_split[4], input_shape=(50, 50, 3)))
                    else:
                        model.add(Conv2D(filters=int(para_split[0]), kernel_size=int(para_split[1]), strides=int(para_split[2]),
                                         padding=para_split[3], activation=para_split[4]))
                elif layer_name[i] == "maxpool":
                    para_split = parameters[i].split(',')
                    model.add(MaxPooling2D(pool_size=int(para_split[0]), strides=int(para_split[1]),padding=para_split[2]))
                else:
                    para_split = parameters[i].split(',')
                    model.add(Dropout(rate=float(para_split[0]), seed=int(para_split[1])))
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            a = keract.get_activations(model, x_test[0:1])
            lname = display_activations(project_id, a)
            del model
            return lname


def maxpooling_function(project_id, index_no, layer_name, parameters):
    filename = "train_" + str(project_id) + ".pickle"
    directory = r"mlp/SaveTrainDataModels/" + str(project_id) + "/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(directory + filename, 'rb') as f:
        x_train, x_test, y_train, y_test = pickle.load(f)
    graph = Graph()
    tf.reset_default_graph()
    with graph.as_default():
        sess = Session()
        with sess.as_default():
            for i in range(len(index_no)):
                if layer_name[i] == "conv2d":
                    para_split = parameters[i].split(',')
                    if i == 0:
                        model = Sequential()
                        model.add(Conv2D(filters=int(para_split[0]), kernel_size=int(para_split[1]), strides=int(para_split[2]),
                                         padding=para_split[3], activation=para_split[4], input_shape=(50, 50, 3)))
                    else:
                        model.add(Conv2D(filters=int(para_split[0]), kernel_size=int(para_split[1]), strides=int(para_split[2]),
                                         padding=para_split[3], activation=para_split[4]))
                elif layer_name[i] == "maxpool":
                    para_split = parameters[i].split(',')
                    model.add(MaxPooling2D(pool_size=int(para_split[0]), strides=int(para_split[1]),padding=para_split[2]))
                else:
                    para_split = parameters[i].split(',')
                    model.add(Dropout(rate=float(para_split[0]), seed=int(para_split[1])))
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            a = keract.get_activations(model, x_test[0:1])
            lname = display_activations(project_id, a)
            del model
            return lname


def dropout_function(project_id, index_no, layer_name, parameters):
    filename = "train_" + str(project_id) + ".pickle"
    directory = r"mlp/SaveTrainDataModels/" + str(project_id) + "/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(directory + filename, 'rb') as f:
        x_train, x_test, y_train, y_test = pickle.load(f)
    graph = Graph()
    tf.reset_default_graph()
    with graph.as_default():
        sess = Session()
        with sess.as_default():
            for i in range(len(index_no)):
                if layer_name[i] == "conv2d":
                    para_split = parameters[i].split(',')
                    if i == 0:
                        model = Sequential()
                        model.add(Conv2D(filters=int(para_split[0]), kernel_size=int(para_split[1]), strides=int(para_split[2]),
                                         padding=para_split[3], activation=para_split[4], input_shape=(50, 50, 3)))
                    else:
                        model.add(Conv2D(filters=int(para_split[0]), kernel_size=int(para_split[1]), strides=int(para_split[2]),
                                         padding=para_split[3], activation=para_split[4]))
                elif layer_name[i] == "maxpool":
                    para_split = parameters[i].split(',')
                    model.add(MaxPooling2D(pool_size=int(para_split[0]), strides=int(para_split[1]),padding=para_split[2]))
                else:
                    para_split = parameters[i].split(',')
                    model.add(Dropout(rate=float(para_split[0]), seed=int(para_split[1])))
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            a = keract.get_activations(model, x_test[0:1])
            lname = display_activations(project_id, a)
            del model
            return lname


def fit_function(project_id, index_no, layer_name, parameters, l_name, params, units, activation):
    filename = "train_" + str(project_id) + ".pickle"
    directory = r"mlp/SaveTrainDataModels/" + str(project_id) + "/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(directory + filename, 'rb') as f:
        x_train, x_test, y_train, y_test = pickle.load(f)
    graph = Graph()
    tf.reset_default_graph()
    with graph.as_default():
        sess = Session()
        with sess.as_default():
            for i in range(len(index_no)):
                if layer_name[i] == "conv2d":
                    para_split = parameters[i].split(',')
                    if i == 0:
                        model = Sequential()
                        model.add(
                            Conv2D(filters=int(para_split[0]), kernel_size=int(para_split[1]), strides=int(para_split[2]),
                                   padding=para_split[3], activation=para_split[4], input_shape=(50, 50, 3)))
                    else:
                        model.add(
                            Conv2D(filters=int(para_split[0]), kernel_size=int(para_split[1]), strides=int(para_split[2]),
                                   padding=para_split[3], activation=para_split[4]))
                elif layer_name[i] == "maxpool":
                    para_split = parameters[i].split(',')
                    model.add(MaxPooling2D(pool_size=int(para_split[0]), strides=int(para_split[1]), padding=para_split[2]))
                else:
                    para_split = parameters[i].split(',')
                    model.add(Dropout(rate=float(para_split[0]), seed=int(para_split[1])))
            for i in range(len(l_name)):
                if l_name[i] == "flatten":
                    model.add(Flatten())

            category = get_categories_path(project_id)

            for i in range(len(units)):
                if int(units[-1]) == len(category):
                    model.add(Dense(units=int(units[i]), activation=activation[i]))
                else:
                    del model
                    return 1

            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            model.summary()
            for i in range(len(l_name)):
                if l_name[i] == "fit":
                    para_split = params[i].split(',')
                    model.summary()
                    model.fit(x_train, y_train, batch_size=int(para_split[0]), epochs=int(para_split[1]),
                              verbose=int(para_split[2]), validation_split=float(para_split[3]))

            model.save(directory + str(project_id)+ ".h5")
            del model
            return 0


def convert_to_array(img):
    url = r"media/mlp/validationImages/" + img.split("/")[-1]
    im = cv2.imread(url)
    img_ = Image.fromarray(im, 'RGB')
    image = img_.resize((50, 50))
    return np.array(image)


def get_category_name(project_id, label):
    labels = []
    category = get_categories_path(project_id)
    for each in category:
        labels.append(each.split("/")[-1])
    return labels[label]


def predict_cell(project_id, url): #index_no, layer_name, parameters, l_name, params, units, activation):
    filename = "train_" + str(project_id) + ".pickle"
    directory = r"mlp/SaveTrainDataModels/" + str(project_id) + "/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    ar = convert_to_array(url)
    ar = ar / 255
    a = []
    a.append(ar)
    a = np.array(a)
    with open(directory + filename, 'rb') as f:
        x_train, x_test, y_train, y_test = pickle.load(f)
    graph = Graph()
    tf.reset_default_graph()
    with graph.as_default():
        sess = Session()
        with sess.as_default():
            model = load_model(directory + str(project_id) + ".h5")
            model.summary()
            score = model.predict(a, verbose=1)
            label_index = np.argmax(score)
            acc = np.max(score)
            category = get_category_name(project_id, label_index)
            del model
    return category