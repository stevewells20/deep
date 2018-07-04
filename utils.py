#!/usr/bin/python3

KERAS_BACKEND='tensorflow'
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import io
import os
from keras.utils import np_utils

# fix random seed for reproducibility
numpy.random.seed(7)

def import_training_data(path_or_url):
    text = ''
    if (os.path.isfile(path_or_url):
        # load ascii text and covert to lowercase
        raw_text = open(path_or_url).read()
        text = raw_text.lower()

    else:
        path = get_file('idk.txt', origin=path_or_url)
        with io.open(path, encoding='utf-8') as f:
            text = f.read().lower()

    print('corpus length:', len(text))

    chars = sorted(list(set(text)))
    print('total chars:', len(chars))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    # cut the text in semi-redundant sequences of maxlen characters
    maxlen = 40
    step = 3
    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])
    print('nb sequences:', len(sentences))

    print('Vectorization...')
    x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1




def load_xy_csv_dataset(pathToDataset):
    # load dataset
    dataset = numpy.loadtxt(pathToDataset, delimiter=",")
    # split into input (X) and output (Y) variables
    X = dataset[:,0:8]
    Y = dataset[:,8]
    return [X,Y]

def create_a_model():
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
    return model

def compile_a_model(model):
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def fit_a_model(model, xy_data):
    # Fit the model
    model.fit(xy_data[0], xy_data[1], epochs=150, batch_size=10, verbose=0)
    return model

def eval_a_model(model, xy_data):
    # evaluate the model
    X, Y = xy_data
    scores = model.evaluate(X, Y, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

def serialize_model(loaded_model, save_name='model.json'):
    # serialize model to JSON
    model_json = loaded_model.to_json()
    with open(save_name, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(save_name+'.h5')
    print("Saved model to disk")

def load_model(pathToModel):
    # load json and create model
    json_file = open(pathToModel, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    print("Loaded model from disk")
    return loaded_model

def load_weights_into_model(loaded_model, pathToWeights):
    # load weights into new model
    loaded_model.load_weights(pathToWeights) # .h5 file
    print("Loaded weights into model")
    return loaded_model

def evaluate_loaded_model(loaded_model, test_data):
    # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    X = test_data[0]
    Y = test_data[1]
    score = loaded_model.evaluate(X, Y, verbose=0)
    print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
