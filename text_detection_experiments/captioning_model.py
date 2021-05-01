import string

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint

from os import listdir
from pickle import dump
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model

from pickle import load

# https://machinelearningmastery.com/develop-a-
# deep-learning-caption-generation-model-in-python/

import cv2

dir_dataset = "datasets/Flickr8k_Dataset"
dir_text = "datasets/Flickr8k_text"

def load_doc(filename):
    file = open(filename, "r")
    text = file.read()
    file.close()
    return text

def load_set(filename):
    doc = load_doc(filename)
    dataset = []
    for line in doc.split("\n"):
        if len(line) == 0:
            continue
        id_ = line.split(".")[0]
        dataset.append(id_)

    return set(dataset)

def load_descriptions(filename, dataset):
    doc = load_doc(filename)
    descriptions = dict()
    for line in doc.split("\n"):
        tokens = line.split()
        image_id, image_desc = tokens[0], tokens[1:]
        # skip images not in the dataset...
        if image_id in dataset:
            if image_id not in descriptions:
                # empty list of descriptions for an image id
                descriptions[image_id] = []

            desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
            descriptions[image_id].append(desc)

    return descriptions

def load_features(filename, dataset):    
    all_features = load(open(filename, "rb"))
    # filter features based on which ones are in the dataset
    features = {k: all_features[k] for k in dataset}
    return features

# convert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
    all_desc = []
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc

# fit a tokenizer given caption descriptions
def create_tokenizer(descriptions):
    lines = to_lines(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

# calculate the length of the description with the most words
def max_length(descriptions):
    lines = to_lines(descriptions)
    return max(len(d.split()) for d in lines)

def create_sequences(tokenizer, max_length, descriptions, photos, vocab_size):
    X1, X2, y = [], [], []

    for key, desc_list in descriptions.items():
        for desc in desc_list:
            seq = tokenizer.texts_to_sequences([desc])[0]
            for i in range(1, len(seq)):
                input_, output_ = seq[0:i], seq[i]
                # pad input if necessary
                input_ = pad_sequences([input_], maxlen=max_length)[0]
                # encode output
                output_ = to_categorical([output_], num_classes=vocab_size)[0]
                # store
                X1.append(photos[key][0])
                X2.append(input_)
                y.append(output_)

    return array(X1), array(X2), array(y)

def tanti_et_al_model(vocab_size, max_length):
    # feature extractor
    inputs1 = Input(shape=(4096,))
    f1 = Dropout(0.5)(inputs1)
    f2 = Dense(256, activation="relu")(f1)
    # sequence
    inputs2 = Input(shape=(max_length,))
    s1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    s2 = Dropout(0.5)(s1)
    s3 = LSTM(256)(s2)
    # decoder
    d1 = add([f2, s3])
    d2 = Dense(256, activation="relu")(d1)
    outputs = Dense(vocab_size, activation="softmax")(d2)
    # compile
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model

descriptions_ = "datasets/descriptions.txt"

# load training dataset (6K)
filename = dir_text + "/Flickr_8k.trainImages.txt"
train = load_set(filename)
print('Dataset:', len(train))

# descriptions
train_descriptions = load_descriptions(descriptions_, train)
print('Descriptions: train=', len(train_descriptions))

# photo features
train_features = load_features('datasets/features.pkl', train)
print('Photos: train=', len(train_features))

# prepare tokenizer
tokenizer = create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary size:', vocab_size)

# all inputs have to be same length
max_length = max_length(train_descriptions)

X1train, X2train, ytrain = create_sequences(tokenizer, max_length, train_descriptions, train_features, vocab_size)

# load test dataset (6K)
filename = dir_text + "/Flickr_8k.testImages.txt"
test = load_set(filename)

# descriptions
train_descriptions = load_descriptions(descriptions_, test)
print('Descriptions: test=', len(test_descriptions))

# photo features
train_features = load_features('datasets/features.pkl', test)
print('Photos: test=', len(test_features))

X1test, X2test, ytest = create_sequences(tokenizer, max_length, test_descriptions, test_features, vocab_size)

# define checkpoint callback
filepath = "model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor="val_loss", verbose=1, save_best_only=True, mode="min")

model.fit([X1train, X2train], ytrain, epochs=20, verbose=2, callbacks=[checkpoint], validation_data=([X1test, X2test], ytest))

