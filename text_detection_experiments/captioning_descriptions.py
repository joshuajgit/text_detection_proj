import string

from os import listdir
from pickle import dump
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model

# https://machinelearningmastery.com/develop-a-
# deep-learning-caption-generation-model-in-python/

import cv2

dir_dataset = "datasets/Flickr8k_Dataset"
dir_text = "datasets/Flickr8k_text"

features_file = "datasets/features.pkl"

extract_feature_step = False

def extract_features(directory):
    model = VGG16()
    print ("loaded model")

    features = dict()

    print ("extracting features...")
    for name in listdir(directory):
        filename = directory + '/' + name
        image = load_img(filename, target_size=(224, 224))

        # convert the image pixels to a numpy array
        image = img_to_array(image)
        # reshape data for the model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

        # prepare image for VGG model
        image = preprocess_input(image)
        # get features
        feature = model.predict(image, verbose=0)
        # get image id
        image_id = name.split('.')[0]
        # store feature
        features[image_id] = feature

    return features

def load_doc(file):
    file = open(file, "r")
    text = file.read()
    file.close()
    return text

def load_descriptions(doc):
    descriptions = dict()

    for line in doc.split("\n"):
        tokens = line.split()
        if len(line) < 2:
            continue

        image_id, image_desc = tokens[0], tokens[1:]

        # keep only the id part
        image_id = image_id.split('.')[0]

        image_desc = ' '.join(image_desc)

        if image_id not in descriptions:
            descriptions[image_id] = []
        descriptions[image_id].append(image_desc)

    return descriptions

def clean_descriptions(descriptions):
    
    punctuation_table = str.maketrans('', '', string.punctuation)
    for keys, values in descriptions.items():
        for i in range(len(values)):
            desc = values[i]
            #print ("Before:", desc)
            desc = desc.split()
            # put everything in lowercase
            desc = [word.lower() for word in desc]
            # remove punctuation
            desc = [w.translate(punctuation_table) for w in desc]
            # remove all 1 letter words
            desc = [word for word in desc if len(word) > 1]
            # remove all non alphabetical characters
            desc = [word for word in desc if word.isalpha()]
            #print ("After:", desc)
            values[i] = ' '.join(desc)

def to_vocabulary(descriptions):
    all_desc = set()
    for key in descriptions.keys():
        [all_desc.update(d.split()) for d in descriptions[key]]
    return all_desc

def save_descriptions(descriptions, filename):
    lines = list()
    for key, value in descriptions.items():
        for i in value:
            lines.append(key + " " + i)
        data = "\n".join(lines)
        file = open(filename, "w")
        file.write(data)
        file.close()

if extract_feature_step:
    features = extract_features(dir_dataset)
    print("Extracted features:", len(features))

    # save to file
    dump(features, open('features.pkl', 'wb'))

filename = dir_text + "/" + "Flickr8k.token.txt"
doc = load_doc(filename)
descriptions = load_descriptions(doc)
print("Loaded descriptions:", len(descriptions))

clean_descriptions(descriptions)

print("New loaded descriptions:", len(descriptions))

vocabulary = to_vocabulary(descriptions)
print("Vocabulary Size:", len(vocabulary))

save_descriptions(descriptions, 'datasets/descriptions.txt')

print ("done")


