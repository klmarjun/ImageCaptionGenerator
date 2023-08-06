# Importing Packages
from numpy import array
from pickle import load
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import Model
from keras.layers import Input,Dense,LSTM,Embedding,Dropout
from tensorflow.keras.layers import concatenate
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from pickle import dump
from keras.applications.vgg16 import VGG16,preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.models import Model
import numpy as np
import pandas as pd
from nltk.translate.bleu_score import corpus_bleu
import urllib.request
from keras.models import load_model
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import string
from os import listdir
import os
from skimage import io
from pickle import dump
from collections import OrderedDict,Counter 
from sklearn.decomposition import PCA
# Loading Data & Feature Extracts 
train_text = "../Code/flickr8k1/Flickr8k_text/Flickr_8k.trainImages.txt"
test_text = "../Code/flickr8k1/Flickr8k_text/Flickr_8k.testImages.txt"
dev_text = "../Code/flickr8k1/Flickr8k_text/Flickr_8k.devImages.txt"
tokens = "../Code/flickr8k1/Flickr8k_text/Flickr8k.token.txt"
images = "../Code/flickr8k/Images/"
vgg_features = "../Code/flickr8k1/Flickr8k_features/VGG16_Features.pkl"
# Loading Data Function
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text
 
filename = tokens
# load descriptions
doc = load_doc(filename)
# Loading Descriptions for all images
def load_descriptions(doc):
    mapping = dict()
    
    # process lines
    for line in doc.split('\n'):
        
        # split line by white space
        tokens = line.split()
        if len(line) < 2:
            continue
            
        # take the first token as the image id, the rest as the description
        image_id, image_desc = tokens[0], tokens[1:]
        
        # remove filename from image id
        image_id = image_id.split('.')[0]
        
        # convert description tokens back to string
        image_desc = ' '.join(image_desc)
        
        # create the list if needed
        if image_id not in mapping:
            mapping[image_id] = list()
            
        # store description
        mapping[image_id].append(image_desc)
    return mapping

# Parsing descriptions
descriptions = load_descriptions(doc)
# Data Cleaning
def clean_descriptions(descriptions):
    
    # prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)

    # Iterating through descriptions
    for key, desc_list in descriptions.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]
            
            # tokenize
            desc = desc.split()
            
            # convert to lower case
            desc = [word.lower() for word in desc]
            
            # remove punctuation from each token
            desc = [w.translate(table) for w in desc]
            
            # remove hanging 's' and 'a'
            desc = [word for word in desc if len(word)>1]
            
            # remove tokens with numbers in them
            desc = [word for word in desc if word.isalpha()]
            
            # store as string
            desc_list[i] =  ' '.join(desc)

# clean descriptions
clean_descriptions(descriptions)
# convert the loaded descriptions into a vocabulary of words
def to_vocabulary(descriptions):
    # build a list of all description strings
    all_desc = set()
    for key in descriptions.keys():
        [all_desc.update(d.split()) for d in descriptions[key]]
    return all_desc

# summarize vocabulary
vocabulary = to_vocabulary(descriptions)
print('Vocabulary Size: %d' % len(vocabulary))
# saving descriptions to file in the format of one per line
def save_descriptions(descriptions, filename):
    lines = list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + ' ' + desc)
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()

# save descriptions
save_descriptions(descriptions, 'descriptions.txt')
# Downloading VGG Weights and loading them
#!wget 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5' --no-check-certificat

modelvgg = VGG16(include_top=True)
## load the locally saved weights 
modelvgg.load_weights("./vgg16_weights_tf_dim_ordering_tf_kernels.h5")
modelvgg.summary()
# Popping the output layer
modelvgg_new = Model(inputs=modelvgg.inputs, outputs=modelvgg.layers[-2].output)
## show the deep learning model
modelvgg_new.summary()
# Loading Train Data
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

# Load a pre-defined list of photo identifiers
def load_set(filename):
    doc = load_doc(filename)
    dataset = list()
    # process line by line
    for line in doc.split('\n'):
        # skip empty lines
        if len(line) < 1:
            continue
        # get the image identifier
        identifier = line.split('.')[0]
        dataset.append(identifier)
    return set(dataset)
# Function for cleaning descriptions
def load_clean_descriptions(filename, dataset):
    # load document
    doc = load_doc(filename)
    descriptions = dict()
    for line in doc.split('\n'):
        # split line by white space
        tokens = line.split()
        # split id from description
        image_id, image_desc = tokens[0], tokens[1:]
        # skip images not in the set
        if image_id in dataset:
            # create list
            if image_id not in descriptions:
                descriptions[image_id] = list()
            # wrap description in tokens
            desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
            # store
            descriptions[image_id].append(desc)
    return descriptions
 
# Function for loading photo features
def load_photo_features(filename, dataset):
    # load all features
    all_features = load(open(filename, 'rb'))
    # filter features
    features = {k: all_features[k] for k in dataset}
    return features
# Applying all functions
filename = train_text

# loading train dataset
train = load_set(filename)
print('Dataset: %d' % len(train))

# descriptions
train_descriptions = load_clean_descriptions('descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))

# photo features
train_features = load_photo_features("../Code/flickr8k1/Flickr8k_features/VGG16_Features.pkl", train)
print('Photos: train=%d' % len(train_features))
# Applying all functions
filename = test_text

# loading train dataset
test = load_set(filename)
print('Dataset: %d' % len(test))

# descriptions
test_descriptions = load_clean_descriptions('descriptions.txt', test)
print('Descriptions: test=%d' % len(test_descriptions))

# photo features
test_features = load_photo_features("../Code/flickr8k1/Flickr8k_features/VGG16_Features.pkl", test)
print('Photos: test=%d' % len(test_features))
# Function to convert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
    all_desc = list()
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc

# Fit a tokenizer given caption descriptions
def create_tokenizer(descriptions):
    lines = to_lines(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

# Initializing tokenizer and 
tokenizer = create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
# create sequences of images, input sequences and output words for an image
def create_sequences(tokenizer, max_length, desc_list, photo, vocab_size):
    X1, X2, y = list(), list(), list()
    # walk through each description for the image
    for desc in desc_list:
        # encode the sequence
        seq = tokenizer.texts_to_sequences([desc])[0]
        # split one sequence into multiple X,y pairs
        for i in range(1, len(seq)):
            # split into input and output pair
            in_seq, out_seq = seq[:i], seq[i]
            # pad input sequence
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            # encode output sequence
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            # store
            X1.append(photo)
            X2.append(in_seq)
            y.append(out_seq)
    return array(X1), array(X2), array(y)
# This function will calculate the length of the description with the most words
def max_length(descriptions):
    lines = to_lines(descriptions)
    return max(len(d.split()) for d in lines)
max_length = max_length(train_descriptions)
# data generator, intended to be used in a call to model.fit_generator()
def data_generator(descriptions, photos, tokenizer, max_length, vocab_size):
    # loop for ever over images
    while 1:
        for key, desc_list in descriptions.items():
            # retrieve the photo feature
            photo = photos[key][0]
            in_img, in_seq, out_word = create_sequences(tokenizer, max_length, desc_list, photo, vocab_size)
            yield [in_img, in_seq], out_word
# Function for training model
def define_model(vocab_size, max_length):
    # feature extractor model
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    # sequence model
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    # decoder model
    decoder1 = concatenate([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    # tie it together [image, seq] [word]
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    # summarize model
    model.summary()
    return model
# define the model
model = define_model(vocab_size, max_length)
# train the model, run epochs manually and save after each epoch
epochs = 1
steps = len(train_descriptions)
for i in range(epochs):
    # create the data generator
    generator = data_generator(train_descriptions, train_features, tokenizer, max_length, vocab_size)
    # fit for one epoch
    model.fit_generator(generator, epochs=1, steps_per_epoch=steps/128, verbose=1)
    # save model
    model.save('model_' + str(i) + '.h5')
# map an integer to a word
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
    # seed the generation process
    in_text = 'startseq'
    # iterate over the whole length of the sequence
    for i in range(max_length):
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad input
        sequence = pad_sequences([sequence], maxlen=max_length)
        # predict next word
        yhat = model.predict([photo,sequence], verbose=0)
        # convert probability to integer
        yhat = np.argmax(yhat)
        # map integer to word
        word = word_for_id(yhat, tokenizer)
        # stop if we cannot map the word
        if word is None:
            break
        # append as input for generating the next word
        in_text += ' ' + word
        # stop if we predict the end of the sequence
        if word == 'endseq':
            break
    return in_text
# Extract features from each photo in the directory
def extract_features(filename):
    # load the model
    model = VGG16()
    # re-structure the model
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    # load the photo
    image = load_img(filename, target_size=(224, 224))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    # get features
    feature = model.predict(image, verbose=0)
    return feature
# Loading Random Images
path_1 = "./img1.jpg"
img_1 = urllib.request.urlretrieve("https://st3.depositphotos.com/3825437/12731/i/950/depositphotos_127311372-stock-photo-two-children-playing-on-beach.jpg", path_1) 
path_2 = "./img2.jpg"
img_2 = urllib.request.urlretrieve("data:image/jpeg;base64,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", path_2) 
path_3 = "./img3.jpg"
img_3 = urllib.request.urlretrieve("https://images.unsplash.com/photo-1565076633790-b0deb5d527c7?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxzZWFyY2h8Mnx8c25vdyUyMGhpa2luZ3xlbnwwfHwwfHw%3D&w=1000&q=80", path_3) 
path_4 = "./img4.jpg"
img_4 = urllib.request.urlretrieve("https://res.cloudinary.com/qna/image/upload/v1646684872/1032460886_4a598ed535_cda8dc.jpg", path_4) 
path_5 = "./img5.jpg"
img_5 = urllib.request.urlretrieve("https://hips.hearstapps.com/hmg-prod.s3.amazonaws.com/images/beginner-mountain-biking-1615604804.jpg", path_5) 
path_6 = "./img6.jpg"
img_6 = urllib.request.urlretrieve("data:image/jpeg;base64,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", path_6)
# Approch 3 for image 1 - With 30 times reinitialization of data generator and 1 epoch and 1 steps per epoch - Stochastic Gradient Descent
myimg = io.imread(path_4) 
print("Input Image")
io.imshow(myimg) 
plt.show()
i=29
# load the model
model = load_model(f'models_30_1_6000/models_30_1_6000/model_{i}_30_1_6000.h5')
# load and prepare the photograph
photo = extract_features('./img4.jpg')
# generate description
description = generate_desc(model, tokenizer, photo, max_length)
description = description.replace("startseq"," ")
description = description.replace("endseq"," ")
print("At Epoch ",i,"- ",description)
