

from os import listdir
from pickle import dump
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from PIL import Image

def extract_features(directory):
    model = VGG16()
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    features = dict()
    for name in listdir(directory):
        filename = directory + '/' + name
        image = load_img(filename, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        feature = model.predict(image, verbose=0)
        image_id = name.split('.')[0]
        features[image_id] = feature
    return features


dir = 'C:/Users/zahraGhorbani/Desktop/dataset/Flicker8k_Dataset'
features = extract_features(dir)
print('Extracted Features: %d' % len(features))
dump(features, open('features.pkl', 'wb'))

def load_document(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

def load_descriptions(doc):
    map = dict()
    for line in doc.split('\n'):
        tokens = line.split()
        if len(line) < 2:
            continue
        image_id, image_desc = tokens[0], tokens[1:]
        image_id = image_id.split('.')[0]
        image_desc = ' '.join(image_desc)
        if image_id not in map:
            map[image_id] = list()
        map[image_id].append(image_desc)
    return map

def clean_descriptions(descriptions):
    table = str.maketrans('', '', string.punctuation)
    for key, desc_list in descriptions.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]
            desc = desc.split()
            desc = [word.lower() for word in desc]
            desc = [w.translate(table) for w in desc]
            desc = [word for word in desc if len(word)>1]
            desc = [word for word in desc if word.isalpha()]
            desc_list[i] =  ' '.join(desc)

def to_vocab(descriptions):
    all_desc = set()
    for key in descriptions.keys():
        [all_desc.update(d.split()) for d in descriptions[key]]
    return all_desc

def save_descriptions(descriptions, filename):
    lines = list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + ' ' + desc)
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()

import string
filename = '/content/drive/My Drive/Flickr8k.token.txt'
doc = load_document(filename)
descriptions = load_descriptions(doc)
print('Loaded: %d ' % len(descriptions))
clean_descriptions(descriptions)
vocabulary = to_vocab(descriptions)
print('Vocabulary Size: %d' % len(vocabulary))
save_descriptions(descriptions, 'descriptions.txt')

def load_set(filename):
    doc = load_document(filename)
    dataset = list()
    for line in doc.split('\n'):
        if len(line) < 1:
            continue
        ident = line.split('.')[0]
        dataset.append(ident)
    return set(dataset)

def load_clean_descriptions(filename, dataset):
    doc = load_document(filename)
    descriptions = dict()
    for line in doc.split('\n'):
        tokens = line.split()
        image_id, image_desc = tokens[0], tokens[1:]
        if image_id in dataset:
            if image_id not in descriptions:
                descriptions[image_id] = list()
            desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
            descriptions[image_id].append(desc)
    return descriptions

from pickle import load
def load_photo_features(filename, dataset):
    all_features = load(open(filename, 'rb'))
    features =  {k:all_features[k]  for k in dataset if k  in all_features.keys()}
    return features

from numpy import array
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint

def to_lines(descriptions):
    all_desc = list()
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc

def create_tokenizer(descriptions):
    lines = to_lines(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

def max_length(descriptions):
    lines = to_lines(descriptions)
    return max(len(d.split()) for d in lines)

def define_model(vocab_size, max_length):
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.summary()
    return model

def create_sequences(tokenizer, max_length, desc_list, photo, vocab_size):
    X1, X2, y = list(), list(), list()
    for desc in desc_list:
        seq = tokenizer.texts_to_sequences([desc])[0]
        for i in range(1, len(seq)):
            in_seq, out_seq = seq[:i], seq[i]
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            X1.append(photo)
            X2.append(in_seq)
            y.append(out_seq)
    return array(X1), array(X2), array(y)

def data_generator(descriptions, photos, tokenizer, max_length, vocab_size):
    while 1:
        for key, desc_list in descriptions.items():
            if key in photos.keys():
                photo = photos[key][0]
                in_img, in_seq, out_word = create_sequences(tokenizer, max_length, desc_list, photo, vocab_size)
                yield [[in_img, in_seq], out_word]

#load training dataset 

filename = 'C:/Users/zahraGhorbani/Desktop/dataset/Flickr_8k.trainImages.txt'
train = load_set(filename)
print('Dataset: %d' % len(train))
train_descriptions = load_clean_descriptions('C:/Users/zahraGhorbani/Desktop/dataset/descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))
train_features = load_photo_features('C:/Users/zahraGhorbani/Desktop/dataset/features.pkl', train)
print('Photos: train=%d' % len(train_features))
tokenizer = create_tokenizer(train_descriptions)
dump(tokenizer, open('tokenizer.pkl', 'wb'))
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
max_length = max_length(train_descriptions)
print('Description Length: %d' % max_length)

#load validation dataset 

filename_v = 'C:/Users/zahraGhorbani/Desktop/dataset/Flickr_8k.devImages.txt'
val = load_set(filename_v)
print('Dataset: %d' % len(val))
val_descriptions = load_clean_descriptions('C:/Users/zahraGhorbani/Desktop/dataset/descriptions.txt', val)
print('Descriptions: validation=%d' % len(val_descriptions))
val_features = load_photo_features('C:/Users/zahraGhorbani/Desktop/dataset/features.pkl', val)
print('Photos: validation=%d' % len(val_features))

# define the model
model = define_model(vocab_size, max_length)

# train the model, run epochs manually and save after each epoch
steps_t = len(train_descriptions)
steps_v = len(val_descriptions)
from keras.models import load_model
for i in range(20):
     generator_t = data_generator(train_descriptions, train_features, tokenizer, max_length, vocab_size)
     generator_v = data_generator(val_descriptions, val_features, tokenizer, max_length, vocab_size)
     history=model.fit_generator(generator_t, epochs=1, steps_per_epoch=steps_t, verbose=1,validation_data=generator_v,validation_steps=steps_v)
     model.save('C:/Users/zahraGhorbani/Desktop/dataset/modell_' + str(i) + '.h5')

from numpy import argmax
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu

def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None

def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        prediction = model.predict([photo,sequence], verbose=0)
        prediction = argmax(prediction)
        word = word_for_id(prediction, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text

def evaluate(model, descriptions, photos, tokenizer, max_length):
    actual, predicted = list(), list()
    for key, desc_list in descriptions.items():
        if key in photos.keys():
            prediction = generate_desc(model, tokenizer, photos[key], max_length)
            references = [d.split() for d in desc_list]
            actual.append(references)
            predicted.append(prediction.split())
    print('BLEU: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    #print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    #print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    #print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))

filename = 'C:/Users/zahraGhorbani/Desktop/dataset/Flickr_8k.testImages.txt'
test = load_set(filename)
print('Dataset: %d' % len(test))
test_descriptions = load_clean_descriptions('C:/Users/zahraGhorbani/Desktop/dataset/descriptions.txt', test)
print('Descriptions: test=%d' % len(test_descriptions))
test_features = load_photo_features('C:/Users/zahraGhorbani/Desktop/dataset/features.pkl', test)
print('Photos: test=%d' % len(test_features))

# load the model
filename = 'C:/Users/zahraGhorbani/Desktop/dataset/modell_0.h5'
model = load_model(filename)
evaluate(model, test_descriptions, test_features, tokenizer, max_length)

from pickle import load
from numpy import argmax
from keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.models import load_model

def extract_feature(filename):
    model = VGG16()
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    image = load_img(filename, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = model.predict(image, verbose=0)
    return feature

tokenizer = load(open('C:/Users/zahraGhorbani/Desktop/dataset/tokenizer.pkl', 'rb'))
max_length = 34
model = load_model('C:/Users/zahraGhorbani/Desktop/dataset/modell_0.h5')
photo = extract_feature('C:/Users/zahraGhorbani/Desktop/dataset/Capture3.PNG')
description = generate_desc(model, tokenizer, photo, max_length)
def convert(lst): 
    return ' '.join(lst).split() 

def reduce(description):
    lst =  [description] 
    ls=convert(lst)
    ls.pop(0)
    ls.pop(-1)
    return ls
description=reduce(description)  
description=' '.join(description)
print(description)
RealCaption ='startseq a man sits on a bench looking at a lake endseq'
import nltk
print('BLEU: %f' % nltk.translate.bleu_score.corpus_bleu([RealCaption],[description], weights=(1.0,0,0,0)))
print('BLEU-2: %f' % nltk.translate.bleu_score.corpus_bleu([RealCaption],[description], weights=(0.5,0.5,0,0)))
print('BLEU-3: %f' % nltk.translate.bleu_score.corpus_bleu([RealCaption],[description], weights=(0.3, 0.3, 0.3, 0)))
print('BLEU-4: %f' % nltk.translate.bleu_score.corpus_bleu([RealCaption],[description], weights=(0.25, 0.25, 0.25, 0.25)))
print( 'RealCaption:', RealCaption)
print( 'description:', description)
#translate
!pip install google_trans_new
from google_trans_new import google_translator  
translator = google_translator()  
translate_text = translator.translate(description ,lang_tgt='fa')  
print(translate_text)

#tts
!pip install gTTS
text_file = open("C:/Users/zahraGhorbani/Desktop/dataset/Output.txt", "w")
text_file.write(description)
text_file.close()
!gtts-cli  --file C:/Users/zahraGhorbani/Desktop/dataset/Output.txt --output C:/Users/zahraGhorbani/Desktop/dataset/Output3.mp3


photo1 = extract_feature('C:/Users/zahraGhorbani/Desktop/dataset/dog.jpg')
description = generate_desc(model, tokenizer, photo1, max_length)
print(description)
print('BLEU-1: %f' % nltk.translate.bleu_score.corpus_bleu([RealCaption],[description1], weights=(1.0,0,0,0)))
print('BLEU-2: %f' % nltk.translate.bleu_score.corpus_bleu([RealCaption],[description1], weights=(0.5,0.5,0,0)))
print('BLEU-3: %f' % nltk.translate.bleu_score.corpus_bleu([RealCaption],[description1], weights=(0.3, 0.3, 0.3, 0)))
print('BLEU-4: %f' % nltk.translate.bleu_score.corpus_bleu([RealCaption],[description1], weights=(0.25, 0.25, 0.25, 0.25)))

