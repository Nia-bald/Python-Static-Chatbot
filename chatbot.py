import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow.keras.models import load_model


lemmatizer = WordNetLemmatizer()
intents = json.loads(open("intents.json").read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot model.model')

# func for cleaning sentence
# func for getting bag of words
# func for predicting class based on sentence

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1 # just says ith word in words is in sentence
    return np.array(bag) # just says index which has a value in bag is the index whose value is in sentence in words
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD] # ith represents the index which can then be mapped with classes to find the sentiment, and r represents the likely of that sentiment being true
    results.sort(key=lambda x:x[1], reverse=True)
    returnList = []
    for r in results:
        returnList.append({'intent':classes[r[0]], 'probability': str(r[1]) })
    return returnList
def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['response'])
            break
    return result

print('BOT RUNNING')
while True:
    message = input("")
    ints = predict_class(message)
    res = get_response(ints, intents)
    print(res)