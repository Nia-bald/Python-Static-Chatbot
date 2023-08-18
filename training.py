import random  # for random respons
import json  # convert json to dict
import pickle  # serialization
import numpy as np  # for numpy stuff
import nltk
from nltk.stem import WordNetLemmatizer
# dk what below libraries do exactly but we'll see
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
nltk.download('punkt')
lemmatizer = WordNetLemmatizer()
intents = json.loads(open("intents.json").read())
words = []  # contains all possible lemma that can be derived from pattern part in training data
classes = [] # list of all the sentiment
documents = [] # each element is a list of words tagged with its intent
ignore_letters = ['?', '!', '.', ',']
# data cleaning and collection

for intent in intents['intents']:
    for pattern in intent['pattern']:
        word_list = nltk.word_tokenize(pattern)  # splits sentences into individual words
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
# lemmatization of words
# why we do this decrease the scope of data?? idk we'll find out
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words)) # why sort?
classes = sorted(set(classes))

# create a new file to store the set of lemmas and sentiments or classes
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# need to convert the data into numerical data
training = []
# each element contains two data
    # first one is output_row -> represents sentiment of the element index which has value one is the index in classes whose sentiment this element represents
    # second one is bag -> represents words involved in this sentiment index with one corresponds to index in words
output_empty = [0]*len(classes)
for document in documents:
    bag = []  # gives data about lemma in list words which comes under the sentiment of current document
    # so essentially index for which the value is 1 is the index which comes under the sentiment of document
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    output_row = list(output_empty)
    # classes.index(document[1]) essentially finds the index of the current sentiment in classes
    # So I guess output row gives the data about sentiments which we have already dealt with
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

train_x = list(training[:, 0])  # each elements is a list of words corresponding to the sentiment with the same index on y axis
train_y = list(training[:, 1])  # each element is a list which represents the sentiment of the current index


# neural network partssss  most of the stuff here is a black box
# what's a sequential model

model = Sequential()
# input layer
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu')) # what is dense
model.add(Dropout(0.5)) # whats dropout
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
# output layer
model.add(Dense(len(train_y[0]), activation='softmax')) # softmax activation what?

# stochastic gradient descent
sgd = SGD(lr=0.01, momentum=0.9, nesterov=True) # what was learning rate, momentum what? nesterov what
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy']) # loss what? metrics what?
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1) # verbose what?
model.save('chatbot model.model', hist)

