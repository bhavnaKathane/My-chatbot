import json
import numpy as np
import nltk
import pickle
from nltk.stem import LancasterStemmer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download('punkt')
# nltk.download('punkt_tab')

stemmer = LancasterStemmer()

with open('data/intents.json') as f:
    data = json.load(f)

sentences = []
labels = []
classes = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        sentences.append(pattern.lower())
        labels.append(intent['tag'])
    if intent['tag'] not in classes:
        classes.append(intent['tag'])

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
X = pad_sequences(sequences, maxlen=20)

y = np.array([classes.index(label) for label in labels])

model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=16, input_length=20),
    LSTM(32),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(len(classes), activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, y, epochs=200, batch_size=4, verbose=1)

model.save('models/chatbot_model.h5')
pickle.dump(tokenizer, open('models/tokenizer.pkl', 'wb'))
pickle.dump(classes, open('models/classes.pkl', 'wb'))

print("Training complete!")