import json
import numpy as np
import pickle
import nltk

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, request, jsonify, render_template

# Load trained model
model = load_model('models/chatbot_model.h5')

# Load tokenizer and classes
tokenizer = pickle.load(open('models/tokenizer.pkl', 'rb'))
classes = pickle.load(open('models/classes.pkl', 'rb'))

# Load intents file
with open('data/intents.json', encoding='utf-8') as f:
    data = json.load(f)

# Initialize Flask app
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False


# Prediction function
def predict(text):
    # Convert text to sequence
    seq = tokenizer.texts_to_sequences([text.lower()])

    # Pad sequence
    padded = pad_sequences(seq, maxlen=20)

    # Predict
    pred = model.predict(padded, verbose=0)

    # Get predicted class
    tag = classes[np.argmax(pred)]

    # Find matching response
    for intent in data['intents']:
        if intent['tag'] == tag:
            return np.random.choice(intent['responses'])

    return "Sorry, I didn't understand that."


# Chat API route
@app.route('/chat', methods=['POST'])
def chat():
    msg = request.json.get('message')

    if not msg:
        return jsonify({'response': 'Please send a message.'})

    response = predict(msg)

    return jsonify({'response': response})


# Home route
@app.route('/')
def home():
    return render_template('index.html')


# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)