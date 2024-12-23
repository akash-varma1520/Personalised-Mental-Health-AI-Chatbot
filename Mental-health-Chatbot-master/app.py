import torch
import torch.nn as nn
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from flask import Flask, render_template, request
import json
import numpy as np
import random
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Load resources
lemmatizer = WordNetLemmatizer()
nltk.download('punkt')

with open('intents.json') as file:
    intents = json.load(file)

words = pickle.load(open('texts.pkl', 'rb'))
classes = pickle.load(open('labels.pkl', 'rb'))

# Load PyTorch model
class ChatbotModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ChatbotModel, self).__init__()
        # Ensure the correct architecture is used
        self.fc1 = nn.Linear(input_size, hidden_size)  # Input -> Hidden layer
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # Hidden -> Hidden layer
        self.fc3 = nn.Linear(hidden_size, output_size)  # Hidden -> Output layer
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

input_size = len(words)  # This is the length of the words list
hidden_size = 128  # You can adjust this as necessary
output_size = len(classes)  # Number of possible classes (tags)

# Initialize the model
model = ChatbotModel(input_size, hidden_size, output_size)

# Load the model weights (make sure weights_only=True to avoid security issues)
model.load_state_dict(torch.load('chatbot_model.pth', weights_only=True))
model.eval()

# Preprocessing functions
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence, model):
    bow_vector = torch.FloatTensor(bow(sentence, words)).unsqueeze(0)
    output = model(bow_vector)
    _, predicted = torch.max(output, dim=1)
    return classes[predicted.item()]

def getResponse(tag):
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "I'm sorry, I didn't understand."

# Flask app
app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    tag = predict_class(userText, model)
    return getResponse(tag)

if __name__ == "__main__":
    app.run()
