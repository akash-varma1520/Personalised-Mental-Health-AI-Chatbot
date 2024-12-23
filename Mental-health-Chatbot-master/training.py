import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import pickle

nltk.download('punkt')

# Load intents data
with open('intents.json') as file:
    intents = json.load(file)

lemmatizer = WordNetLemmatizer()
ignore_words = ['?', '!']
words = []
classes = []
documents = []

# Preprocessing
for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = sorted(set(lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words))
classes = sorted(set(classes))

training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in doc[0]]
    for w in words:
        bag.append(1 if w in pattern_words else 0)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)

train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

# Define the model
class ChatbotModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ChatbotModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # input to hidden layer
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # hidden to hidden layer (if you want)
        self.fc3 = nn.Linear(hidden_size, output_size)  # hidden to output layer
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))  # Apply relu after fc1
        x = self.relu(self.fc2(x))  # Apply relu after fc2
        x = self.fc3(x)  # Output layer
        return x


input_size = len(train_x[0])
hidden_size = 128
output_size = len(classes)

model = ChatbotModel(input_size, hidden_size, output_size)

# Training the model
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
epochs = 200

for epoch in range(epochs):
    for i, (x, y) in enumerate(zip(train_x, train_y)):
        x_tensor = torch.FloatTensor(x).unsqueeze(0)
        y_tensor = torch.LongTensor([np.argmax(y)])
        optimizer.zero_grad()
        output = model(x_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

# Save the trained model and data
torch.save(model.state_dict(), 'chatbot_model.pth')
pickle.dump(words, open('texts.pkl', 'wb'))
pickle.dump(classes, open('labels.pkl', 'wb'))

print("Training complete. Model saved as chatbot_model.pth.")
