# Chatbot Project
## Getting Started


## 1.Import and load the data file
The data file is in JSON format so we used the json package to parse the JSON file into Python:

```python
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random

words = []
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('intents.json').read()
intents = json.loads(data_file)
```

This snippet demonstrates the initial setup and data loading for the chatbot. It imports the necessary libraries, including nltk for natural language processing, and loads the intents data from the `intents.json` file.

## 2. Preprocess data

In text data analysis, it is essential to preprocess the data before building machine learning or deep learning models. Preprocessing involves applying different operations to the data based on specific requirements.

One of the fundamental steps in text preprocessing is tokenization, which involves breaking the entire text into smaller units, such as words.

In this project, we iterate through the patterns and tokenize each sentence using the nltk.word_tokenize() function. We then add each word to the words list. Additionally, we create a separate list of classes to represent the tags associated with the patterns.

```python
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word
        w = nltk.word_tokenize(pattern)
        words += w  # Extend the words list
        # Add documents to the corpus
        documents.append((w, intent['tag']))

        # Add to the classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
```
Next, we will perform lemmatization on each word and eliminate duplicate words from the list. Lemmatization involves converting words into their base or lemma form. After that, we will create a pickle file to store the necessary Python objects, which will be used for prediction purposes.

```python
# Lemmatize, convert to lowercase, and remove duplicates
words = sorted(list(set([lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words])))

# Sort classes
classes = sorted(list(set(classes)))

# Combine patterns and intents to form documents
documents = list(zip(words, [intent['tag'] for intent in intents['intents'] for pattern in intent['patterns']]))

print(len(documents), "documents")
print(len(classes), "classes", classes)
print(len(words), "unique lemmatized words", words)

# Save words and classes using pickle
with open('words.pkl', 'wb') as f:
    pickle.dump(words, f)
with open('classes.pkl', 'wb') as f:
    pickle.dump(classes, f)

```

## 3. Generate Training and Testing Data

In this step, we create the training and testing data for our chatbot. The training data consists of input patterns and their corresponding output classes. However, since computers cannot directly understand text, we need to convert the text into numerical representations.

```python
# Create the training data
training = []

# Create an empty array for the output
output_empty = [0] * len(classes)

# Iterate over the documents
for doc in documents:
    # Initialize the bag of words
    bag = []

    # Tokenize and lemmatize the pattern words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in doc[0]]

    # Create the bag of words array
    for word in words:
        bag.append(1) if word in pattern_words else bag.append(0)

    # Create the output row
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# Shuffle and convert the training data to numpy array
random.shuffle(training)
training = np.array(training)

# Separate the features and labels
train_x = list(training[:, 0])
train_y = list(training[:, 1])

print("Training data created")
```

## 4. Construct the Model

With the training data prepared, we can now build a deep neural network model consisting of three layers. In this project, we utilize the Keras sequential API for building the model. After training the model for 200 epochs, we have achieved a remarkable accuracy of 100%. We will save this trained model as 'chatbot_model.h5'.

```python
# Create the model - 3 layers: 128 neurons in the first layer, 64 neurons in the second layer, and the output layer with a number of neurons equal to the number of intents to predict output intent using softmax
model = Sequential([
    Dense(128, input_shape=(len(train_x[0]),), activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(train_y[0]), activation='softmax')
])

# Compile the model using stochastic gradient descent with Nesterov accelerated gradient
model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])

# Fit the model and save it
history = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5')

print("Model created")
```

## 5. Predict the Response (Graphical User Interface)

To predict sentences and generate responses from the chatbot, let's create a new file called 'chatapp.py'.

In this file, we will load the trained model and implement a graphical user interface that allows the bot to predict responses. Since the model only provides us with the predicted class, we will create functions to identify the class and randomly select a response from the list of predefined responses.

Once again, we import the required packages and load the 'words.pkl' and 'classes.pkl' pickle files that were created during the model training phase.

```python
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import load_model
import json
import random

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load the trained model
model = load_model('chatbot_model.h5')

# Load intents data from JSON file
with open('intents.json', 'r') as file:
    intents = json.load(file)

# Load words and classes from pickle files
with open('words.pkl', 'rb') as file:
    words = pickle.load(file)

with open('classes.pkl', 'rb') as file:
    classes = pickle.load(file)
```
To make predictions on the input text, we need to preprocess it in the same manner as during the training phase. To accomplish this, we will develop functions that handle text preprocessing and subsequently predict the class based on the processed input.

```python
def clean_up_sentence(sentence):
    # Tokenize the pattern - split words into an array
    sentence_words = nltk.word_tokenize(sentence)
    # Lemmatize each word - create a short form for the word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    # Tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # Bag of words - matrix of N words, vocabulary matrix
    bag = [1 if word in sentence_words else 0 for word in words]
    if show_details:
        for word in words:
            if word in sentence_words:
                print("found in bag: %s" % word)
    return np.array(bag)

def predict_class(sentence, model):
    # Filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # Sort by the strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]
    return return_list
```
After predicting the class, we will get a random response from the list of intents.

```python
def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for intent in list_of_intents:
        if intent['tag'] == tag:
            result = random.choice(intent['responses'])
            break
    return result

def chatbot_response(text):
    intents_list = predict_class(text, model)
    response = get_response(intents_list, intents)
    return response
```

Next, we will create a graphical user interface (GUI) using the Tkinter library, which provides a wide range of functionalities for building GUI applications. The GUI will allow the user to enter messages, and we will utilize the helper functions we developed earlier to obtain a response from the chatbot. The response will be displayed on the GUI. Below is the complete source code for the GUI application.

```python
import tkinter as tk
from tkinter import *

def send():
    msg = entry_box.get("1.0", 'end-1c').strip()
    entry_box.delete("0.0", END)

    if msg != '':
        chat_log.config(state=NORMAL)
        chat_log.insert(END, "You: " + msg + '\n\n')
        chat_log.config(foreground="#442265", font=("Verdana", 12))

        res = chatbot_response(msg)
        chat_log.insert(END, "Bot: " + res + '\n\n')

        chat_log.config(state=DISABLED)
        chat_log.yview(END)

base = tk.Tk()
base.title("Hello")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)

# Create Chat window
chat_log = Text(base, bd=0, bg="white", height="8", width="50", font="Arial")
chat_log.config(state=DISABLED)

# Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=chat_log.yview, cursor="heart")
chat_log['yscrollcommand'] = scrollbar.set

# Create Button to send message
send_button = Button(base, font=("Verdana", 12, 'bold'), text="Send", width="12", height=5,
                     bd=0, bg="#32de97", activebackground="#3c9d9b", fg='#ffffff',
                     command=send)

# Create the box to enter message
entry_box = Text(base, bd=0, bg="white", width="29", height="5", font="Arial")

scrollbar.place(x=376, y=6, height=386)
chat_log.place(x=6, y=6, height=386, width=370)
entry_box.place(x=128, y=401, height=90, width=265)
send_button.place(x=6, y=401, height=90)

base.mainloop()
```

## Run the Chatbot

To run the chatbot, you will need to execute two main files: `train_chatbot.py` and `chatapp.py`.

First, train the model by running the following command in the terminal:

python train_chatbot.py

If the training process completes without any errors, it means that the model has been successfully created. Next, to run the chatbot application, execute the second file:

python chatgui.py

After a few seconds, a GUI window will open, allowing you to easily interact with the chatbot.



## Contributions

Contributions to this project are welcome! If you find any bugs, have suggestions for improvements, or would like to add new features, please feel free to open an issue or submit a pull request. The project's author will review and respond to them as soon as possible.

## License

This project is licensed under the [MIT License](LICENSE).

##

 Acknowledgments

The author would like to acknowledge the valuable support and guidance received from the open-source community, various online resources, and the AWS and Azure platforms for their contribution to this project.

Thank you for visiting this repository!
