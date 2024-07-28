from flask import Flask, render_template, request, jsonify
import pickle
import json
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# Load the model and other necessary files
model = tf.keras.models.load_model('C:\chatbot using python\chatbot\chatbot.h5')
lemmatizer = WordNetLemmatizer()

with open(r'C:\chatbot using python\chatbot\intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

with open('C:\chatbot using python\chatbot\words.pkl', 'rb') as file:
    words = pickle.load(file)

with open('C:\chatbot using python\chatbot\classes.pkl', 'rb') as file:
    classes = pickle.load(file)

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence, model):
    bow = bag_of_words(sentence, words)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = i['responses']
            break
    return result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    message = request.json['message']
    ints = predict_class(message, model)
    res = get_response(ints, intents)
    return jsonify({"response": res})

if __name__ == "__main__":
    app.run(debug=True)
