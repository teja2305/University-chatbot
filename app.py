import nltk
import pickle
import numpy as np
import json
import random
import speech_recognition as sr
import pyttsx3
import threading

from nltk.stem import WordNetLemmatizer
from keras.models import load_model
from flask import Flask, render_template, request
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('popular')

lemmatizer = WordNetLemmatizer()

model = load_model('model.h5')

intents = json.loads(open('data.json', encoding='utf-8').read())
words = pickle.load(open('texts.pkl','rb'))
classes = pickle.load(open('labels.pkl','rb'))

def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    
    # Pass the sentence through a sentiment analysis model
    sentiment_score = SentimentIntensityAnalyzer().polarity_scores(sentence)['compound']
    
    # Adjust the probability scores based on the sentiment analysis result
    for i, r in results:
        if sentiment_score > 0:
            res[i] = r * 1.1
        elif sentiment_score < 0:
            res[i] = r * 0.9
    
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    text_response = res
    voice_response = res
    return text_response, voice_response


def speak(text):
    engine = pyttsx3.init()
    # set voice to a female voice
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id) # 1 is the index of a female voice
    engine.say(text)
    engine.runAndWait()


app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_response():
    userText = request.args.get('msg')
    text_response, voice_response = chatbot_response(userText)
    t1 = threading.Thread(target=speak, args=(voice_response,))
    t1.start()
    return text_response

@app.route("/speech")
def speech_response():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something!")
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            print("You said: ", text)
            text_response, voice_response = chatbot_response(text)
            t1 = threading.Thread(target=speak, args=(voice_response,))
            t1.start()
            return text_response
        except:
            print("Sorry, I could not understand what you said.")
            return "Sorry, I could not understand what you said."

if __name__ == "__main__":
    app.run(debug=True)

