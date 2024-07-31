from flask import Flask, render_template, request
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from scipy.sparse import csr_matrix

import pandas as pd
import pickle
import string
from nltk.corpus import stopwords
import nltk
import joblib
from nltk.stem.porter import PorterStemmer
app = Flask(__name__)

phish_model = open('final_model2.pkl','rb')
phish_model_ls = joblib.load(phish_model)
vectorizer = CountVectorizer(tokenizer=RegexpTokenizer(r'[A-Za-z]+').tokenize, stop_words='english')


@app.route('/')
def home():
    return render_template('newlanding_page.html')

@app.route('/depression',methods=['GET','POST'])
def depression():
    return render_template('depression_test.html')

@app.route('/education',methods=['GET','POST'])
def education():
    return render_template('education.html')
	
@app.route('/quiz',methods=['GET','POST'])
def quiz():
    return render_template('quiz.html')

@app.route('/subscription',methods=['GET','POST'])
def subscription():
    return render_template('sms_subs.html')
	

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model1 = pickle.load(open('final_model2.pkl','rb'))

@app.route('/predict_sms', methods=['POST'])

def predict_depression():
    input_sms = request.form['feedback']

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model1.predict(vector_input)[0]
    # 4. Display
    if result == 0:

        return render_template('depression_test.html', result="normal", feedback = input_sms)
    
    else:
        return render_template('depression_test.html', result="depression", feedback = input_sms)


if __name__ == "__main__":
    app.run(debug=True)
