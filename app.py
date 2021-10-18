import numpy as np
import pandas as pd
import pickle
from nltk.tokenize import WhitespaceTokenizer
import joblib
from sklearn.preprocessing import StandardScaler
from flask import Flask, jsonify, request, render_template
import re

tk = WhitespaceTokenizer()
scaler = StandardScaler(with_mean = False)
model = joblib.load('feature.pkl')
tfid = joblib.load('tfid.pkl')
scaler = joblib.load('scaler.pkl')

#func
def wordToken(text):
  tokens = tk.tokenize(text)
  return tokens

stopword_list= ["\n", "\t", "<", ">", "+", "-", "*", "%", "=", "==", "."]
def removeStopWords(text):
  new_tokens = [word for word in text if word.lower() not in stopword_list]
  return new_tokens

#func
def removeNum(text):
  NumRemoved = []
  for x in text:
    x = re.sub('\d+', 'N', x)
    x = re.sub('N,N', 'N', x)
    x = re.sub('N.N', 'N', x)
    NumRemoved.append(x)
  return NumRemoved

# app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

# routes
@app.route('/predict',methods=['POST'])
def predict():
	# global tfid, scaler
	data = request.form.get('bo data')
	txt = wordToken(data)
	txt = removeStopWords(txt)
	txt = removeNum(txt)
	txt = [" ".join(txt)]
	txt = tfid.transform(txt)
	txt = scaler.transform(txt)
	result = model.predict(txt)
	output = int(result[0])
	return render_template('index.html', prediction_text = 'Output is: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)