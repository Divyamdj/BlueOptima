import numpy as np
import pandas as pd
import pickle
from nltk.tokenize import WhitespaceTokenizer
import joblib
from sklearn.preprocessing import StandardScaler
from flask import Flask, jsonify, request, render_template
import re
import time
import psutil
import os

tk = WhitespaceTokenizer()
scaler = StandardScaler(with_mean = False)
model = joblib.load('model.pkl')
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
	
	start_time = time.time()

	data = request.form.get('Enter Code')
	txt = wordToken(data)
	txt = removeStopWords(txt)
	txt = removeNum(txt)
	txt = [" ".join(txt)]
	txt = tfid.transform(txt)
	txt = scaler.transform(txt)
	result = model.predict(txt)
	output = int(result[0])

	if output == 0:
		result = "csproj"
	elif output == 1:
		result = "jenkinsfile"
	elif output == 2:
		result = "kt"
	elif output == 3:
		result = "mak"
	elif output == 4:
		result = "ml"
	elif output == 5:
		result = "rexx"

	end_time = time.time()
	total_time = 1/(end_time - start_time)

	return render_template('index.html', prediction_text = 'Language is: {}  \n    Number of Execution(per sec): {}\n   RAM percentage used: {} '.format(result, total_time, psutil.cpu_percent(2)))

if __name__ == "__main__":
    app.run(debug=True)
