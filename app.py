import numpy as np
import pandas as pd
import pickle
from nltk.tokenize import WhitespaceTokenizer
import joblib
from sklearn.preprocessing import StandardScaler
# from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, jsonify, request, render_template
import re

tk = WhitespaceTokenizer()
scaler = StandardScaler(with_mean = False)
# vectorizer_tfidf = TfidfVectorizer()
model = joblib.load('feature.pkl')
tfid = joblib.load('tfid.pkl')
# print(len(list(tfid.get_feature_names())))
scaler = joblib.load('scaler.pkl')


#func
def wordToken(text):
  tokens = tk.tokenize(text)
  return tokens

stopword_list= ["\n", "\t", "<", ">", "+", "-", "*", "%", "=", "==", "."]
#func
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
	# txt_tfidf = vectorizer_tfidf.transform(txt).toarray()
	# txt_tfidf = scaler.transform(txt_tfidf)
	txt = tfid.transform(txt)
	print("1")
	txt = scaler.transform(txt)
	print("2")
	result = model.predict(txt)
	print("3")
	output = int(result[0])
	return render_template('index.html', prediction_text = 'Output is: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)