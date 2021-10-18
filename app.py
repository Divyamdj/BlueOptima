# from sklearn.externals import joblib
import numpy as np
import pandas as pd
from numpy import loadtxt
import joblib
from flask import Flask, jsonify, request, render_template
import pickle

model=joblib.load('Mango_UV_ShelfLife.pkl')

scaler=joblib.load('Mango_UV_ShelfLife_scale.pkl')

pca=joblib.load('Mango_UV_ShelfLife_pca.pkl')

# app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

# routes
@app.route('/predict',methods=['POST'])



def predict():
	data = request.form.getlist('UV data')

	X=data[0].split()
	X=np.array(X).astype(float)

	X_1=X[0:288].reshape(1,-1)
	X_scaled=scaler.transform(X_1)
	X_pca=pca.transform(X_scaled)
	result=model.predict(X_pca)

	# # output = {'results': int(result[0])}
	output=int(result[0])

	# return jsonify(results=output)
	return render_template('index.html', prediction_text='Output is: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)