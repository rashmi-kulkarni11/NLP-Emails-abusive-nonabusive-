# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 20:41:05 2021

@author: deshp
"""
import pandas as pd 
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.naive_bayes import MultinomialNB
from flask import Flask, render_template, request
import pickle
import joblib

#filename='nlp_model.pkl'
clf= joblib.load(open('nlp_model.pkl','rb'))
cv= joblib.load(open('vector.pkl','rb'))

app=Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('home.htm')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method=='POST':
        message= request.form['message']
        data= [message]
        vect= cv.transform(data).toarray()
        my_prediction= clf.predict(vect)
    return render_template('result.htm', prediction =my_prediction)

if __name__=='__main__':
    app.run(debug=True)