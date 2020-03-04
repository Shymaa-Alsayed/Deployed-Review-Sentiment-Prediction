# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 21:59:04 2020

@author: LEGION
"""
# import necessary dependencies
from flask import Flask,render_template,request,url_for
import re 
import pickle
import numpy as np


# load the model, vectorizer, stemmer
model=pickle.load(open('nlp_model.pkl','rb'))
vectorizer=pickle.load(open('tfidf.pkl','rb'))
stemmer=pickle.load(open('porter_stem.pkl','rb'))

#create flask app
app=Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    def process(review):
        review=re.sub('[^a-zA-Z]',' ',review)
        review=review.lower()
        review=review.split()
        stemmed_review=[stemmer.stem(word) for word in review]
        join=lambda x:' '.join(x)
        joined_review=join(stemmed_review)
        sparse_review=vectorizer.transform(np.array([joined_review]))
        numeric_review=sparse_review.toarray()
        return numeric_review
    
    if request.method=='POST':
    	message = request.form['message']
    	processed_data = process(message)
    	predicted = model.predict(processed_data)
    return render_template('result.html',prediction = predicted)    
    
    
if __name__ == '__main__':
	app.run(debug=True)
    
    
    
    
    
    
    
    
    
   