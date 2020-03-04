# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 19:23:01 2020

@author: LEGION
"""

# Importing libraries
import pandas as pd
import pandas as pd
import re 
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle



# Loading data
reviews=pd.read_csv('amazon_reviews_sample.csv')



# Data Cleaning 
## removing punctuation
reviews['review']=[re.sub('[^a-zA-Z]',' ',review) for review in reviews.review]

## lowercasing
reviews['review']=reviews['review'].str.lower()

## splitting the review into words
reviews['review']=reviews['review'].str.split()

## stemming 
stemmer=PorterStemmer()
def stem_review(review):
    stemmed_review=[stemmer.stem(word) for word in review]
    return stemmed_review
pickle.dump(stemmer,open('porter_stem.pkl','wb'))


reviews['review']=reviews['review'].apply(stem_review)

## rejoining each review after being prepared and cleaned
reviews['review']=reviews['review'].apply(lambda x:' '.join(x))



# Convering text data into numeric by creating a BOW (bag of words) excluding stopwords
vectorizer=TfidfVectorizer(max_features=2500,ngram_range=(1,3))
vectorizer.fit(reviews.review)
sparse_matrix_x=vectorizer.transform(reviews.review)
reviews_numeric_df=pd.DataFrame(sparse_matrix_x.toarray(),columns=vectorizer.get_feature_names())
reviews_numeric_df['sentiment']=reviews['score']


pickle.dump(vectorizer,open('tfidf.pkl','wb'))


# Creating dataset of feature matrix and labels
X=reviews_numeric_df.drop(['sentiment'],axis=1)
y=reviews_numeric_df['sentiment']



# Splitting the dataset into the Training set and Test set
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=123,stratify=y)



# Fitting Logistic Regression model to the training set and prediction 
log_reg=LogisticRegression(solver='newton-cg',C=1)
log_reg.fit(X_train,y_train)

pickle.dump(log_reg,open('nlp_model.pkl','wb'))
