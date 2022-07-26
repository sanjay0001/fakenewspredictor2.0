# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 00:05:55 2022

@author: Sanjay A
"""

import pickle
import pandas as pd
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
#nltk.download('stopwords')
port_stem=PorterStemmer()
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',str(content))
    stemmed_content = stemmed_content.lower() 
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')] 
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content


load_model=pickle.load(open('modelnew.pkl','rb'))
load_tfidf=pickle.load(open('tfidf.pkl','rb'))
demo="Trump Is Mentally Unstable - The New York Times"
demo=stemming(demo)
demo=load_tfidf.transform([demo])
prediction=load_model.predict(demo)
if(prediction==1):
    print("Fake news")
else:
    print("Real news")

print("Done")

