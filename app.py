from flask import Flask
from flask import render_template
from flask import request
from flask import redirect
from flask import url_for
from flask import flash
from flask import jsonify
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk

app=Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html",prediction="")

@app.route('/predict',methods=['POST'])
def predict():
    load_model=pickle.load(open('modelnew.pkl','rb'))
    load_tfidf=pickle.load(open('tfidf.pkl','rb'))
    port_stem=PorterStemmer()
    def stemming(content):
        stemmed_content = re.sub('[^a-zA-Z]',' ',str(content))
        stemmed_content = stemmed_content.lower() 
        stemmed_content = stemmed_content.split()
        stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')] 
        stemmed_content = ' '.join(stemmed_content)
        return stemmed_content
    if request.method=='POST':
        message=request.form['message']
        message=stemming(message)
        message=load_tfidf.transform([message])
        prediction=load_model.predict(message)
        if(prediction==1):
            return render_template('index.html',prediction="Fake news")
        else:
            return render_template('index.html',prediction="Real news")
if __name__=='__main__':
    app.run(debug=False)
