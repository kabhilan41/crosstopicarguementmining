# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 08:58:55 2021

@author: RIOT
"""


import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
#nltk.download('punkt')
import pandas as pd
import tensorflow as tf
import numpy as np


x='Vaccines are bad'
data={'raw_data':[x]}
#train_data=pd.DataFrame(data)
#pre-processing


#y=np.argmax(load_model.predict(x))
import os
from flask import Flask, Blueprint, render_template, flash, request, redirect, url_for
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'UPLOAD_FOLDER'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif','xlsx'}

hello = Blueprint('hello', __name__)

@hello.route('/')
def hello_world():
    return render_template("index.html")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@hello.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print(filename)
            filename='UPLOAD_FOLDER\\'+filename
            train_data=pd.read_excel(filename)
            from string import punctuation
            
            
            def remove_stopwords(text):
                stpword = stopwords.words('english')
                no_punctuation = [char for char in text if char not in punctuation]
                no_punctuation = ''.join(no_punctuation)
                return ' '.join([word for word in no_punctuation.split() if word.lower() not in stpword])
            train_data['stop_rem'] = train_data['raw_data'].apply(remove_stopwords)
            
            
            train_one_hot = pd.get_dummies(train_data['sentiment'])
            train_data = pd.concat([train_data['stop_rem'],train_one_hot],axis=1)
            y_train = train_data.drop('stop_rem',axis=1).values

            #feature extraction
            
            
            from sklearn.feature_extraction.text import CountVectorizer 
            #Define input
            sentences_train = train_data['stop_rem'].values
            
            print(sentences_train)
            #Convert sentences into vectors
            vectorizer = CountVectorizer()
            X_train = vectorizer.fit_transform(sentences_train)
            
           
            
            
            
            from sklearn.feature_extraction.text import TfidfTransformer
            tfidf = TfidfTransformer()
            X_train = tfidf.fit_transform(X_train)
            X_train = X_train.toarray()
            
            
            
            load_model= tf.keras.models.load_model('D:\Codes\ML stuff\pandas\my_model')
            load_model.summary()
            y_train= np.argmax(y_train, axis=1)
            y_pred = np.argmax(load_model.predict(X_train), axis=-1)
            from sklearn.metrics import classification_report
            vari=classification_report(y_train,y_pred, target_names=['indifferent', 'for', 'against'],output_dict=True)
            
            print(vari)
            print(vari['indifferent']['precision'])
            return render_template("index.html", ival=vari['indifferent']['recall'],pval=vari['for']['recall'],nval=vari['against']['recall'])
                    
        return ""

if __name__ == '__main__':
    app = Flask(__name__)
    app.register_blueprint(hello, url_prefix='/')
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

    app.run()