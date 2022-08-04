import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
#nltk.download('punkt')
import tensorflow as tf
import numpy as np
import pandas as pd



#pre-processing

from string import punctuation

train_data=pd.read_excel('train_dataset.xlsx')
test_data=pd.read_excel('test_dataset.xlsx')

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

print(vectorizer.vocabulary_)



from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer()
X_train = tfidf.fit_transform(X_train)
X_train = X_train.toarray()



#model



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
def create_deep_model(factor, rate):
    model = Sequential()      
    model.add(Dense(units=4096,kernel_regularizer=l2(factor),activation='relu')), Dropout(rate),
    model.add(Dense(units=512,kernel_regularizer=l2(factor),activation='relu')), Dropout(rate),
    model.add(Dense(units=512,kernel_regularizer=l2(factor), activation='relu')), Dropout(rate),
    #Output layer
    model.add(Dense(units=3, activation='softmax'))
    return model


model= create_deep_model(factor=0.0001, rate=0.1)


learningrate=0.001
early_stop = EarlyStopping(monitor='val_loss', mode='min',verbose=1, patience=5)
opt=tf.keras.optimizers.Adam(learning_rate=learningrate)
model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])

batchsize=48
epochs=10
X_train_enc, X_val, y_train_enc, y_val = train_test_split(X_train, y_train, test_size=0.25, shuffle= False)

history=model.fit(x=X_train_enc, y=y_train_enc, batch_size=batchsize, epochs=epochs, validation_data=(X_val, y_val), verbose=1, callbacks=early_stop)
model.summary()

y_train= np.argmax(y_train, axis=1)
y_test= np.argmax(y_val, axis=1)

y_test=y_val
y_test=np.argmax(y_test, axis=1)
y_pred = np.argmax(model.predict(X_val), axis=-1)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, target_names=['indifferent', 'for', 'against']))


model.save('my_model')

