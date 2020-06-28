import pandas as pd
import matplotlib.pyplot as plt
import pickle
#cleaning text data
import nltk
import numpy as np
import re #regular expressions(rmeoving fullstops etc)
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


dataset= pd.read_csv('train.csv')#,encoding = "ISO-8859-1",nrows=9000)



def preprocess_tweet(tweet):
    #Preprocess the text in a single tweet
    #arguments: tweet = a single tweet in form of string 
    #convert the tweet to lower case
    tweet.lower()
    #convert all urls to sting "URL"
    tweet= re.sub('[^a-zA-Z]',' ',tweet)
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    #convert all @username to "AT_USER"
    tweet = re.sub('@[^\s]+','AT_USER', tweet)
    #correct all multiple white spaces to a single white space
    tweet = re.sub('[\s]+', ' ', tweet)
    #convert "#topic" to just "topic"
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    return tweet
dataset['text'] = dataset['text'].apply(preprocess_tweet)

c=[]
for i in range(0,9000):
    review1= re.sub('[^a-zA-Z]',' ',dataset['text'].iloc[i])
    #cover it as lower case
    review1=review1.split()#splits words and forms as list
    ps= PorterStemmer()
    review1=[ps.stem(word) for word in review1 if not word in set(stopwords.words('english'))]
    review1=' '.join(review1)
    c.append(review1)
#print(c)
    #creating a bag of words model
from  sklearn.feature_extraction.text import CountVectorizer
cv= CountVectorizer(max_features=15000)
x= cv.fit_transform(c).toarray()
y= dataset.iloc[:,0:1].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
pickle.dump(cv,open("cf19.pkl","wb"))
from keras.models import Sequential
from keras.layers import Dense
model=Sequential()


model.add(Dense(input_dim=x.shape[1],init="random_uniform",activation="sigmoid",output_dim=50))
model.add(Dense(init="random_uniform",activation="sigmoid",output_dim=10))
model.add(Dense(output_dim=1,init='random_uniform',activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#optimizing model
model.fit(x_train,y_train,epochs=50,batch_size=130)
y_pred=model.predict(x_test)
#print(x_test)
y_pred=(y_pred>0.5)
model.save('final3.h5')
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)
#loaded_vec =CountVectorizer(decode_error="replace",vocabulary=pickle.load(open("cf19.pkl", "rb")))
#da=""
#da=da.split("delimiter")
#anresult= model.predict(loaded_vec.transform(da))
