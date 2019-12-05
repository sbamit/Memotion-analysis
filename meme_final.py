#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 18:30:13 2019
@author: admin
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
#from sklearn.metrics import accuracy_score
import os    
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
#import nltk
#nltk.download('all')
datapath = os.getcwd()
datapath = datapath + "/data_6512_new.csv" 
if not os.path.exists(datapath):
    print ("Data cannot be located!")

data = pd.read_csv(datapath)

# np_data = np.array(data)
# print(arr.shape)
# 
# for i in range(3):
#    np_data = np.delete(np_data,0,axis=1)
# print(np_data.shape)

data = data.drop('Image_name', axis=1)
data = data.drop('Image_URL', axis=1)
data = data.drop('OCR_extracted_text', axis=1)

#Making all letters lowercase
data['Corrected_text'] = data['Corrected_text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
#Removing Punctuation, Symbols
data['Corrected_text'] = data['Corrected_text'].str.replace('[^\w\s]',' ')

#Removing Stop Words using NLTK
from nltk.corpus import stopwords
stop = stopwords.words('english')
data['Corrected_text'] = data['Corrected_text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

##Lemmatisation
#from textblob import Word
#data['Corrected_text'] = data['Corrected_text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

#Correcting Letter Repetitions
import re
def de_repeat(text):
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", text)

data['Corrected_text'] = data['Corrected_text'].apply(lambda x: " ".join(de_repeat(x) for x in x.split()))

# Code to find the top 10,000 rarest words appearing in the data
freq = pd.Series(' '.join(data['Corrected_text']).split()).value_counts()[-10000:]
# Removing all those rarely appearing words from the data
freq = list(freq.index)
data['Corrected_text'] = data['Corrected_text'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))

########################## Feature Extraction ##########################

#Encoding output labels 'sadness' as '1' & 'happiness' as '0'
from sklearn import preprocessing
lbl_enc = preprocessing.LabelEncoder()
#y = lbl_enc.fit_transform(data.Humour.["not funny"],data.Humour.["funny"],data.Humour.["very funny"])

humor_col = lbl_enc.fit_transform(data.Humour.values)
sentiments = np.array(humor_col)
sentiments = np.reshape(sentiments, (-1, 1))

sarcasm_col = lbl_enc.fit_transform(data.Sarcasm.values)
sarcasm_np = np.array(sarcasm_col)
sarcasm_np = np.reshape(sarcasm_np, (-1,1))
sentiments = np.append(sentiments, sarcasm_np, axis = 1)

offensive_col = lbl_enc.fit_transform(data.Offensive.values)
offensive_np = np.array(offensive_col)
offensive_np = np.reshape(offensive_np, (-1,1))
sentiments = np.append(sentiments, offensive_np, axis = 1)

motivational_col = lbl_enc.fit_transform(data.Motivational.values)
motivational_np = np.array(motivational_col)
motivational_np = np.reshape(motivational_np, (-1,1))
sentiments = np.append(sentiments, motivational_np, axis = 1)

overall_col = lbl_enc.fit_transform(data.Overall_Sentiment.values)
overall_np = np.array(overall_col)
overall_np = np.reshape(overall_np, (-1,1))
sentiments = np.append(sentiments, overall_np, axis = 1)

#print("Sentiments Shape: ",sentiments.shape)
#print(sentiments)
#overall_col = np.reshape(overall_col,-1,1)
#print(overall_col.shape)

# print(sarcasm_col)
# print(data.Sarcasm.values)

#print ("Data Shape: ",data.shape)
   
#### Splitting into training and testing data in 90:10 ratio
X_train, X_test, y_train, y_test = train_test_split(data.Corrected_text.values, sentiments, random_state=42, test_size=0.1, shuffle=True)
#print("X-Train Shape: ",X_train.shape)
#print("X-test Shape: ",X_test.shape)
#print("Y-Train Shape: ",y_train.shape) 
#print("y-test Shape: ",y_test.shape)
print("\n\n")

##### Extracting TF-IDF parameters
tfidf = TfidfVectorizer(max_features=1000, analyzer='word',ngram_range=(1,3))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.fit_transform(X_test)

#### Extracting Count Vectors Parameters
count_vect = CountVectorizer(analyzer='word')
count_vect.fit(data['Corrected_text'])
X_train_count_vect = count_vect.transform(X_train)
X_test_count_vect =  count_vect.transform(X_test)


#print("X_train_tfidf rows: ", X_train_tfidf.shape[0])
#print("X_test_tfidf rows: ",X_test_tfidf.shape[0])


def runMultinomialNB(X_train_extra, X_test_extra):    
    
    # Model 1: Multinomial Naive Bayes Classifier

    nb1 = MultinomialNB()
    nb2 = MultinomialNB()
    nb3 = MultinomialNB()
    nb4 = MultinomialNB()
    nb5 = MultinomialNB()
    
    nb1.fit(X_train_extra, y_train[:,0])
    nb2.fit(X_train_extra, y_train[:,1])
    nb3.fit(X_train_extra, y_train[:,2])
    nb4.fit(X_train_extra, y_train[:,3])
    nb5.fit(X_train_extra, y_train[:,4])
    
    Accuracies=[]
    rows = X_test_extra.shape[0]
    for i in range(rows):            
        Predictions = []
        pred1 = nb1.predict(X_test_extra[i,:].reshape(1,-1))[0]    
        Predictions.append(pred1)
        pred2 = nb2.predict(X_test_extra[i,:].reshape(1,-1))[0]
        Predictions.append(pred2)
        pred3 = nb3.predict(X_test_extra[i,:].reshape(1,-1))[0]
        Predictions.append(pred3)
        pred4 = nb4.predict(X_test_extra[i,:].reshape(1,-1))[0]
        Predictions.append(pred4)
        pred5 = nb5.predict(X_test_extra[i,:].reshape(1,-1))[0]
        Predictions.append(pred5)
        ##print(Predictions)
        intersection = np.count_nonzero(np.bitwise_and(y_test[i,:],Predictions), axis=0)
        union = np.count_nonzero(np.bitwise_or(y_test[i,:],Predictions), axis=0)
        # print (intersecTP,"  ",unionTP)
        Accuracies.append(float(intersection)/float(union))

    #Average this value over all the test samples to compute the final test accuracy
    percent_avg = sum(Accuracies)/rows*100
    print ("Avg accuracy - MultinomialNB: %",end=" " ) #percent_avg,"%") 
    print('%0.2f'% percent_avg)
    

def runSGDClassifier(X_train_extra, X_test_extra):    
        
    # Model 2: Linear SVM
    lsvm1 = SGDClassifier(alpha=0.001, random_state=5, max_iter=15, tol=None)
    lsvm2 = SGDClassifier(alpha=0.001, random_state=5, max_iter=15, tol=None)
    lsvm3 = SGDClassifier(alpha=0.001, random_state=5, max_iter=15, tol=None)
    lsvm4 = SGDClassifier(alpha=0.001, random_state=5, max_iter=15, tol=None)
    lsvm5 = SGDClassifier(alpha=0.001, random_state=5, max_iter=15, tol=None)
    
    lsvm1.fit(X_train_extra, y_train[:,0])
    lsvm2.fit(X_train_extra, y_train[:,1])
    lsvm3.fit(X_train_extra, y_train[:,2])
    lsvm4.fit(X_train_extra, y_train[:,3])
    lsvm5.fit(X_train_extra, y_train[:,4])
    
    Accuracies=[]
    rows = X_test_extra.shape[0]
    for i in range(rows):            
        Predictions = []
        pred1 = lsvm1.predict(X_test_extra[i,:].reshape(1,-1))[0]    
        Predictions.append(pred1)
        pred2 = lsvm2.predict(X_test_extra[i,:].reshape(1,-1))[0]
        Predictions.append(pred2)
        pred3 = lsvm3.predict(X_test_extra[i,:].reshape(1,-1))[0]
        Predictions.append(pred3)
        pred4 = lsvm4.predict(X_test_extra[i,:].reshape(1,-1))[0]
        Predictions.append(pred4)
        pred5 = lsvm5.predict(X_test_extra[i,:].reshape(1,-1))[0]
        Predictions.append(pred5)
        ##print(Predictions)
        intersection = np.count_nonzero(np.bitwise_and(y_test[i,:],Predictions), axis=0)
        union = np.count_nonzero(np.bitwise_or(y_test[i,:],Predictions), axis=0)
        # print (intersecTP,"  ",unionTP)
        Accuracies.append(float(intersection)/float(union))

    #Average this value over all the test samples to compute the final test accuracy
    percent_avg = sum(Accuracies)/rows*100
    print ("Avg accuracy - LinearSVM: %",end=" " ) #percent_avg,"%") 
    print('%0.2f'% percent_avg)


def runLogisticRegression(X_train_extra, X_test_extra):
    # Model 3: logistic regression
    
    logreg1 = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=4000)
    logreg2 = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=4000)
    logreg3 = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=4000)
    logreg4 = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=4000)
    logreg5 = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=4000)
    
    logreg1.fit(X_train_extra, y_train[:,0])
    logreg2.fit(X_train_extra, y_train[:,1])
    logreg3.fit(X_train_extra, y_train[:,2])
    logreg4.fit(X_train_extra, y_train[:,3])
    logreg5.fit(X_train_extra, y_train[:,4])

    Accuracies=[]
    rows = X_test_tfidf.shape[0]
    for i in range(rows):            
        Predictions = []
        pred1 = logreg1.predict(X_test_extra[i,:].reshape(1,-1))[0]    
        Predictions.append(pred1)
        pred2 = logreg2.predict(X_test_extra[i,:].reshape(1,-1))[0]
        Predictions.append(pred2)
        pred3 = logreg3.predict(X_test_extra[i,:].reshape(1,-1))[0]
        Predictions.append(pred3)
        pred4 = logreg4.predict(X_test_extra[i,:].reshape(1,-1))[0]
        Predictions.append(pred4)
        pred5 = logreg5.predict(X_test_extra[i,:].reshape(1,-1))[0]
        Predictions.append(pred5)
        ##print(Predictions)
        intersection = np.count_nonzero(np.bitwise_and(y_test[i,:],Predictions), axis=0)
        union = np.count_nonzero(np.bitwise_or(y_test[i,:],Predictions), axis=0)
        # print (intersecTP,"  ",unionTP)
        Accuracies.append(float(intersection)/float(union))

    #Average this value over all the test samples to compute the final test accuracy
    percent_avg = sum(Accuracies)/rows*100
    print ("Avg accuracy-Logistic Regresssion: %",end=" " ) #percent_avg,"%") 
    print('%0.2f'% percent_avg)


def runRandomForestClassifier(X_train_extra, X_test_extra):
     # Model 4: Random Forest Classifier

    rf1 = RandomForestClassifier(n_estimators=500)
    rf2 = RandomForestClassifier(n_estimators=500)
    rf3 = RandomForestClassifier(n_estimators=500)
    rf4 = RandomForestClassifier(n_estimators=500)
    rf5 = RandomForestClassifier(n_estimators=500)
        
    rf1.fit(X_train_extra, y_train[:,0])
    rf2.fit(X_train_extra, y_train[:,1])
    rf3.fit(X_train_extra, y_train[:,2])
    rf4.fit(X_train_extra, y_train[:,3])
    rf5.fit(X_train_extra, y_train[:,4])

    Accuracies=[]
    rows = X_test_tfidf.shape[0]
    for i in range(rows):            
        Predictions = []
        pred1 = rf1.predict(X_test_extra[i,:].reshape(1,-1))[0]    
        Predictions.append(pred1)
        pred2 = rf2.predict(X_test_extra[i,:].reshape(1,-1))[0]
        Predictions.append(pred2)
        pred3 = rf3.predict(X_test_extra[i,:].reshape(1,-1))[0]
        Predictions.append(pred3)
        pred4 = rf4.predict(X_test_extra[i,:].reshape(1,-1))[0]
        Predictions.append(pred4)
        pred5 = rf5.predict(X_test_extra[i,:].reshape(1,-1))[0]
        Predictions.append(pred5)
        ##print(Predictions)
        intersection = np.count_nonzero(np.bitwise_and(y_test[i,:],Predictions), axis=0)
        union = np.count_nonzero(np.bitwise_or(y_test[i,:],Predictions), axis=0)
        # print (intersecTP,"  ",unionTP)
        Accuracies.append(float(intersection)/float(union))

    #Average this value over all the test samples to compute the final test accuracy
    percent_avg = sum(Accuracies)/rows*100
    print ("Avg accuracy - Random Forest: %",end=" " ) #percent_avg,"%") 
    print('%0.2f'% percent_avg)
    
    
print("for TF-IDF:")
runMultinomialNB(X_train_tfidf, X_test_tfidf)
runSGDClassifier(X_train_tfidf, X_test_tfidf)
runLogisticRegression(X_train_tfidf, X_test_tfidf)
runRandomForestClassifier(X_train_tfidf, X_test_tfidf)
print("\nfor count vector:")
runMultinomialNB(X_train_count_vect, X_test_count_vect)
runSGDClassifier(X_train_count_vect, X_test_count_vect)
runLogisticRegression(X_train_count_vect, X_test_count_vect)
runRandomForestClassifier(X_train_count_vect, X_test_count_vect)





