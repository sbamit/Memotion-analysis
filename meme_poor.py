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
from sklearn.metrics import accuracy_score
import os
#import nltk
#nltk.download('all')
datapath = os.getcwd()
datapath = datapath + "/data_7000_new.csv" 
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
#overall_col = np.reshape(overall_col,-1,1)
#print(overall_col.shape)

# print(sarcasm_col)
# print(data.Sarcasm.values)

#print ("Data Shape: ",data.shape)

def runNLP(labels_array):
    
    # Splitting into training and testing data in 90:10 ratio
    from sklearn.model_selection import train_test_split
    #X_train, X_test, y_train, y_test = train_test_split(data.Corrected_text.values, y, stratify=y, random_state=42, test_size=0.1, shuffle=True)
    X_train_humour, X_test_humour, y_train_humour, y_test_humour = train_test_split(data.Corrected_text.values, labels_array, random_state=42, test_size=0.1, shuffle=True)
    print ("X-Train Shape: ",X_train_humour.shape)
    print ("X-test Shape: ",X_test_humour.shape)
    print ("Y-Train Shape: ",y_train_humour.shape) 
    print ("y-test Shape: ",y_test_humour.shape)
    
    
    ##### Extracting TF-IDF parameters
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer(max_features=1000, analyzer='word',ngram_range=(1,3))
    X_train_tfidf_humour = tfidf.fit_transform(X_train_humour)
    X_test_tfidf_humour = tfidf.fit_transform(X_test_humour)
    
    # Extracting Count Vectors Parameters
    from sklearn.feature_extraction.text import CountVectorizer
    count_vect = CountVectorizer(analyzer='word')
    count_vect.fit(data['Corrected_text'])
    X_train_count_humour =  count_vect.transform(X_train_humour)
    X_test_count_humour =  count_vect.transform(X_test_humour)
    
    
    #########################Training Models using TF-IDF ###########################
     
    # Model 1: Multinomial Naive Bayes Classifier
    from sklearn.naive_bayes import MultinomialNB
    nb = MultinomialNB()
    nb.fit(X_train_tfidf_humour, y_train_humour)
    y_pred = nb.predict(X_test_tfidf_humour)
    print('naive bayes tfidf accuracy %s' % accuracy_score(y_pred, y_test_humour))
    
    # Model 2: Linear SVM
    from sklearn.linear_model import SGDClassifier
    lsvm = SGDClassifier(alpha=0.001, random_state=5, max_iter=15, tol=None)
    lsvm.fit(X_train_tfidf_humour, y_train_humour)
    y_pred = lsvm.predict(X_test_tfidf_humour)
    print('svm using tfidf accuracy %s' % accuracy_score(y_pred, y_test_humour))
    
    # Model 3: logistic regression
    from sklearn.linear_model import LogisticRegression
    logreg = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=4000)
    logreg.fit(X_train_tfidf_humour, y_train_humour)
    y_pred = logreg.predict(X_test_tfidf_humour)
    print('log reg tfidf accuracy %s' % accuracy_score(y_pred, y_test_humour))
    #log reg tfidf accuracy 0.5443159922928709
    
    # Model 4: Random Forest Classifier
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=500)
    rf.fit(X_train_tfidf_humour, y_train_humour)
    y_pred = rf.predict(X_test_tfidf_humour)
    print('random forest tfidf accuracy %s' % accuracy_score(y_pred, y_test_humour))
    
    #########################Training Models using Count vector ###########################
    
    # Model 1: Multinomial Naive Bayes Classifier
    from sklearn.naive_bayes import MultinomialNB
    nb = MultinomialNB()
    nb.fit(X_train_count_humour, y_train_humour)
    y_pred = nb.predict(X_test_count_humour)
    print('naive bayes count vectors accuracy %s' % accuracy_score(y_pred, y_test_humour))
    #naive bayes count vectors accuracy 0.7764932562620424
    
    # Model 2: Linear SVM
    from sklearn.linear_model import SGDClassifier
    lsvm = SGDClassifier(alpha=0.001, random_state=5, max_iter=15, tol=None)
    lsvm.fit(X_train_count_humour, y_train_humour)
    y_pred = lsvm.predict(X_test_count_humour)
    print('lsvm using count vectors accuracy %s' % accuracy_score(y_pred, y_test_humour))
    #lsvm using count vectors accuracy 0.7928709055876686
    #Model 3: Logistic Regression
    from sklearn.linear_model import LogisticRegression
    logreg = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=4000)
    logreg.fit(X_train_count_humour, y_train_humour)
    y_pred = logreg.predict(X_test_count_humour)
    print('log reg count vectors accuracy %s' % accuracy_score(y_pred, y_test_humour))
    #log reg count vectors accuracy 0.7851637764932563
    
    # Model 4: Random Forest Classifier
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=500)
    rf.fit(X_train_count_humour, y_train_humour)
    y_pred = rf.predict(X_test_count_humour)
    print('random forest with count vectors accuracy %s' % accuracy_score(y_pred, y_test_humour))
    #random forest with count vectors accuracy 0.7524084778420038
    
    
    
    #print(data['Corrected_text'])
print("\n************************humor***************************")
runNLP(humor_col)
print("\n************************sarcasm***************************")
runNLP(sarcasm_col)
print("\n************************offensive***************************")
runNLP(offensive_col)
print("\n************************motivation***************************")
runNLP(motivational_col)
print("\n************************overall***************************")
runNLP(overall_col)




