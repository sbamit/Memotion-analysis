#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from sklearn.svm import SVC
import numpy as np
import pandas as pd
import os
#import nltk
#nltk.download('stopwords')
from textblob import Word
datapath = os.getcwd()
datapath = datapath + "/data_6512.csv" 
if not os.path.exists(datapath):
    print ("Data cannot be located!")

data = pd.read_csv(datapath)
# =============================================================================
# np_data = np.array(data)
# print(arr.shape)
# 
# for i in range(3):
#    np_data = np.delete(np_data,0,axis=1)
# print(np_data.shape)
# =============================================================================

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

#Lemmatisation
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
y = lbl_enc.fit_transform(data.sentiment.values)
# Splitting into training and testing data in 90:10 ratio
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(data.content.values, y, stratify=y, random_state=42, test_size=0.1, shuffle=True)


print(data['Corrected_text'])
# =============================================================================
# for row in arr:
#     print(row[0])
# file = open(datapath, 'r')
# reader = csv.reader(file)
# for row in reader:
#     print(row[3])
# =============================================================================
