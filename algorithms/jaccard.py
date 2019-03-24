import gensim
import xml.etree.ElementTree as ElementTree
from sklearn.feature_extraction.text import CountVectorizer

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from nltk.stem import PorterStemmer
from collections import Counter
from num2words import num2words

import os
import numpy as np
import math

corpusPath = r"C:\newCorpus"


# Preprocessing

def convert_lower_case(data):
    return np.char.lower(data)


def remove_stop_words(data):
    stop_words = stopwords.words('english')
    stop_words += ['\n\n', '\n', "summary", "end summary"]
    words = word_tokenize(str(data))
    new_text = ""
    for w in words:
        if w not in stop_words and len(w) > 1:
            new_text = new_text + " " + w
    return new_text


def remove_punctuation(data):
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    for i in range(len(symbols)):
        data = np.char.replace(data, symbols[i], ' ')
        data = np.char.replace(data, "  ", " ")
    data = np.char.replace(data, ',', '')
    return data

def remove_apostrophe(data):
    return np.char.replace(data, "'", "")


def stemming(data):
    stemmer = PorterStemmer()

    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        new_text = new_text + " " + stemmer.stem(w)
    return new_text


def convert_numbers(data):
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        try:
            w = num2words(int(w))
        except:
            a = 0
        new_text = new_text + " " + w
    new_text = np.char.replace(new_text, "-", " ")
    return new_text


def preprocess(data):
    data = convert_lower_case(data)
    data = remove_punctuation(data)     # remove comma seperately
    data = remove_apostrophe(data)
    data = remove_stop_words(data)
    data = convert_numbers(data)
    data = stemming(data)
    data = remove_punctuation(data)
    data = convert_numbers(data)
    data = stemming(data)   # needed again as we need to stem the words
    data = remove_punctuation(data)  # needed again as num2word is giving few hypens and commas fourty-one
    data = remove_stop_words(data)  # needed again as num2word is giving stop words 101 - one hundred and one
    return data


#Extracting Data

processed_name = []
processed_text = []
processed_title = []

full_name = []
full_text = []
full_title = []


# counter = 0
# while counter < 3:
for folder in os.listdir(corpusPath):
    for document in os.listdir(os.path.join(corpusPath, folder)):
        fileDir = os.path.join(os.path.join(corpusPath, folder), document)
        with open(fileDir) as Content:
            xml = Content.read()
        xml = '<ROOT>' + xml + '</ROOT>'  # Let's add a root tag
        root = ElementTree.fromstring(xml)
        for doc in root:
            docno = doc.find('DOCNO').text.strip()
            title = doc.find('HEADER').find('H3').find('TI').text.strip()
            text = doc.find('TEXT').text.strip()
            processed_text.append(word_tokenize(str(preprocess(text))))
            processed_title.append(word_tokenize(str(preprocess(title))))
            processed_name.append(word_tokenize(str(preprocess(docno))))
            full_name.append(docno)
            full_text.append(text)
            full_title.append(title)

# for i in range(len(processed_text)):
#     print(processed_text[i])

dictionary = gensim.corpora.Dictionary(processed_text)


# print(dictionary[5])
# print("Number of words in dictionary:",len(dictionary))
# for i in range(len(dictionary)):
#     print(i, dictionary[i])


# create a corpus. A corpus is a list of bags of words

corpus = [dictionary.doc2bow(doc) for doc in processed_text]

#Now we create a tf-idf model from the corpus



X = vectorizer.fit_transform(corpus)