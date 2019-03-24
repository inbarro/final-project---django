
import xml.etree.ElementTree as ElementTree


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter
from num2words import num2words

import os
import numpy as np
import math



corpusPath = r"C:\newCorpus"

query= "Falkland petroleum exploration"

alpha = 0.3


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


# Calculating DF for all words

N = len(processed_name)

DF = {}

for i in range(N):
    tokens = processed_text[i]
    for w in tokens:
        try:
            DF[w].add(i)
        except:
            DF[w] = {i}

    tokens = processed_title[i]
    for w in tokens:
        try:
            DF[w].add(i)
        except:
            DF[w] = {i}
for i in DF:
    DF[i] = len(DF[i])


total_vocab_size = len(DF)

total_vocab_size

total_vocab = [x for x in DF]

print(total_vocab[:20])


def print_doc(id):
    print("document name: ", full_name[id])
    print("document title: ", full_title[id])
    # print("document text: ", full_text[id])


def doc_freq(word):
    c = 0
    try:
        c = DF[word]
    except:
        pass
    return c


# Calculating TF-IDF for body, we will consider this as the actual tf-idf as we will add the title weight to this

doc = 0

tf_idf = {}

for i in range(N):

    tokens = processed_text[i]

    counter = Counter(tokens + processed_title[i])
    words_count = len(tokens + processed_title[i])

    for token in np.unique(tokens):
        tf = counter[token] / words_count
        df = doc_freq(token)
        idf = np.log((N + 1) / (df + 1))

        tf_idf[doc, token] = tf * idf

    doc += 1
# tf_idf


# Calculating TF-IDF for Title

doc = 0

tf_idf_title = {}

for i in range(N):

    tokens = processed_title[i]
    counter = Counter(tokens + processed_text[i])
    words_count = len(tokens + processed_text[i])

    for token in np.unique(tokens):
        tf = counter[token] / words_count
        df = doc_freq(token)
        idf = np.log((N + 1) / (df + 1))  # numerator is added 1 to avoid negative values

        tf_idf_title[doc, token] = tf * idf

    doc += 1
# tf_idf_title

#Merging the TF-IDF according to weights

for i in tf_idf:
    tf_idf[i] *= alpha

for i in tf_idf_title:
    tf_idf[i] = tf_idf_title[i]

len(tf_idf)


# TF-IDF Matching Score Ranking

def matching_score(k, query):
    preprocessed_query = preprocess(query)
    tokens = word_tokenize(str(preprocessed_query))

    print("Matching Score")
    print("\nQuery:", query)
    print("")
    print(tokens)

    query_weights = {}

    for key in tf_idf:

        if key[1] in tokens:
            try:
                query_weights[key[0]] += tf_idf[key]
            except:
                query_weights[key[0]] = tf_idf[key]

    query_weights = sorted(query_weights.items(), key=lambda x: x[1], reverse=True)

    print("")

    l = []

    for i in query_weights[:10]:
        l.append(i[0])

    print(l)


matching_score(10, query)


#TF-IDF Cosine Similarity Ranking
def cosine_sim(a, b):
    cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    return cos_sim

#Vectorising tf-idf
D = np.zeros((N, total_vocab_size))
for i in tf_idf:
    try:
        ind = total_vocab.index(i[1])
        D[i[0]][ind] = tf_idf[i]
    except:
        pass


def gen_vector(tokens):
    Q = np.zeros((len(total_vocab)))

    counter = Counter(tokens)
    words_count = len(tokens)

    query_weights = {}

    for token in np.unique(tokens):

        tf = counter[token] / words_count
        df = doc_freq(token)
        idf = math.log((N + 1) / (df + 1))

        try:
            ind = total_vocab.index(token)
            Q[ind] = tf * idf
        except:
            pass
    return Q


def cosine_similarity(k, query):
    print("Cosine Similarity")
    preprocessed_query = preprocess(query)
    tokens = word_tokenize(str(preprocessed_query))

    print("\nQuery:", query)
    print("")
    print(tokens)

    d_cosines = []

    query_vector = gen_vector(tokens)

    for d in D:
        d_cosines.append(cosine_sim(query_vector, d))

    out = np.array(d_cosines).argsort()[-k:][::-1]

    print("")

    print(out)


#     for i in out:
#         print(i, dataset[i][0])

Q = cosine_similarity(10, query)

print_doc(0)