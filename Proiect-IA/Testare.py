import nltk
import pandas as pd
import numpy as np
import random
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import RegexpTokenizer
from collections import Counter


def tokenize(text):
    '''Generic wrapper around different tokenization methods.
    '''
    return nltk.RegexpTokenizer('\w+').tokenize(text)


def get_representation(toate_cuvintele, how_many):
    '''Extract the first most common words from a vocabulary
    and return two dictionaries: word to index and index to word
           @  che  .   ,   di  e
    text0  0   1   0   2   0   1
    text1  1   2 ...
    ...
    textN  0   0   1   1   0   2
    '''
    most_comm = toate_cuvintele.most_common(how_many)
    wd2idx = {}
    idx2wd = {}
    for idx, itr in enumerate(most_comm):
        cuvant = itr[0]
        wd2idx[cuvant] = idx
        idx2wd[idx] = cuvant
    return wd2idx, idx2wd


def get_corpus_vocabulary(corpus):
    '''Write a function to return all the words in a corpus.
    '''
    counter = Counter()
    for text in corpus:
        tokens = tokenize(text)
        new_tokens = []
        for cuvant in tokens:
            if len(cuvant) > 3:
                new_tokens.append(cuvant)
        counter.update(new_tokens)
    return counter


def text_to_bow(text, wd2idx):
    '''Convert a text to a bag of words representation.
           @  che  .   ,   di  e
    text   0   1   0   2   0   1
    '''
    features = np.zeros(len(wd2idx))
    nr_cuvinte = 0
    for token in tokenize(text):
        nr_cuvinte += 1
        if token in wd2idx:
            features[wd2idx[token]] += 1
    return features / nr_cuvinte



def corpus_to_bow(corpus, wd2idx):
    '''Convert a corpus to a bag of words representation.
           @  che  .   ,   di  e
    text0  0   1   0   2   0   1
    text1  1   2 ...
    ...
    textN  0   0   1   1   0   2
    '''

    all_features = []
    for text in corpus:
        all_features.append(text_to_bow(text, wd2idx))
    all_features = np.array(all_features)
    return all_features


def write_prediction(out_file, predictions):
    '''A function to write the predictions to a file.
    id,label
    5001,1
    5002,1
    5003,1
    ...
    '''
    with open(out_file, 'w') as fout:
        # aici e open in variabila 'fout'
        fout.write('id,label\n')
        start_id = 5001
        for i, pred in enumerate(predictions):
            linie = str(i + start_id) + ',' + str(pred) + '\n'
            fout.write(linie)
    # aici e fisierul closed


def split(date, labels, procentaj_valid=0.25):
    '''75% train, 25% valid
    important! mai intai facem shuffle la date
    '''
    indici=np.arange(0, len(labels))
    random.shuffle(indici)
    N = int((1 - procentaj_valid) * len(labels))
    train = data[indici[:N]]
    valid = data[indici[N:]]
    y_train = labels[indici[:N]]
    y_valid = labels[indici[N:]]
    return train, valid, y_train, y_valid

def acc(y_true, y_pred):
    return np.mean((y_true == y_pred).astype(int)) *100

def cross_validate(k, data, labels):
    '''Split the data into k chunks.
    iteration 0:
        chunk 0 is for validation, chunk[1:] for train
    iteration 1:
        chunk 1 is for validation, chunk[0] + chunk[2:] for train
    ...
    iteration k:
        chunk k is for validation, chunk[:k] for train
    '''
    chunk_size = len(labels) // k
    indici = np.arange(0, len(labels))
    random.shuffle(indici)
    for i in range(0, len(labels), chunk_size):
        valid_indici = indici[i:i + chunk_size]
        train_indici = np.concatenate([indici[0:i], indici[i + chunk_size:]])
        train = data[train_indici]
        valid = data[valid_indici]
        y_train = labels[train_indici]
        y_valid = labels[valid_indici]
        yield train, valid, y_train, y_valid





train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
corpus = train_df['text']

toate_cuvintele = get_corpus_vocabulary(corpus)
print(toate_cuvintele)
wd2idx, idx2wd = get_representation(toate_cuvintele, 1000)


data = corpus_to_bow(corpus, wd2idx)
labels = train_df['label'].values


test_data = corpus_to_bow(test_df['text'], wd2idx)

from sklearn import svm

clf = svm.SVC()
clf.fit(data, labels)
predictii = clf.predict(test_data)

write_prediction('sample_submission.csv', predictii)


'''acurateti = []
for train, valid, y_train, y_valid in cross_validate(10, data, labels):
    clf.fit(train, y_train)
    predictii = clf.predict(valid)
    val = acc(y_valid, predictii)
    #print(val)
    acurateti.append(val)
print(np.mean(acurateti))'''


