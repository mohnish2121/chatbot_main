import nltk 
import numpy as np

from nltk.stem.porter import PorterStemmer

stemmer=PorterStemmer()

def tokenized(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of(toke_sentence, allwords):
    toke_sentence=[stem(w) for w in toke_sentence]
    bag= np.zeros(len(allwords),dtype=np.float32)
    for ind, w in enumerate(allwords):
        if w in toke_sentence:
            bag[ind]=1.0

    return bag
