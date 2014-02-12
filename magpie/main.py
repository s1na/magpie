# -*- coding: utf-8 -*-
import nltk

from hazm import Normalizer, Stemmer, word_tokenize
from hazm.HamshahriReader import HamshahriReader

import config


def doc_features(doc, dist_words):
    words_set = set(doc['words'])
    features = {}
    for word in dist_words:
        features['contains(%s)' % word] = (word in words_set)

    return features

if __name__ == '__main__':
    rd = HamshahriReader(config.corpora_root)
    #docs = [doc for doc in rd.docs()]
    docs = []
    normalizer = Normalizer()
    stemmer = Stemmer()
    for doc in rd.docs():
        doc['text'] = normalizer.normalize(doc['text'])
        doc['words'] = [stemmer.stem(word) for word in word_tokenize(doc['text'])]
        docs.append(doc)

    all_words = []
    for doc in docs:
        all_words.extend(doc['words'])

    dist = nltk.FreqDist(word for word in all_words)
    word_features = [word for word in set(all_words) if len(word) > 4 and dist[word] > 10]
    features_set = [(doc_features(doc, word_features), doc['categories_en'][0]) for doc in docs]
    train_set, test_set = features_set[:60], features_set[60:100]
    classifier = nltk.NaiveBayesClassifier.train(train_set)

    print nltk.classify.accuracy(classifier, test_set)
    classifier.show_most_informative_features(10)
