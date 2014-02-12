# -*- coding: utf-8 -*-
import nltk

from hazm import word_tokenize
from hazm.HamshahriReader import HamshahriReader

import config


def doc_features(doc, dist_words):
    words_set = set(word_tokenize(doc['text']))
    features = {}
    for word in dist_words:
        features['contains(%s)' % word] = (word in words_set)

    return features

if __name__ == '__main__':
    rd = HamshahriReader(config.corpora_root)
    docs = [doc for doc in rd.docs()]
    all_words = []
    for doc in docs:
        all_words.extend(word_tokenize(doc['text']))

    dist = nltk.FreqDist(word for word in all_words)
    word_features = dist.keys()[:200]
    features_set = [(doc_features(doc, word_features), doc['categories_en'][0]) for doc in docs]
    train_set, test_set = features_set[:40], features_set[40:80]
    classifier = nltk.NaiveBayesClassifier.train(train_set)

    print nltk.classify.accuracy(classifier, test_set)
    classifier.show_most_informative_features(5)
