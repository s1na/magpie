# -*- coding: utf-8 -*-
import nltk
from collections import Counter

from hazm import Normalizer, Stemmer, word_tokenize
from hazm.HamshahriReader import HamshahriReader

import config
from old_hamshahri_reader import OldHamshahriReader


def doc_features(doc, dist_words):
    words_set = set(doc['words'])
    features = {}
    for word in dist_words:
        features['contains(%s)' % word] = (word in words_set)

    return features

if __name__ == '__main__':
    rd = OldHamshahriReader(config.corpora_root)
    #docs = rd.docs(count=100)
    #rd = HamshahriReader(config.corpora_root)
    #docs = [doc for doc in rd.docs()]
    counter = Counter()
    docs = []
    normalizer = Normalizer()
    stemmer = Stemmer()
    for doc in rd.docs(count=config.documents_count):
        doc['text'] = normalizer.normalize(doc['text'])
        doc['words'] = [stemmer.stem(word) for word in word_tokenize(doc['text'])]
        counter.update([doc['cat']])
        docs.append(doc)

    print counter
    all_words = []
    for doc in docs:
        all_words.extend(doc['words'])

    dist = nltk.FreqDist(word for word in all_words)
    word_features = [word for word in set(all_words) if len(word) > 4 and dist[word] > 40]
    print len(word_features) / float(len(all_words)) * 100.0
    features_set = [(doc_features(doc, word_features), doc['cat']) for doc in docs]
    train_set, test_set = features_set[:len(docs)/2], features_set[len(docs)/2:len(docs)]
    classifier = nltk.NaiveBayesClassifier.train(train_set)

    print nltk.classify.accuracy(classifier, test_set)
    classifier.show_most_informative_features(10)
