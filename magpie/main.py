# -*- coding: utf-8 -*-
from collections import Counter
import cPickle as pickle
import heapq

import nltk
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.grid_search import RandomizedSearchCV
from hazm import Normalizer, Stemmer, word_tokenize
#from hazm.HamshahriReader import HamshahriReader

import config
from old_hamshahri_reader import OldHamshahriReader


def dimension_reduction(terms, dist):
    return [term for term in set(terms) if len(term) > 4 and dist[term] > 40]

def doc_features(doc, dist_words):
    words_set = set(doc['words'])
    features = {}
    for word in dist_words:
        features['contains(%s)' % word] = (word in words_set)

    return features

def evaluate(classifier, gold, labels):
    accuracy, precision, recall = 0.0, 0.0, 0.0
    confusion_matrix = np.zeros((len(labels), len(labels)))
    results = classifier.batch_classify([fs for (fs,l) in gold])

    for ((fs, l), r) in zip(gold, results):
        confusion_matrix[labels.index(l), labels.index(r)] += 1

    accuracy = confusion_matrix.diagonal().sum() / confusion_matrix.sum()
    col_sums = confusion_matrix.sum(0)
    precision = (
        confusion_matrix.diagonal()[col_sums.nonzero()] /
        col_sums[col_sums.nonzero()]).sum() / len(col_sums[col_sums.nonzero()])
    row_sums = confusion_matrix.sum(1)
    recall = (
        confusion_matrix.diagonal()[row_sums.nonzero()] /
        row_sums[row_sums.nonzero()]).sum() / len(row_sums[row_sums.nonzero()])

    #print labels
    #print confusion_matrix
    return precision, recall, accuracy


if __name__ == '__main__':
    rd = OldHamshahriReader(config.corpora_root)
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

    word_features = dimension_reduction(all_words, dist)
    print len(word_features) / float(len(all_words)) * 100.0

    features_set = [(doc_features(doc, word_features), doc['cat']) for doc in docs]
    #train_set, test_set = features_set[:len(docs)/2], features_set[len(docs)/2:len(docs)]
    print len(features_set), len(docs)
    train_set, test_set, unlabeled_set = features_set[:500], features_set[500:1000], features_set[1000:2000]

    classifier = None
    if config.classifier_type == 'NaiveBayes':
        classifier = nltk.NaiveBayesClassifier.train(train_set)

        if config.semi_supervised:
            loops = 0
            probs = []
            while loops < 5:
                most_promisings = 100 * [(0, 0, None, None)]

                i = 0
                for (fs, l) in unlabeled_set:
                    res = classifier.prob_classify(fs)
                    (p, l) = max([(res.prob(l), l) for l in res.samples()])
                    if p > most_promisings[0][0]:
                        heapq.heappushpop(most_promisings, (p, i, fs, l))
                    i += 1

                train_set.extend([(fs, l) for (p, i, fs, l) in most_promisings])
                indices = [i for (p, i, fs, l) in most_promisings]
                indices.sort(reverse=True)
                for i in indices:
                    del(unlabeled_set[i])

                classifier = nltk.NaiveBayesClassifier.train(train_set)

                print [p for (p, i, fs, l) in most_promisings]
                print loops
                loops += 1

    elif config.classifier_type == 'DecisionTree':
        classifier = nltk.classify.DecisionTreeClassifier.train(train_set, entropy_cutoff=0, support_cutoff=0)
    elif config.classifier_type == 'SVC':
        classifier = nltk.classify.scikitlearn.SklearnClassifier(SVC(), sparse=False).train(train_set)
    elif config.classifier_type == 'LinearSVC':
        classifier = nltk.classify.scikitlearn.SklearnClassifier(LinearSVC(), sparse=False).train(train_set)
    else:
        raise ValueError, "Classifier type unknown."

    precision, recall, accuracy = evaluate(classifier, test_set, counter.keys())
    print "Precision: %g\tRecall: %g\tAccuracy: %g" % (precision, recall, accuracy)
    #classifier.show_most_informative_features(10)
