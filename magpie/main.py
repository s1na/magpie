# -*- coding: utf-8 -*-
from collections import Counter
import cPickle as pickle

import nltk
from sklearn.svm import SVC, LinearSVC
import numpy as np
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
    #confusion_matrix = len(labels) * [len(labels) * [0]]
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

    return precision, recall, accuracy

#    TP, FP, FN, TN = 0, 1, 2, 3
    #total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0

    #labels_stats = dict([(l, [0, 0, 0, 0]) for l in labels]) # TP, FP, FN, TN
    #results = classifier.batch_classify([fs for (fs,l) in gold])
    ##correct = [l==r for ((fs,l), r) in zip(gold, results)]
    #for ((fs, l), r) in zip(gold, results):
        ## True Positive
        #if l == r:
            #labels_stats[l][TP] += 1
        #else:
            ## False Positive
            #labels_stats[r][FP] += 1

            ## False Negative
            #labels_stats[l][FN] += 1

            ## True Negative
            #for label in labels:
                #if label != l and label != r:
                    #labels_stats[label][TN] += 1

    #vals = labels_stats.values()
    #total_tp = sum([l[TP] for l in vals])
    #total_fp = sum([l[FP] for l in vals])
    #total_fn = sum([l[FN] for l in vals])
    #total_tn = sum([l[TN] for l in vals])
#    precision = float(total_tp) / (total_tp + total_fp)
    #recall = float(total_tp) / (total_tp + total_fn)
#    accuracy = float(total_tp + total_tn) / (total_tp + total_tn + total_tn + total_fn)



if __name__ == '__main__':
    rd = OldHamshahriReader(config.corpora_root)
    #docs = rd.docs(count=100)
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
    train_set, test_set = features_set[:len(docs)/2], features_set[len(docs)/2:len(docs)]

    classifier = None
    if config.classifier_type == 'NaiveBayes':
        classifier = nltk.NaiveBayesClassifier.train(train_set)
    elif config.classifier_type == 'DecisionTree':
        classifier = nltk.classify.DecisionTreeClassifier.train(train_set, entropy_cutoff=0, support_cutoff=0)
    elif config.classifier_type == 'SVC':
        classifier = nltk.classify.scikitlearn.SklearnClassifier(SVC(), sparse=False).train(train_set)
    elif config.classifier_type == 'LinearSVC':
        classifier = nltk.classify.scikitlearn.SklearnClassifier(LinearSVC(), sparse=False).train(train_set)
    else:
        raise ValueError, "Classifier type unknown."

    #print nltk.classify.accuracy(classifier, test_set)
    precision, recall, accuracy = evaluate(classifier, test_set, counter.keys())
    print "Precision: %g\tRecall: %g\tAccuracy: %g" % (precision, recall, accuracy)
    #classifier.show_most_informative_features(10)
