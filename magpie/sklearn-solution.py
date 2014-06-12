
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, chi2, f_classif
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC, SVC
from sklearn import metrics

from old_hamshahri_reader import OldHamshahriReader
import config

if __name__ == '__main__':
    rd = OldHamshahriReader(root=config.CORPORA_ROOT)
    docs, labels = rd.sklearn_docs(config.TOT_DOCS)
    #vectorizer = CountVectorizer(docs)
    vectorizer = TfidfVectorizer()

    fs = vectorizer.fit_transform(docs)
    selector = SelectPercentile(chi2, percentile=10)
    selector.fit(fs, labels)
    fs = selector.transform(fs)
    fs_train, fs_test, labels_train, labels_test = train_test_split(
        fs, labels, test_size=0.4, random_state=0
    )

    clf = BernoulliNB()
    clf.fit(fs_train, labels_train)
    pred = clf.predict(fs_test)
    clf2 = LinearSVC()
    clf2.fit(fs_train, labels_train)
    pred2 = clf2.predict(fs_test)
    #print metrics.classification_report(labels_test, pred)
    print "BernoulliNB ** Accuracy: %f\tPrecision: %f\tRecall: %f\tF1: %f" % (
        metrics.accuracy_score(labels_test, pred),
        metrics.precision_score(labels_test, pred),
        metrics.recall_score(labels_test, pred),
        metrics.f1_score(labels_test, pred),
    )
    #print metrics.classification_report(labels_test, pred2)
    print "LinearSVC ** Accuracy: %f\tPrecision: %f\tRecall: %f\tF1: %f" % (
        metrics.accuracy_score(labels_test, pred2),
        metrics.precision_score(labels_test, pred2),
        metrics.recall_score(labels_test, pred2),
        metrics.f1_score(labels_test, pred2),
    )

