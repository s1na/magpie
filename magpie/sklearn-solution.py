
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, chi2, f_classif
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn import metrics

from old_hamshahri_reader import OldHamshahriReader
import config



tuned_params = [{'C': [1, 10, 100, 1000]}]
svc_tuned_params = [{'kernel': ['rbf'], 'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001]},]
if __name__ == '__main__':
    rd = OldHamshahriReader(root=config.CORPORA_ROOT)
    docs, labels = rd.sklearn_docs(config.TOT_DOCS)
    #vectorizer = CountVectorizer(docs)
    vectorizer = TfidfVectorizer(lowercase=False, max_df=0.8)

    fs = vectorizer.fit_transform(docs)
    #vectorizer.build_preprocessor()
    selector = SelectPercentile(chi2, percentile=10)
    selector.fit(fs, labels)
    fs = selector.transform(fs)
    fs_train, fs_test, labels_train, labels_test = train_test_split(
        fs, labels, test_size=0.4, random_state=0
    )

    clf = None
    pred = None
    grid_search = False
    if config.CLASSIFIER == 'NaiveBayes':
        clf = BernoulliNB()
    elif config.CLASSIFIER == 'LinearSVC':
        if config.SELF_TRAINING:
            clf = LinearSVC(C=1)
        else:
            clf = GridSearchCV(LinearSVC(), tuned_params, cv=5, scoring='accuracy')
            grid_search = True
    elif config.CLASSIFIER == 'SVC':
        clf = GridSearchCV(SVC(), svc_tuned_params, cv=5, scoring='accuracy')
        grid_search = True
    elif config.CLASSIFIER == 'DecisionTree':
        clf = DecisionTreeClassifier()
        fs_train = fs_train.toarray()
        fs_test = fs_test.toarray()
    elif config.CLASSIFIER == 'Ensemble':
        #clf = AdaBoostClassifier(n_estimators=100)
        clf = GradientBoostingClassifier(n_estimators=5, random_state=0)
        fs_train = fs_train.toarray()
        fs_test = fs_test.toarray()


    if config.SELF_TRAINING:
        fl = fs_train.shape[0]
        ll = labels_train.shape[0]
        fsarr = fs_train.toarray()
        cur_fs = fsarr[:fl / 10]
        cur_labels = labels_train[:ll / 10]

        clf.fit(cur_fs, cur_labels)
        print clf.classes_
        for i in range(1, 10):
            new_fs = fsarr[(i * fl) / 10:((i + 1) * fl) / 10]
            confidence_scores = clf.decision_function(new_fs)
            most_confident_samples = confidence_scores.max(axis=1).argsort()[
                -1 * (confidence_scores.shape[0] / 10):]
            most_confident_labels = confidence_scores[most_confident_samples].argmax(axis=1)
            cur_fs = np.append(cur_fs, new_fs[most_confident_samples], axis=0)
            cur_labels = np.append(cur_labels, clf.classes_[most_confident_labels])
            clf.fit(cur_fs, cur_labels)
        pred = clf.predict(fs_test)


    else:
        clf.fit(fs_train, labels_train)
        pred = clf.predict(fs_test)

    if grid_search:
        print clf.best_estimator_
    #print metrics.classification_report(labels_test, pred)
    print "%s ** Accuracy: %f\tPrecision: %f\tRecall: %f\tF1: %f" % (
        config.CLASSIFIER,
        metrics.accuracy_score(labels_test, pred),
        metrics.precision_score(labels_test, pred),
        metrics.recall_score(labels_test, pred),
        metrics.f1_score(labels_test, pred),
    )
