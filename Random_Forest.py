from collections import OrderedDict

import numpy as np
from numpy import newaxis
import math
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_validate,cross_val_predict,cross_val_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer

class Random_Forest():
    def __init__(self, X = None, y = None, X_train = None, y_train = None, X_test = None, y_test = None):
        self.X = X
        self.y = y
        self.X_train =  X_train
        self.y_train = y_train
        self.X_train = X_train
        self.y_train = y_train

    def set_X_y(self, filepath):

        reviews_train = load_files("aclImdb/train")
        self.X, self.y = reviews_train.data, reviews_train.target
        self.X = [doc.replace(b"<br />", b" ") for doc in self.X]

        TfidfVectorizer(min_df=5, norm='l2')

    # As in decision tree, can use the best hyperparameters (criterion, alpha (can tune if time), max_features)
    # and bootstrap the input values to comment on variance
    # and use CV to comment on accuracy
    def model_and_plot_oob_error(self, ccp_alpha = 0.15):

        # adopted from https://scikit-learn.org/stable/auto_examples/ensemble/plot_ensemble_oob.html

        RANDOM_STATE = 123

        # NOTE: Setting the `warm_start` construction parameter to `True` disables
        # support for parallelized ensembles but is necessary for tracking the OOB
        # error trajectory during training.
        ensemble_clfs = [
            ("max_features='sqrt', alpha=0, criterion='gini'",
             RandomForestClassifier(warm_start=True, oob_score=True,
                                    max_features="sqrt",
                                    random_state=RANDOM_STATE)),
            ("max_features='sqrt', alpha=0.15, criterion='gini'",
             RandomForestClassifier(warm_start=True, oob_score=True,
                                    max_features="sqrt",
                                    random_state=RANDOM_STATE,
                                    ccp_alpha=ccp_alpha)),
            ("max_features='sqrt', alpha=0, criterion='entropy'",
             RandomForestClassifier(warm_start=True, oob_score=True,
                                    max_features="sqrt",
                                    random_state=RANDOM_STATE)),
            ("max_features='log2', alpha=0, criterion='gini'",
             RandomForestClassifier(warm_start=True, max_features='log2',
                                    oob_score=True,
                                    random_state=RANDOM_STATE)),
            ("max_features=None, alpha=0, criterion='gini'",
             RandomForestClassifier(warm_start=True, max_features=None,
                                    oob_score=True,
                                    random_state=RANDOM_STATE))
        ]

        # Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
        oob_error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

        # Range of `n_estimators` values to explore.
        min_estimators = 25
        max_estimators = 175

        for label, clf in ensemble_clfs:
            for i in range(min_estimators, max_estimators, 25):
                clf.set_params(n_estimators=i)
                clf.fit(self.X, self.y)

                # Record the OOB error for each `n_estimators=i` setting.
                oob_error = 1 - clf.oob_score_
                oob_error_rate[label].append((i, oob_error))

        # Generate the "OOB error rate" vs. "n_estimators" plot.
        for label, clf_err in oob_error_rate.items():
            xs, ys = zip(*clf_err)
            plt.plot(xs, ys, label=label)

        plt.xlim(min_estimators, max_estimators)
        plt.xlabel("n_estimators")
        plt.ylabel("OOB error rate")
        plt.legend(loc="upper right")
        plt.show()

    # can generate one for 3-fold CV and one for 5-fold (if time permits)
    def model_and_plot_cv_error(self, ccp_alpha = 0.15, k=3):

        # adopted from https://scikit-learn.org/stable/auto_examples/ensemble/plot_ensemble_oob.html

        RANDOM_STATE = 123

        # NOTE: Setting the `warm_start` construction parameter to `True` disables
        # support for parallelized ensembles but is necessary for tracking the OOB
        # error trajectory during training.
        ensemble_clfs = [
            ("max_features='sqrt', alpha=0, criterion='gini'",
             RandomForestClassifier(warm_start=True, oob_score=True,
                                    max_features="sqrt",
                                    random_state=RANDOM_STATE)),
            ("max_features='sqrt', alpha=0.15, criterion='gini'",
             RandomForestClassifier(warm_start=True, oob_score=True,
                                    max_features="sqrt",
                                    random_state=RANDOM_STATE,
                                    ccp_alpha=ccp_alpha)),
            ("max_features='sqrt', alpha=0, criterion='entropy'",
             RandomForestClassifier(warm_start=True, oob_score=True,
                                    max_features="sqrt",
                                    random_state=RANDOM_STATE)),
            ("max_features='log2', alpha=0, criterion='gini'",
             RandomForestClassifier(warm_start=True, max_features='log2',
                                    oob_score=True,
                                    random_state=RANDOM_STATE)),
            ("max_features=None, alpha=0, criterion='gini'",
             RandomForestClassifier(warm_start=True, max_features=None,
                                    oob_score=True,
                                    random_state=RANDOM_STATE))
        ]

        # Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
        cv_error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

        # Range of `n_estimators` values to explore.
        min_estimators = 25
        max_estimators = 175

        for label, clf in ensemble_clfs:
            for i in range(min_estimators, max_estimators, 25):
                clf.set_params(n_estimators=i)
                clf.fit(self.X, self.y)

                # Record the CV error for each `n_estimators=i` setting.
                cv_error = 1 - self.k_fold_CV_avg(clf, k)
                cv_error_rate[label].append((i, cv_error))

        # Generate the "3-fold CV error rate" vs. "n_estimators" plot.
        for label, clf_err in cv_error_rate.items():
            xs, ys = zip(*clf_err)
            plt.plot(xs, ys, label=label)

        plt.xlim(min_estimators, max_estimators)
        plt.xlabel("n_estimators")
        plt.ylabel("3-fold CV error rate")
        plt.legend(loc="upper right")
        plt.show()

    # use best hyperparameters from bootstrap=true to see the effect of using the same data set each time
    # for each tree in the random forest
    # evaluate the accuracy with 3-fold CV
    def bootstrap_false(self, ccp_alpha = 0.15):
        RANDOM_STATE = 123
        clf = RandomForestClassifier(max_features="sqrt",
                                     ccp_alpha=ccp_alpha,
                                     bootstrap=False,
                               random_state=RANDOM_STATE)
        print("Bootstrap = False k-fold CV average: %f" %(self.k_fold_CV_avg(clf)))
    def k_fold_CV_avg(self, model, k = 3):
        #np.random.seed(45)  #setting random seed allows us to replicate results
        kf = KFold(n_splits=k)
        results = np.zeros((2, k))
        counter = 0
        for train_index, test_index in kf.split(self.X):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            model.fit(X_train, y_train)
            results[0, counter] = (model.score(X_test, y_test))
            counter = counter + 1
        # returns the average accuracy over k folds
        return np.average(results, axis=1)