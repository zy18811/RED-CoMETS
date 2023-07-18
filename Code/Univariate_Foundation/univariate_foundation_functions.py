import numpy as np
from pyts.approximation import SymbolicFourierApproximation
from sklearn.model_selection import cross_val_score

from Univariate_Foundation.coeye_functions import dynamic_voting
from Utilities.sax  import  sax_transform

"""
Random Pair Selection
"""

def random_lenses(n_lenses, X_train, seed):
    maxCoof = 130
    if (X_train.shape[1] < maxCoof): maxCoof = X_train.shape[1] - 1
    if (X_train.shape[1] < 100):
        n_segments = list(range(5, maxCoof, 5))
    else:
        n_segments = list(range(10, maxCoof, 10))

    maxBin = 26
    if (X_train.shape[1] < maxBin): maxBin = X_train.shape[1] - 2
    if (X_train.shape[0] < maxBin): maxBin = X_train.shape[0] - 2
    alphas = range(3, maxBin)

    rng = np.random.default_rng(seed)
    lenses = np.array(list(zip(rng.choice(n_segments, size=n_lenses), rng.choice(alphas, size=n_lenses))))
    return lenses

"""
Voting
"""

classic_voting = dynamic_voting

def sum_rule_uniform(matrices, labels):
    return [labels[i] for i in np.sum(np.array(matrices), axis = 0).argmax(axis = 1)]

def sum_rule_meanmax(rf_matrices, classes, ret_fused_mats=False):
    w = np.array([np.mean(mat.max(axis=1)) for mat in rf_matrices]).reshape(-1, 1)
    weighted_mats = rf_matrices * w[:, np.newaxis]
    if ret_fused_mats:
        return np.sum(weighted_mats, axis=0)
    return sum_rule_uniform(weighted_mats, classes)

def sum_rule_validation(X_train, y_train, sax_lenses, sax_clfs, sfa_lenses, sfa_clfs, rf_matrices, classes):
    accs = []

    for clf, (n_coefs, n_bins) in zip(sfa_clfs, sfa_lenses):
        SFA = SymbolicFourierApproximation(strategy='uniform', n_coefs=n_coefs, n_bins=n_bins, alphabet='ordinal')
        X_sfa = SFA.fit_transform(X_train)
        cv = np.min([5, len(y_train) // len(list(set(y_train)))])
        scores = cross_val_score(clf, X_sfa, y_train, cv=cv)#, n_jobs=-1)
        accs.append(scores.mean())

    for clf, (n_coefs, n_bins) in zip(sax_clfs, sax_lenses):
        X_sax = sax_transform(X_train, alphabet_size=n_bins, word_length=n_coefs)
        cv = np.min([5, len(y_train) // len(list(set(y_train)))])
        scores = cross_val_score(clf, X_sax, y_train, cv=cv)#, n_jobs=-1)
        accs.append(scores.mean())

    weighted_mats = rf_matrices * np.array(accs).reshape(-1, 1)[:, np.newaxis]
    return sum_rule_uniform(weighted_mats, classes)
