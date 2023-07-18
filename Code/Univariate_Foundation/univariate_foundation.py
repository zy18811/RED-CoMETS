import sys
sys.path.append('..')

import numpy as np
from imblearn.over_sampling import SMOTE, RandomOverSampler
from collections import Counter
from pyts.approximation import SymbolicFourierApproximation
from sklearn.ensemble import  RandomForestClassifier
from sklearn.metrics import accuracy_score

from Univariate_Foundation.univariate_foundation_functions import random_lenses, classic_voting, sum_rule_uniform, sum_rule_meanmax, sum_rule_validation
from Utilities.sax import  sax_transform

def univariate_foundation(X_train, y_train, X_test, y_test, p_length, voting_method, n_trees=100, random_seed=42, n_jobs=1, return_mats=False):
    n_lens = int(p_length*len(X_train[0]) // 100)
       
    min_neighbours= min(Counter(y_train).items(), key=lambda k: k[1])[1]
    max_neighbours= max(Counter(y_train).items(), key=lambda k: k[1])[1]

    if(min_neighbours==max_neighbours):
            SMOTE_Xtrain= X_train
            SMOTE_ytrain= y_train

    else:
        if (min_neighbours>5): min_neighbours= 6
        try:
            SMOTE_Xtrain, SMOTE_ytrain = SMOTE(sampling_strategy="all", k_neighbors= min_neighbours-1, random_state=42).fit_resample(X_train, y_train )
        except ValueError:
            SMOTE_Xtrain, SMOTE_ytrain = RandomOverSampler(sampling_strategy='all',random_state=42).fit_resample(X_train,y_train)

    sax_lenses = random_lenses(n_lens, SMOTE_Xtrain, seed=random_seed)
    sfa_lenses = random_lenses(n_lens, SMOTE_Xtrain, seed=random_seed)

    RFmatrices = []

    sfa_clfs = []
    for n_coefs, n_bins  in sfa_lenses:
        SFA = SymbolicFourierApproximation(strategy='uniform', n_coefs=n_coefs, n_bins=n_bins, alphabet='ordinal')
        # Transform to SFA
        X_train_SFA = SFA.fit_transform(SMOTE_Xtrain)
        X_test_SFA = SFA.fit_transform(X_test)
        # Build RF on each lense
        RF_clf = RandomForestClassifier(n_estimators=n_trees, random_state=random_seed, n_jobs=n_jobs)
        RF_clf.fit(X_train_SFA, SMOTE_ytrain)
        if voting_method == 'validation': sfa_clfs.append(RF_clf)
        # Store prediction probability for test data
        model_pred = RF_clf.predict_proba(X_test_SFA)
        # accumulate RFmatrices for a lense
        RFmatrices.append(model_pred)

    length = len(RFmatrices)

    sax_clfs = []
    for n_coefs, n_bins in sax_lenses:
        X_sax = sax_transform(SMOTE_Xtrain, alphabet_size=n_bins, word_length=n_coefs)
        X_test_sax = sax_transform(X_test, alphabet_size=n_bins, word_length=n_coefs)
        # Build RF for each SAX lense
        RF_clf = RandomForestClassifier(n_estimators=n_trees, random_state=random_seed, n_jobs=n_jobs)
        RF_clf.fit(X_sax, SMOTE_ytrain)
        if voting_method == 'validation': sax_clfs.append(RF_clf)
        # Store prediction probability for test data
        model_pred = RF_clf.predict_proba(X_test_sax)
        # Sax only lenses
        # accumulate RFmatrices for all lenses
        RFmatrices.append(model_pred)

    RFmatrices = np.array(RFmatrices)
    
    # For use in RED CoMETS
    if return_mats:
        return RFmatrices, RF_clf.classes_, length
    
    else:    
        if voting_method == 'classic':
            acc = accuracy_score(y_test, classic_voting(RFmatrices, RF_clf.classes_, length))
            
        elif voting_method == 'uniform':
            acc = accuracy_score(y_test, sum_rule_uniform(RFmatrices, RF_clf.classes_))
            
        elif voting_method == 'meanmax':
            acc = accuracy_score(y_test, sum_rule_meanmax(RFmatrices, RF_clf.classes_))
            
        elif voting_method == 'validation':
            acc = accuracy_score(y_test, sum_rule_validation(SMOTE_Xtrain,SMOTE_ytrain,sax_lenses, sax_clfs, sfa_lenses, sfa_clfs, RFmatrices, RF_clf.classes_))
            
        return acc
