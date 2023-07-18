import sys
sys.path.append('..')
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.metrics import accuracy_score
from collections import Counter
from pyts.approximation import SymbolicFourierApproximation
from sklearn.ensemble import  RandomForestClassifier

from Univariate_Foundation.coeye_functions import searchLense_SFA, searchLense_SAX, classic_voting
from Utilities.sax import  sax_transform

def coeye(X_train, y_train, X_test, y_test, n_trees=100, random_seed=0, n_jobs=1):
    min_neighbours= min(Counter(y_train).items(), key=lambda k: k[1])[1]
    max_neighbours= max(Counter(y_train).items(), key=lambda k: k[1])[1]

    if(min_neighbours==max_neighbours):
            SMOTE_Xtrain= X_train
            SMOTE_ytrain= y_train
    else:
        if (min_neighbours>5): min_neighbours= 6
        try:
            SMOTE_Xtrain, SMOTE_ytrain = SMOTE(sampling_strategy="all", k_neighbors= min_neighbours-1, random_state=42).fit_resample(X_train, y_train)
        except ValueError:
            SMOTE_Xtrain, SMOTE_ytrain = RandomOverSampler(sampling_strategy='all',random_state=42).fit_resample(X_train,y_train)
      
    RFmatrices = []

    sfaPairs = searchLense_SFA(SMOTE_Xtrain, SMOTE_ytrain, n_jobs)
    for n_coefs, n_bins in sfaPairs:
        SFA = SymbolicFourierApproximation(strategy='uniform',n_coefs=n_coefs, n_bins=n_bins, alphabet='ordinal')
        # Transform to SFA
        X_train_SFA = SFA.fit_transform(SMOTE_Xtrain)
        X_test_SFA = SFA.fit_transform(X_test)
        # Build RF on each lense
        RF_clf = RandomForestClassifier(n_estimators=n_trees, random_state=random_seed, n_jobs=n_jobs)
        RF_clf.fit(X_train_SFA, SMOTE_ytrain)
        # Store prediction probability for test data
        model_pred = RF_clf.predict_proba(X_test_SFA)
        RFmatrices.append(model_pred)

    length = len(RFmatrices)
    
    saxPairs = searchLense_SAX(SMOTE_Xtrain, SMOTE_ytrain, n_jobs)
    for n_coefs,n_bins in saxPairs:

        X_sax = sax_transform(SMOTE_Xtrain, alphabet_size=n_bins, word_length=n_coefs)
        X_test_sax = sax_transform(X_test, alphabet_size=n_bins, word_length=n_coefs)
        # Build RF for each SAX lense
        RF_clf = RandomForestClassifier(n_estimators = n_trees, random_state=random_seed, n_jobs=n_jobs)
        RF_clf.fit(X_sax, SMOTE_ytrain)
        # Store prediction probability for test data
        model_pred = RF_clf.predict_proba(X_test_sax)
        RFmatrices.append(model_pred)
        
    acc = accuracy_score(y_test, classic_voting(RFmatrices, RF_clf.classes_, length))
    
    return acc


