import sys
sys.path.append('..')
import random
from collections import Counter
import numpy as np
from pyts.approximation import SymbolicFourierApproximation
from Utilities.sax  import  sax_transform
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import  RandomForestClassifier

"""
Pair Selection
"""

def searchLense_SFA(X_train, y_train, n_jobs):

    # Input is training data (X_train, y_train)
    # Returns selected pairs for SFA transformation based on cross validation

    # Set ranges (Seg, alpha) parameters
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

    pairs = []

    # Learning parameteres using 5 folds cross validation
    for alpha in alphas:
        s = []
        for seg in n_segments:
            SFA = SymbolicFourierApproximation(strategy='uniform',n_coefs=seg, n_bins=alpha, alphabet='ordinal')
            X_SFA = SFA.fit_transform(X_train)
            scores = 0
            RF_clf = RandomForestClassifier(n_estimators=100, random_state=0,n_jobs=n_jobs)
            cv = np.min([5,len(y_train)//len(list(set(y_train)))])
            scores = cross_val_score(RF_clf, X_SFA, y_train, cv=cv, n_jobs=n_jobs)
            s.append(scores.mean())
        winner = np.argwhere(s >= np.amax(s) - 0.01)
        for i in winner.flatten().tolist():
            bestCof = n_segments[i]
            pairs.append((bestCof, alpha))

    return pairs


def searchLense_SAX(X_train, y_train, n_jobs):
    # Input is training data (X_train, y_train)
    # Returns selected pairs for SAX transformation based on cross validation

    # Set range (alpha) parameter
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

    pairs = []

    # Learning parameteres using 5 folds cross validation

    for alpha in alphas:
        s = []
        for seg in n_segments:
            X_SAX = sax_transform(X_train,seg,alpha)
            scores = 0
            RF_clf = RandomForestClassifier(n_estimators=100, random_state=0,n_jobs=n_jobs)
            cv = np.min([5, len(y_train) // len(list(set(y_train)))])
            scores = cross_val_score(RF_clf, X_SAX, y_train, cv=cv, n_jobs=n_jobs)
            s.append(scores.mean())
        winner = np.argwhere(s >= np.amax(s) - 0.01)
        for i in winner.flatten().tolist():
            bestCof = n_segments[i]
            pairs.append((bestCof, alpha))
    
    return pairs

"""
Voting
"""

def dynamic_voting(matrices, labels, sfa_n):
    # Marerices is of length L: number of lenses/forests (total for SAX and SFA)
    # Each matrix in matrices is the probalistic accuracy of test data
    # Matrix of size m X n: where m is the number of test instances,n is the number of classes (labels)
    # Sax_n number of lenses in SFA

    predLabel = []
    multiSFA = False
    multiSAX = False

    # For each test data, we define the most confident and second-most confident for each presentation, then vote between them
    for row in range(len(matrices[0])):
        # Initialise
        maxConf_SFA = 0
        SFAConf = []
        SFALables = []
        SAXConf = []
        SAXLables = []
        # Look into SFA only matrices
        # Find the max confidence/ prbability in SFA
        for mat in matrices[:sfa_n]:
            for col in range(len(labels)):
                SFAConf.append(mat[row][col])
                SFALables.append(labels[col])
                if (mat[row][col] > maxConf_SFA):
                    maxConf_SFA = mat[row][col]
                    Conflabel_SFA = labels[col]

        # Second best flag
        SB1 = False
        # Special case: when multiple lenses have the same accuracy (most confident)
        if (SFAConf.count(maxConf_SFA) > 1):
            indices = [i for i, x in enumerate(SFAConf) if x == maxConf_SFA]
            l = [SFALables[i] for i in indices]
            if (len(set(l)) != 1):
                maxIr = max(l.count(y) for y in set(l))
                k = [l.count(y) for y in set(l)]

                # Find the most common label
                if (k.count(maxIr) == 1):
                    cnt = Counter(l)
                    Conflabel_SFA = cnt.most_common(2)[0][0]
                    secondBestSFALabel = cnt.most_common(2)[1][0]

                    # If no common label, in case of tie, the label is chosen randomly.
                else:
                    shuff = []
                    for it in set(l):
                        if (l.count(it) == maxIr): shuff.append(it)
                    shuff = list(set(l))
                    random.shuffle(shuff)
                    Conflabel_SFA = shuff[0]
                    secondBestSFALabel = shuff[1]
                SB1 = True
                # Set the second best
                secondBestSFA = maxConf_SFA

        if (SB1 == False):
            secondBestSFA = max(n for n in SFAConf if n != maxConf_SFA)
            secondBestSFALabel = SFALables[SFAConf.index(secondBestSFA)]

        # Same steps for SAX
        maxConf_SAX = 0
        SB2 = False
        for mat in matrices[sfa_n:]:
            for col in range(len(labels)):
                SAXConf.append(mat[row][col])
                SAXLables.append(labels[col])
                if (mat[row][col] > maxConf_SAX):
                    maxConf_SAX = mat[row][col]
                    Conflabel_SAX = labels[col]
        if (SAXConf.count(maxConf_SAX) > 1):
            indices = [i for i, x in enumerate(SAXConf) if x == maxConf_SAX]
            l = [SAXLables[i] for i in indices]
            if (len(set(l)) != 1):
                #                 print ("Conflict on max confident", l)
                maxIr = max(l.count(y) for y in set(l))
                k = [l.count(y) for y in set(l)]
                #                 How many equal items with max confident value
                if (k.count(maxIr) == 1):
                    cnt = Counter(l)
                    Conflabel_SAX = cnt.most_common(2)[0][0]
                    secondBestSAXLabel = cnt.most_common(2)[1][0]
                else:
                    shuff = []
                    for it in set(l):
                        if (l.count(it) == maxIr): shuff.append(it)
                    #                     print ("tie", shuff)
                    random.shuffle(shuff)
                    Conflabel_SAX = shuff[0]
                    secondBestSAXLabel = shuff[1]

                secondBestSAX = maxConf_SAX
                #                 print("most common", Conflabel_SAX, "Second Best", secondBestSAXLabel)
                SB2 = True
        #             print("indecies", indices, "labels" ,labels)
        if (SB2 == False):
            secondBestSAX = max(n for n in SAXConf if n != maxConf_SAX)
            secondBestSAXLabel = SAXLables[SAXConf.index(secondBestSAX)]

        #         print ("Best SAX",maxConf_SAX , Conflabel_SAX)
        #         print ("Second Best SAX",secondBestSAX , secondBestSAXLabel)

        # -----------------------------------
        # In case of agreement between most Conf SAX and SFA
        if (Conflabel_SAX == Conflabel_SFA):
            best = Conflabel_SAX
        # If no agreement, then second best is testes
        elif (secondBestSAX > secondBestSFA):
            best = secondBestSAXLabel
        else:
            best = secondBestSFALabel
        # Accumulate labels with the best choice
        predLabel.append(best)
    #         print("Best SFA, SAX", Conflabel_SAX, Conflabel_SFA,"Second best SFA, SAX", secondBestSAXLabel, secondBestSFALabel,"Best",  best)
    #         print ("----------------------------------------")

    return predLabel

