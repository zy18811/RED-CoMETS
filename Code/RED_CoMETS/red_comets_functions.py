import sys
sys.path.append('..')
import numpy as np

def glue(X):
    return np.hstack(X)

def glue_dimensions(X_train, X_test):
    return glue(X_train), glue(X_test)

