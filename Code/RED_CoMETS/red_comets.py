import sys
sys.path.append('..')
import numpy as np
from sklearn.metrics import accuracy_score

from red_comets_functions import glue_dimensions
from Univariate_Foundation.univariate_foundation import univariate_foundation, sum_rule_uniform, sum_rule_meanmax

def red_comets(X_train, y_train, X_test, y_test, id_number, p_length=5, n_trees=100, random_seed=42, n_jobs=1):
    
    voting_methods_lookup = ['uniform', 'meanmax', 'validation', ('uniform',  None), ('meanmax', None), ('uniform', 'uniform'), ('uniform',  'meanmax'), ('meanmax', 'uniform'), ('meanmax','meanmax')]
    
    n_dims, test_size, _ =  X_test.shape
    n_classes = len(np.unique(y_train))
        
    # Glueing Dimensions
    if id_number in [1,2,3]:
        voting_method = voting_methods_lookup[id_number-1]
        p_length /= n_dims # number of lenses computed from the length of the time series prior to glueing
        X_train, X_test = glue_dimensions(X_train, X_test)
        return univariate_foundation(X_train, y_train, X_test, y_test,  p_length, voting_method, n_trees, random_seed, n_jobs)
    
    # Ensembling Dimensions
    elif id_number in [4,5,6,7,8,9]:       
        
        # Approach 1 or Approach 2
        if id_number in [4,5]:
            approach = 1
            sax_mats_all = None
            sfa_mats_all = None
            
        elif id_number in [6,7,8,9]:
            approach = 2
            rf_mats_fused = np.zeros(shape=(n_dims, test_size, n_classes))
        
        voting_method_one, voting_method_two = voting_methods_lookup[id_number-1]
                
        for d in range(n_dims):
            X_train_d = X_train[d]
            X_test_d = X_test[d]
            
            rf_mats, classes, length = univariate_foundation(X_train_d, y_train, X_test_d, y_test, p_length, None, n_trees, random_seed, n_jobs, return_mats=True)
            
            if approach == 1:
                sfa_mats = rf_mats[:length]
                sax_mats = rf_mats[length:]
                
                if sfa_mats_all is None:
                    sfa_mats_all = sfa_mats
                else:
                    sfa_mats_all = np.concatenate((sfa_mats_all, sfa_mats))

                if sax_mats_all is None:
                    sax_mats_all = sax_mats
                else:
                    sax_mats_all = np.concatenate((sax_mats_all, sax_mats))
            
            elif approach == 2:
                if voting_method_one == 'uniform':
                    rf_mats_fused[d] = np.sum(rf_mats, axis=0)
                elif voting_method_one == 'meanmax':
                    rf_mats_fused[d] = sum_rule_meanmax(rf_mats, classes, True)
                
        rf_mats_all = np.concatenate((sfa_mats_all, sax_mats_all))
        
        #  Approach 1
        if approach == 1:
            if voting_method_one == 'uniform':
                acc = accuracy_score(y_test, sum_rule_uniform(rf_mats_all, classes))
            
            elif voting_method_one == 'meanmax':
                acc = accuracy_score(y_test, sum_rule_meanmax(rf_mats_all, classes))
                    
        # Approach 2
        elif approach == 2:
            if voting_method_two == 'uniform':
                acc = accuracy_score(y_test, sum_rule_uniform(rf_mats_fused, classes))
            elif voting_method_two == 'meanmax':
                acc = accuracy_score(y_test, sum_rule_meanmax(rf_mats_fused, classes))
            
    return acc
        
