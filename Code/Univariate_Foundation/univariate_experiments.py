import sys
sys.path.append('..')

from Utilities.utils import Evaluator
from coeye import coeye
from univariate_foundation import univariate_foundation

UNIVARIATE_DATASETS = 'path/to/datasets/folder'

if __name__ == '__main__':
    
    """
    Evaluate Classic Co-eye
    """

    Evaluator(coeye, 'Classic Co-eye', [None], UNIVARIATE_DATASETS, n_samples=30, mv=False).evaluate('Classic_Coeye_TESTFOLDS.csv')

    """
    Evaluate Random Pair Selection:  R5%, R10%, R15%, and R20%
    """

    Evaluator(univariate_foundation, 'R5%',  [5, 'classic'], UNIVARIATE_DATASETS, n_samples=30, mv=False).evaluate('R5%_TESTFOLDS.csv')
    Evaluator(univariate_foundation, 'R10%', [10, 'classic'], UNIVARIATE_DATASETS, n_samples=30, mv=False).evaluate('R10%_TESTFOLDS.csv')
    Evaluator(univariate_foundation, 'R15%', [15, 'classic'], UNIVARIATE_DATASETS, n_samples=30, mv=False).evaluate('R15%_TESTFOLDS.csv')
    Evaluator(univariate_foundation, 'R20%', [20, 'classic'], UNIVARIATE_DATASETS, n_samples=30, mv=False).evaluate('R20%_TESTFOLDS.csv')
    
    """ 
    Evaluate Sum Rule Uniform, Mean-Max, and Validation voting methods with R5% Pair Selection
    """
 
    Evaluator(univariate_foundation, 'R5% SR Uniform',  [5, 'uniform'], UNIVARIATE_DATASETS, n_samples=30, mv=False).evaluate('R5%_SR_Uniform_TESTFOLDS.csv')
    Evaluator(univariate_foundation, 'R5% SR Mean-Max',  [5, 'meanmax'], UNIVARIATE_DATASETS, n_samples=30, mv=False).evaluate('R5%_SR_Mean-Max_TESTFOLDS.csv')
    Evaluator(univariate_foundation, 'R5% SR Validation',  [5, 'validation'], UNIVARIATE_DATASETS, n_samples=30, mv=False).evaluate('R5%_SR_Validation_TESTFOLDS.csv')
