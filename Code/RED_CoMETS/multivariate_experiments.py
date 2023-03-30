import sys
sys.path.append('..')

from Utilities.utils import Evaluator
from red_comets import red_comets

MULTIVARIATE_DATASETS = 'path/to/datasets/folder'

if __name__ == '__main__':
    
    """
    Evaluate RED CoMETS-<1-9>
    """

    for i in range(9):
        Evaluator(red_comets, f'RED CoMETS-{i+1}', [i+1], MULTIVARIATE_DATASETS, n_samples=30, mv=True).evaluate(f'RED_CoMETS-{i+1}_TESTFOLDS.csv')
