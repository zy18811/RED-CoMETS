# RED-CoMETS
***RED CoMETS: an ensemble classifier for symbolically represented multivariate time series***

[arXiv:2307.13679](https://arxiv.org/abs/2307.13679) (preprint)
> <div align="justify">Multivariate time series classification is a rapidly growing area of research that has numerous practical applications in diverse fields such as finance, healthcare, and engineering. The problem of classifying multivariate time series data is challenging due to the high dimensionality, temporal dependencies, and varying lengths of the time series.  This paper proposes a novel ensemble classifier called RED CoMETS. The proposed method builds upon the success of Co-eye, an ensemble classifier for symbolically represented univariate time series, and extends it to handle multivariate data. RED CoMETS is evaluated on benchmark datasets from the UCR archive and achieves competitive accuracy with state-of-the-art techniques in multivariate settings and yields the highest accuracy reported in the literature for the 'HandMovementDirection' dataset. The proposed method also offers a significant reduction in computation time compared to classic Co-eye, making it an efficient and effective option for multivariate time series classification.</div>

Accepted by the [8th Workshop on Advanced Analytics and Learning on Temporal Data](https://ecml-aaltd.github.io/aaltd2023/) at ECML PKDD 2023.

## Results
Test accuracy is given for all experiments.   
Timing results have only been recorded for Co-eye, R5%, R10%, R15%, and R20%.

### Univariate Foundation
#### Co-eye
* [Co-eye](Results/Univariate_Foundation/Coeye_TESTFOLDS.csv)
#### Random Pair Selection
* [R5%](Results/Univariate_Foundation/R5_TESTFOLDS.csv)
* [R10%](Results/Univariate_Foundation/R10_TESTFOLDS.csv)
* [R15%](Results/Univariate_Foundation/R15_TESTFOLDS.csv)
* [R20%](Results/Univariate_Foundation/R20_TESTFOLDS.csv)

#### Voting Methods
* [Sum Rule Uniform](Results/Univariate_Foundation/R5_SR_Uniform_TESTFOLDS.csv)
* [Sum Rule Mean-Max](Results/Univariate_Foundation/R5_SR_Mean-Max_TESTFOLDS.csv)
* [Sum Rule Validation](Results/Univariate_Foundation/R5_SR_Validation_TESTFOLDS.csv)

### RED CoMETS
* [RED CoMETS-1](Results/RED_CoMETS/RED_CoMETS-1_TESTFOLDS.csv)
* [RED CoMETS-2](Results/RED_CoMETS/RED_CoMETS-2_TESTFOLDS.csv)
* [RED CoMETS-3](Results/RED_CoMETS/RED_CoMETS-3_TESTFOLDS.csv)
* [RED CoMETS-4](Results/RED_CoMETS/RED_CoMETS-4_TESTFOLDS.csv)
* [RED CoMETS-5](Results/RED_CoMETS/RED_CoMETS-5_TESTFOLDS.csv)
* [RED CoMETS-6](Results/RED_CoMETS/RED_CoMETS-6_TESTFOLDS.csv)
* [RED CoMETS-7](Results/RED_CoMETS/RED_CoMETS-7_TESTFOLDS.csv)
* [RED CoMETS-8](Results/RED_CoMETS/RED_CoMETS-8_TESTFOLDS.csv)
* [RED CoMETS-9](Results/RED_CoMETS/RED_CoMETS-9_TESTFOLDS.csv)

## Code
### Requirements
* Python < 3.11
* [Python Modules](requirements.txt)

### Loading Datasets

[UCR time series classification archive datasets](https://www.timeseriesclassification.com/dataset.php) in .arff format. 

```python
from Code.Utilities.utils import DatasetLoader

dataset = '/path/to/dataset/folder'

# Univariate Datasets
X_train, y_train, X_test, y_test = DatasetLoader(dataset).load_uv_dataset_to_numpy()

# Multivariate Datasets
X_train, y_train, X_test, y_test = DatasetLoader(dataset).load_mv_dataset_to_numpy()
```

### Co-eye

```python
from Code.Univariate_Foundation.coeye import coeye

[...] # load dataset

"""
Args:
    p_length (float): percentage of timeseries length used to determine number of SAX and SFA lenses
    voting_method  (string):  {'uniform', 'meanmax', 'validation'}
"""
accuracy = coeye(X_train, y_train, X_test, y_test)
```

### Univariate Foundation

```python
from Code.Univariate_Foundation.univariate_foundation import univariate_foundation

[...] # load dataset

"""
Args:
    p_length (float): percentage of timeseries length used to determine number of SAX and SFA lenses
    voting_method  (string):  {'uniform', 'meanmax', 'validation'}
"""
accuracy = univariate_foundation(X_train, y_train, X_test, y_test, p_length, voting_method)
```

### RED CoMETS

| Name         | Approach  | Sub-Approach | Voting Method 1 | Voting Method 2 |
|--------------|------------|----------|:---------------:|-----------------|
| RED CoMETS-1 | Concatenating | n/a      |     Uniform     | n/a             |
| RED CoMETS-2 | Concatenating | n/a      |     Mean-Max    | n/a             |
| RED CoMETS-3 | Concatenating | n/a      |    Validation   | n/a             |
| RED CoMETS-4 | Ensembling    | 1        |     Uniform     | n/a             |
| RED CoMETS-5 | Ensembling    | 1        |     Mean-Max    | n/a             |
| RED CoMETS-6 | Ensembling    | 2        |     Uniform     | Uniform         |
| RED CoMETS-7 | Ensembling    | 2        |     Uniform     | Mean-Max        |
| RED CoMETS-8 | Ensembling    | 2        |     Mean-Max    | Mean-Max        |
| RED CoMETS-9 | Ensembling    | 2        |     Mean-Max    | Uniform         |

```python
from Code.RED_CoMETS.red_comets import red_comets

[...] # load dataset

"""
Args:
  id_number (int): RED CoMETS-<ID>
"""
accuracy = red_comets(X_train, y_train, X_test, y_test, id_number)
```

## Reproducing the Experiments
We used [85 univariate](Results/Univariate_Foundation/uv_datasets.txt) and [26 multivariate](Results/RED_CoMETS/mv_datasets.txt) datasets from the UCR timeseries classification archive. 

### Co-eye and Univariate Foundation
[univariate_experiments.py](Code/Univariate_Foundation/univariate_experiments.py)

### RED CoMETS
[multivariate_experiments.py](Code/RED_CoMETS/multivariate_experiments.py)

## Acknowledgements
We thank all the people who have contributed to and who maintain the UCR time series classification archive. Critical difference diagrams in our paper showing were produced using code from [Ismail Fawaz et al. (2019)](https://github.com/hfawaz/cd-diagram). Our implementation of the SAX transform uses code modified from [saxpy](https://github.com/seninp/saxpy).
