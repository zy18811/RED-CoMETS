"""
Utilities
"""
import numpy as np
import pandas as pd
import os
from scipy.io.arff import loadarff
import glob
from tqdm import tqdm
import time
import gc
from pathos.multiprocessing import Pool, cpu_count

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from numba import NumbaWarning
warnings.filterwarnings("ignore", category=NumbaWarning)


def load_arff_as_df(file):
    data = loadarff(file)
    df = pd.DataFrame(data[0])
    return df

def separate_dimensions_arff(file):
    df = load_arff_as_df(file)

    data = df[df.columns[:-1]].to_numpy()[:, 0]
    data = np.array([np.array([np.array(list(d_row)) for d_row in row]) for row in data])
    n_dimensions = data.shape[1]

    mux = pd.MultiIndex.from_product([[f'Dimension {i}' for i in range(n_dimensions)], range(data.shape[2])])
    dimension_df = pd.DataFrame(np.hstack([data[:, i, :] for i in range(n_dimensions)]), index=df.index, columns=mux)

    dimension_df['Class'] = df[df.columns[-1]].apply(lambda s: s.decode('utf-8'))
    return dimension_df

def mv_df_to_np(df):
        n_dims, ts_length = df.columns.levshape
        
        # extra 1 due to class column
        n_dims -= 1    
        ts_length -=1 
        n_samples = df.shape[0]
        
        X = np.empty((n_dims, n_samples, ts_length))
        
        for i in range(n_dims):
            X[i] = df[f'Dimension {i}'].values
            
        return X
    
class DatasetLoader():
    def __init__ (self, path_to_folder):
        self.path = path_to_folder
        
        self.folder_name =  os.path.split(self.path)[1]
        
        self.train_path = os.path.join(self.path, self.folder_name + '_TRAIN.arff')
        self.test_path = os.path.join(self.path, self.folder_name + '_TEST.arff')
        
    def load_uv_dataset(self):
        train_df = load_arff_as_df(self.train_path).rename(columns = {'target':'Class'})
        train_df['Class'] = train_df['Class'].apply(lambda s: s.decode('utf-8'))
        
        test_df = load_arff_as_df(self.test_path).rename(columns = {'target':'Class'})
        test_df['Class'] = test_df['Class'].apply(lambda s: s.decode('utf-8'))
        
        return train_df, test_df
        
    def load_mv_dataset(self):
        train_df = separate_dimensions_arff(self.train_path)
        test_df = separate_dimensions_arff(self.test_path)
        
        return train_df, test_df
    
    def strat_samples(self, n_samples, load_func):
        sample_dict = {}

        train_df, test_df = load_func()
        N = train_df.index.values.size

        sample_dict[0] = (train_df, test_df)
        dataset_df = train_df.append(test_df, ignore_index=True)
        for n in range(1,n_samples):
            train_sample = dataset_df.groupby('Class', group_keys=False).apply(
                lambda x: x.sample(int(np.rint(N * len(x) / len(dataset_df))), random_state=n)).sample(frac=1,
                                                                                                    random_state=n)
            test_sample = dataset_df.drop(train_sample.index).reset_index(drop = True)
            train_sample = train_sample.reset_index(drop=True)
            sample_dict[n] = (train_sample, test_sample)
        return sample_dict 
        
    def uv_strat_samples(self, n_samples):
        return self.strat_samples(n_samples, self.load_uv_dataset)
    
    def mv_strat_samples(self, n_samples): 
        return self.strat_samples(n_samples, self.load_mv_dataset)
    
    def load_uv_dataset_to_numpy(self):
        train_df, test_df = self.load_uv_dataset()
        y_train = train_df['Class'].values.ravel()
        y_test = test_df['Class'].values.ravel()

        X_train = train_df.drop(['Class'], axis=1).values
        X_test = test_df.drop(['Class'], axis=1).values
        
        return X_train, y_train, X_test, y_test
        
    def load_mv_dataset_to_numpy(self):
        train_df, test_df  = self.load_mv_dataset()
        y_train = train_df['Class'].values.ravel()
        y_test = test_df['Class'].values.ravel()

        X_train = mv_df_to_np(train_df)
        X_test = mv_df_to_np(test_df)
        
        return X_train, y_train, X_test, y_test
        
class Evaluator():
    def __init__(self, clf, clf_name, clf_args, datasets_folder, n_samples, mv):
        self.clf = clf
        self.clf_name = clf_name
        self.clf_args = clf_args
        
        glob_folder = datasets_folder + '\*\\'
        self.datasets = [g.strip('\\') for g in glob.glob(glob_folder)]
    
        self.n_samples = n_samples
        self.mv = mv
        
        mux_ix = pd.MultiIndex.from_product([[self.clf_name], [*map(os.path.basename, self.datasets)]])
        mux_cols = pd.MultiIndex.from_product([['Accuracy', 'Run Time'], range(self.n_samples)])

        self.df = pd.DataFrame(index=mux_ix, columns=mux_cols)
    
    
    def acc_and_time(self, train_df, test_df):
        y_train = train_df['Class'].values.ravel()
        y_test = test_df['Class'].values.ravel()

        if not self.mv:
            X_train = train_df.drop(['Class'], axis=1).values
            X_test = test_df.drop(['Class'], axis=1).values
        
        else:
            X_train = mv_df_to_np(train_df)
            X_test = mv_df_to_np(test_df)
            
        start = time.time()
    
        acc = self.clf(X_train, y_train, X_test, y_test, *self.clf_args)
        
        end = time.time()
        
        run_time = end-start
        return acc, run_time

    def eval_sample(self, sample_number, samples_dict):
        train_df, test_df  =  samples_dict[sample_number]
        
        acc, t = self.acc_and_time(train_df, test_df)
        
        # Helps with memory
        del train_df
        del test_df
        gc.collect()
        
        return (acc,t)
    
    def eval_wrapper(self,args):
        return self.eval_sample(*args)
    
    def evaluate(self, save):
        with tqdm(self.datasets) as pbar:
            for arff in pbar:
                loader = DatasetLoader(arff)
                
                data_name = os.path.basename(arff)
                pbar.set_description(data_name)
                
                if self.mv:
                    strat_samples_dict = loader.mv_strat_samples(self.n_samples)
                else:
                    strat_samples_dict = loader.uv_strat_samples(self.n_samples)
                    
                args = [(i, strat_samples_dict) for i in range(self.n_samples)]
                
                threads = cpu_count()-1
                with Pool(threads) as p:
                    res = np.array(list(tqdm(p.imap(self.eval_wrapper, args), total=len(args), desc='Sample', leave=False)))
                
                self.df.loc[self.clf_name, data_name].loc['Accuracy'] = res[:, 0]
                self.df.loc[self.clf_name, data_name].loc['Run Time'] = res[:, 1]
                
        self.df.droplevel(0).to_csv(save)  


