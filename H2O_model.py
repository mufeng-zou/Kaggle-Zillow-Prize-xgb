# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 21:19:16 2017

@author: MZ
"""

import h2o
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.grid.grid_search import H2OGridSearch
import pandas as pd
import math

if __name__ == '__main__':

    file1 = './downloads/properties_2016.csv'
    df1 = pd.read_csv(file1)
    df1.head()
    df1.info()
    file2 = './downloads/train_2016_v2.csv'
    df2 = pd.read_csv(file2)
    df2.head()
    df2.info()
    
    df_merged = df1.merge(df2, how='inner', on='parcelid')
    df_merged.head()
    df_merged.info()
    df_merged['transactiondate'].head()
    df_merged['transactiondate_month'] = df2['transactiondate'].str[5:7]

    cols_categorical = df_merged.columns[df_merged.dtypes=='object']
    nu = df_merged[cols_categorical].apply(pd.Series.nunique, dropna=False)
    cols_remove = cols_categorical[nu>=200]
    cols_categorical = cols_categorical[nu<200]
    
    df_merged2 = pd.get_dummies(df_merged.drop(cols_remove, axis=1), dummy_na=True, columns=cols_categorical)
    df_merged2.head()
    df_merged2.info()
    df_merged2.columns
    df_merged2['logerror'].head()

    #shuffle for random train/valid split
    df_merged2 = df_merged2.sample(frac=1).reset_index(drop=True)
    
    #h2o gbm
    h2o.init()

    split = math.ceil(0.8 * len(df_merged2))
    train = h2o.H2OFrame(df_merged2)
    #no validation because i use cv, no test because this is quick model
    
    X = list(df_merged2.columns[df_merged2.columns!='logerror'])
    Y = 'logerror'
    
    model = H2OGradientBoostingEstimator(
            ntrees=10000, #annealing learning rate and early stopping
            learn_rate=0.05,
            learn_rate_annealing = 0.99,
            col_sample_rate = 0.8,
            seed = 1234,
            score_tree_interval = 10,
            nfolds = 3,
            stopping_rounds = 5,
            stopping_metric = "mae",
            stopping_tolerance = 1e-4)
    hyper_parameters = {"max_depth": list(range(1,11))}
    search_criteria = {"strategy":"Cartesian"}

    grid = H2OGridSearch(model,
                         hyper_params=hyper_parameters,
                         search_criteria=search_criteria)
    grid.train(x=X,
               y=Y,
               training_frame=train)
    sorted_grid = grid.get_grid(sort_by='mae')
    print(sorted_grid)

    #h2o.shutdown()