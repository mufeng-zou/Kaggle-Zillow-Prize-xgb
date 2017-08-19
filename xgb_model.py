# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 15:20:35 2017

@author: mxz20
"""

import xgboost as xgb
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
import random

if __name__ == '__main__':

    random.seed(25252)

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
    
    X = df_merged2.loc[:,df_merged2.columns!='logerror']
    Y = df_merged2['logerror']
    # CV model

    model = xgb.XGBRegressor()
    params = {
        'n_estimators': [1000],
        'objective': ['reg:linear'],
        'max_depth': list(range(3,11)),
        'learning_rate': [0.01, 0.1],
        'subsample': [0.9, 1.0],
        'colsample_bytree': [0.5, 0.9, 1.0]
    }
    clf = RandomizedSearchCV(model, params, cv=3, n_iter=20, verbose=1, scoring='neg_mean_absolute_error')
    fit_params = {
        'verbose': 1
#        'eval_metric': 'mae',
#        'early_stopping_rounds': 10
    }
    clf.fit(X, Y, None, **fit_params)
