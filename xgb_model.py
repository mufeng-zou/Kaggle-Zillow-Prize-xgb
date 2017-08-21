# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 15:20:35 2017

@author: mxz20
"""

import xgboost as xgb
import pandas as pd
import random
import math
import re
#from sklearn.model_selection import GridSearchCV
#regressor sklearn interface does not support earlystopping

if __name__ == '__main__':

    random.seed(25252)
    
    df_merged2 = pd.read_pickle('sample.pkl')
    
    split = math.ceil(0.8*len(df_merged2))
    X = df_merged2.loc[:,df_merged2.columns!='logerror']
    Y = df_merged2['logerror']
    Xtrain, Xvalid, Ytrain, Yvalid = X[:split], X[split:], Y[:split], Y[split:]
    dtrain = xgb.DMatrix(Xtrain, label=Ytrain)
    dvalid = xgb.DMatrix(Xvalid, label=Yvalid)
    
    #quick grid search for max_depth
    best_mae = 999
    best_params = {}
    best_model = None
    for depthi in range(1,11):
        print('Fitting model depth',depthi)
        params = {
            'objective': 'reg:linear',
            'eval_metric': 'mae',
            'max_depth': depthi,
            'eta': 0.02,
            'silent': 1
        }
        clf = xgb.train(params=params,
                        dtrain=dtrain,
                        num_boost_round=10000,
                        evals=[(dtrain, 'train'), (dvalid, 'valid')],
                        early_stopping_rounds=200,
                        verbose_eval=10)
        clf.get_fscore()
        clf.get_score()
        mae = float(re.sub(r'(.*:)', '', re.sub("'","",str(clf.eval(dvalid)))))
        if mae<best_mae:
            best_mae=mae
            best_params=params
            best_model = clf
    