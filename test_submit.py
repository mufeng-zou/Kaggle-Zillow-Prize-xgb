# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 15:37:57 2017

@author: mxz20
"""
import pandas as pd
import pickle
import gc

def test_submit(best_model, x_cols):
    df_sub = pd.read_csv('./downloads/sample_submission.csv')
    df_prop = pd.read_csv('./downloads/properties_2016.csv')
    df_sub['parcelid'] = df_sub['ParcelId']
    df_sub = df_sub.merge(df_prop, on='parcelid', how='left')   
    
    cols_categorical = df_sub.columns[df_sub.dtypes=='object']
    nu = df_sub[cols_categorical].apply(pd.Series.nunique, dropna=False)
    print(nu)
    cols_remove = cols_categorical[nu>=50]
    cols_dummy = cols_categorical[nu<50]
    
    del df_prop; gc.collect()
    
    df_sub = pd.get_dummies(df_sub.drop(cols_remove, axis=1), dummy_na=True, columns=cols_dummy)
    df_sub['transactiondate_month'] = 0
    cols_to_remove = [x for x in df_sub.columns.values if x not in x_cols]
    df_sub.drop(cols_to_remove, axis=1, inplace=True)
    gc.collect()
    df_sub=df_sub.fillna(-1)
    
    df_submit = pd.read_csv('./downloads/sample_submission.csv')
    
    for month in range(10,13):
        print('Predicting month',month)
        df_sub['transactiondate_month']=month
        x_test = df_sub.drop(['parcelid'], axis=1).values
        p_test = best_model.predict(x_test)
        df_submit['2016'+str(month)] = p_test
        df_submit['2017'+str(month)] = p_test

    print('Writing csv ...')
    df_submit.to_csv('keras_submit.csv', index=False, float_format='%.4f')
    
def test_submit_autoencoder_nn(best_model,x_cols):
    df_sub = pd.read_csv('./downloads/sample_submission.csv')
    df_prop = pd.read_csv('./downloads/properties_2016.csv')
    df_sub['parcelid'] = df_sub['ParcelId']
    df_sub = df_sub.merge(df_prop, on='parcelid', how='left')   
    
    cols_categorical = df_sub.columns[df_sub.dtypes=='object']
    nu = df_sub[cols_categorical].apply(pd.Series.nunique, dropna=False)
    print(nu)
    cols_remove = cols_categorical[nu>=50]
    cols_dummy = cols_categorical[nu<50]
    
    del df_prop; gc.collect()
    
    df_sub = pd.get_dummies(df_sub.drop(cols_remove, axis=1), dummy_na=True, columns=cols_dummy)
    df_sub['transactiondate_month'] = 0
    cols_to_remove = [x for x in df_sub.columns.values if x not in x_cols]
    df_sub.drop(cols_to_remove, axis=1, inplace=True)
    gc.collect()
    df_sub=df_sub.fillna(-1)
    
    df_submit = pd.read_csv('./downloads/sample_submission.csv')
    
    for month in range(10,13):
        print('Predicting month',month)
        df_sub['transactiondate_month']=month
        x_test = df_sub.values
        (mean_x,std_x) = pickle.load(open( "./data/mean_std_x.pkl", "rb" ))
        for i in range(0,len(mean_x)):
            x_test[:,i] = (x_test[:,i]-mean_x[i])/std_x[i]
        p_test = best_model.predict([x_test[:,:-1],x_test])
        df_submit['2016'+str(month)] = p_test
        df_submit['2017'+str(month)] = p_test

    print('Writing csv ...')
    df_submit.to_csv('keras_submit_autoencoder_nn.csv', index=False, float_format='%.4f')
