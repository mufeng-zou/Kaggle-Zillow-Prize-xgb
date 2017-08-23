import pandas as pd
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
    df_merged['transactiondate_month'] = pd.to_numeric(df2['transactiondate'].str[5:7])

    cols_categorical = df_merged.columns[df_merged.dtypes=='object']
    nu = df_merged[cols_categorical].apply(pd.Series.nunique, dropna=False)
    print(nu)
    cols_remove = cols_categorical[nu>=50]
    cols_dummy = cols_categorical[nu<50]
    
    df_merged2 = pd.get_dummies(df_merged.drop(cols_remove, axis=1), columns=cols_dummy)
    df_merged2.head()
    df_merged2.info()
    df_merged2.columns
    df_merged2['logerror'].head()
    
    df_merged2 = df_merged2.sample(frac=1).reset_index(drop=True)
    
    df_merged2.to_pickle('./data/sample.pkl')
    
    #full property sample for autoencoder
    file1 = './downloads/properties_2016.csv'
    df1 = pd.read_csv(file1)
    cols_categorical = df1.columns[df1.dtypes=='object']
    nu = df1[cols_categorical].apply(pd.Series.nunique, dropna=False)
    print(nu)
    cols_remove = cols_categorical[nu>=50]
    cols_dummy = cols_categorical[nu<50]
    df1 = pd.get_dummies(df1.drop(cols_remove, axis=1), columns=cols_dummy)
    df1.to_pickle('./data/full_properties_2016.pkl')