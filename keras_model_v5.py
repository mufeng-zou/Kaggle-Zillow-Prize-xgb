# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 10:28:32 2017

@author: mxz20
"""
from keras.models import load_model
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Input, BatchNormalization, Dense, Dropout, Activation, ActivityRegularization, concatenate
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import pandas as pd
import random
import test_submit
import numpy as np
import pickle

if __name__ == '__main__':

    random.seed(25252)
    
    #model 1 simple sequential nn
    df_merged2 = pd.read_pickle('./data/sample.pkl')
    
    sample_nn = df_merged2.fillna(-1)
    sample_nn.dtypes
    
    x_train = sample_nn.drop(['parcelid', 'logerror'], axis=1).values
    y_train = sample_nn['logerror'].values
    model = Sequential([
        BatchNormalization(input_shape=(59,)),
        Dropout(0.2),
        Dense(100, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(100, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(1, activation='tanh')
    ])
    
    model.compile(loss='mse', optimizer='sgd', metrics=['mae'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=100, min_delta=0.0001, mode='min')
    model_chk = ModelCheckpoint('./weights/best_model_nn.hdf5',save_best_only=True)
    #reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.001)
    model.fit(x_train, y_train, epochs=10000, validation_split=0.2, batch_size=7222, callbacks=[early_stopping, model_chk])
    best_model = load_model('./weights/best_model_nn.hdf5')
    best_model.evaluate(x_train,y_train)
    #0.068272271262240919
    
    #test submit
    test_submit.test_submit(best_model, sample_nn.drop(['parcelid', 'logerror'], axis=1).columns.values)
    #this gets score(mae) 0.0649729 on public LB, lower than xgboost benchmark
    
    
    
    
    
    #model 2: train autoencoder with the full property dataset then fit against outcome on subset
    df_prop = pd.read_pickle('./data/full_properties_2016.pkl').drop(['parcelid'], axis=1)
    df_prop.iloc[:1000,50:].describe()
    autoencoder_train = df_prop.fillna(-1).values
    
    #normalise and store parameters
    mean_x = [autoencoder_train[:,col].mean().astype(np.float32) for col in range(0,autoencoder_train.shape[1])]
    std_x = [autoencoder_train[:,col].std().astype(np.float32) for col in range(0,autoencoder_train.shape[1])]
    pickle.dump((mean_x,std_x),open('./data/mean_std_x.pkl','wb'))
    
    for i in range(0,autoencoder_train.shape[1]):
        autoencoder_train[:,i] = (autoencoder_train[:,i]-mean_x[i])/std_x[i]
        
    #subset output to important columns only, for reasonable loss
    important_cols = ['bathroomcnt', 'bedroomcnt', 'calculatedbathnbr', 'calculatedfinishedsquarefeet',
                      'finishedsquarefeet12', 'finishedsquarefeet50', 'fireplacecnt', 'fullbathcnt',
                      'garagecarcnt', 'garagetotalsqft', 'lotsizesquarefeet', #'propertylandusetypeid',
                      'roomcnt', 'threequarterbathnbr', 'yearbuilt', 'numberofstories', 'structuretaxvaluedollarcnt',
                      'taxvaluedollarcnt', 'landtaxvaluedollarcnt', 'taxamount']
    colno = [df_prop.columns.get_loc(x) for x in important_cols]
    autoencoder_train_out = autoencoder_train[:,colno]
    
    #autoencoder structure
    l_input = Input(shape=(55,))
    encoded = Dense(32, activation='relu')(l_input)
    encoded = Dense(10, activation='relu', name = 'encoded_layer')(encoded)
    decoded = Dense(15, activation='relu')(encoded)
    decoded = Dense(19, activation='tanh')(decoded)
    
    autoencoder = Model(inputs=l_input, outputs=decoded)
    autoencoder.compile(optimizer='rmsprop',loss='mse')
    early_stopping = EarlyStopping(monitor='val_loss', patience=30, min_delta=0.0001, mode='min')
    model_chk = ModelCheckpoint('./weights/best_autoencoder.hdf5',save_best_only=True)
    autoencoder.fit(autoencoder_train, autoencoder_train_out, epochs=10000, validation_split=0.2, batch_size=100000, callbacks=[early_stopping, model_chk])
    autoencoder.evaluate(autoencoder_train, autoencoder_train_out, batch_size=100000)
    #0.49614194566732595
    best_autoencoder = load_model('./weights/best_autoencoder.hdf5')
    best_autoencoder.evaluate(autoencoder_train, autoencoder_train_out, batch_size=100000)
    #0.49751613074507522
    
    #prepare regression data
    df_merged2 = pd.read_pickle('./data/sample.pkl')
    
    sample_nn = df_merged2.fillna(-1)
    sample_nn.dtypes

    (mean_x,std_x) = pickle.load(open( "./data/mean_std_x.pkl", "rb" ))
    
    x_train_sub = sample_nn.drop(['parcelid', 'logerror', 'transactiondate_month'], axis=1)
    x_train_sub['transactiondate_month'] = sample_nn['transactiondate_month']
    x_train_sub = x_train_sub.values
    for i in range(0,len(mean_x)):
        x_train_sub[:,i] = (x_train_sub[:,i]-mean_x[i])/std_x[i]
    y_train_sub = sample_nn['logerror'].values
    
    #now merge with outputs    
    best_autoencoder = load_model('./weights/best_autoencoder.hdf5')
    
    l_input2 = Input(shape=(56,))
    l_merged = concatenate([best_autoencoder.get_layer('encoded_layer').output, l_input2])
    l_dense1 = Dropout(0.2)(Dense(100, activation='relu')(l_merged))
    l_dense2 = Dropout(0.2)(Dense(100, activation='relu')(l_dense1))
    l_output = Dense(1, activation='tanh')(l_dense2)
    
    model = Model(inputs=[best_autoencoder.layers[0].input, l_input2], outputs=l_output)
    #fix weights for autoencoder part
    for layer in model.layers[:3]:
        layer.trainable = False

    model.compile(loss='mae', optimizer='rmsprop', metrics=['mse'])
    early_stopping2 = EarlyStopping(monitor='val_loss', patience=100, min_delta=0.0001, mode='min')
    model_chk2 = ModelCheckpoint('./weights/best_model_autoencoder_nn.hdf5',save_best_only=True)
    model.fit([x_train_sub[:,:-1], x_train_sub], y_train_sub, epochs=10000, 
              validation_split=0.2, batch_size=14444, callbacks=[early_stopping2, model_chk2])

    model.evaluate([x_train_sub[:,:-1], x_train_sub],y_train_sub,batch_size=10000)
    #0.066913471161304042
    best_model = load_model('./weights/best_model_autoencoder_nn.hdf5')
    best_model.evaluate([x_train_sub[:,:-1], x_train_sub],y_train_sub,batch_size=10000)
    #0.067341217576706386
    
    test_submit.test_submit_autoencoder_nn(best_model, sample_nn.drop(['parcelid', 'logerror'], axis=1).columns.values)
    
    
    
    
    
    
    #not used: functional with concatenate after each layer
    if False:
        l_input = Input(shape=(59,), dtype='float32')
        l_norm = BatchNormalization()(l_input)
        
        l_relu1 = Dense(100, activation='relu')(l_norm)
        l_norm1 = BatchNormalization()(l_relu1)
        l_concat1 = concatenate([l_norm, l_norm1])
        l_dropout1 = Dropout(0.5)(l_concat1)
        
        l_relu2 = Dense(100, activation='relu')(l_dropout1)
        l_norm2 = BatchNormalization()(l_relu2)
        l_concat2 = concatenate([l_norm, l_norm1, l_norm2])
        l_dropout2 = Dropout(0.5)(l_concat2)
        
        l_relu3 = Dense(100, activation='relu')(l_dropout2)
        l_norm3 = BatchNormalization()(l_relu3)
        l_concat3 = concatenate([l_norm, l_norm1, l_norm2, l_norm3])
        l_dropout3 = Dropout(0.5)(l_concat3)
        
        l_output = Dense(1, activation='tanh')(l_dropout3)
        
        model = Model(inputs=l_input, outputs=l_output)    
        