# Kaggle-Zillow-Prize-xgboost-H2O-keras
My attempt for Kaggle Competition Zillow Prize using xgboost, H2O and keras https://www.kaggle.com/c/zillow-prize-1


I have made the following attempts: (numeric value is Kaggle LB score, which is MAE on test dataset)

~0.0655 - xgboost benchmark on Kaggle Kernel https://www.kaggle.com/anokas/simple-xgboost-starter-0-0655

0.0652418 - My xgboost implementation with dummy variables and hyperparameter tuning

similar as above - H2O implementation gives very similar results as xgboost (was trying mainly for grid search with more parameters).

0.0649729 - Keras simple 2 hidden relu layer neural network

0.0649136 - Keras simple autoencoder + 2 hidden relu layer neural network

0.0649310 - Keras optimized autoencoder + 2 hidden relu layer neural network



The last 2 models are picked for final submissions.


Possible to improve on:
1. Remove meaningless inputs such as censusid/countyid to avoid overfitting.
2. Spend more time in tuning autoencoder 
3. Use external data source such as census data (is this legit?)

oh wait... Why don't I do a ensemble model at the end?
