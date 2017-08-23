# Kaggle-Zillow-Prize-xgboost-keras
My attempt for Kaggle Competition Zillow Prize using xgboost and keras
https://www.kaggle.com/c/zillow-prize-1

I have made the following attempts: (numeric value is Kaggle LB score, which is MAE on test dataset)
1. xgboost benchmark: ~0.0655     https://www.kaggle.com/anokas/simple-xgboost-starter-0-0655
2. My xgboost implementation with dummy variables and hyperparameter tuning: 0.0652418
3. Keras simple 2 hidden relu layer nn: 0.0649729
4. Keras simple autoencoder + 2 hidden relu layer nn: 0.0649136
5. Keras optimized autoencoder + 2 hidden relu layer nn: 0.0649310

Models 4 and 5 are picked for final submissions.

Possible to improve on:
1. Remove meaningless inputs such as censusid/countyid to avoid overfitting.
2. Spend more time in tuning autoencoder 
3. Use external data source such as census data (is this legit?)
