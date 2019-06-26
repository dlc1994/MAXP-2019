#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import sklearn
import sklearn.externals.joblib as joblib
import random


LoanStats3a=pd.read_csv('processed_test_datasetALL.csv')
X= LoanStats3a[['in_len', 'number', 'vout_num_origin', 'value_origin', 'out_len', 'out_value', 'out_number', 'is_coinbase']]

print(X.shape)

X = np.array(X)


# sc = joblib.load('standarsc')
# X_std = sc.transform(X)

# pca = joblib.load('geopca.m') #加载地形降维模型
# # 使用PCA对特征进行降维
# xGeoPCA = pca.transform(xGeoStd)

clf_best = joblib.load('RF_FIXED.m') #加载离线预测模型
print(clf_best.get_params())
y_predict = clf_best.predict(X)

df = LoanStats3a[['address']].copy()
df['label'] = y_predict
df.to_csv('submission_0524.csv',index=False)
print('done!')


# all prediction 'services'  3-17  84
# Random Forest_GridSearch_0.89-0421.m   4-21   89.42
# RF_10_50_8_0.88-0407.m  4-7   88.976
# XGBoost_GridSearch_0.89-421.m  4-21  88.8409
# Random Forest_GridSearch_0.8960.m   4-22   85.623
# Random Forest_GridSearch_0.8800.m  4-30  78.299
# Random Forest_GridSearch_0.9089.m  5-2  85.4487





