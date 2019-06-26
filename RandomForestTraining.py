# encoding: utf-8
'''
@author: Lingcheng Dai
@contact: 2013210288@bupt.edu.cn
@file: RandomForestTraining.py
@time: 2019/5/21 18:08
'''
import pandas as pd
import sklearn.externals.joblib as joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import time

#读取数据
# x_preliminary=pd.read_csv('preliminary_train_dataset.csv')
# x_rematch = pd.read_csv('rematch_test_datasetALL.csv')
# # y_rematch = pd.read_csv('rematch_test_submission.csv')
# # x_rematch['label'] = y_rematch['label']
# # x_rematch.to_csv('rematch_test_datasetALL.csv', index=False)
# LoanStats3a = pd.concat([x_preliminary, x_rematch])
#
# x_services = LoanStats3a[LoanStats3a['label']=='services']
# x_exchanges = LoanStats3a[LoanStats3a['label']=='exchanges']
# x_exchanges_copy = x_exchanges.copy()
# for i in range(8):
#     x_exchanges = pd.concat([x_exchanges, x_exchanges_copy])
# x_gamblings = LoanStats3a[LoanStats3a['label']=='gambling']
# x_gamblings_copy = x_gamblings.copy()
# for j in range(120):
#     x_gamblings = pd.concat([x_gamblings, x_gamblings_copy])
# print(x_services.shape, x_exchanges.shape, x_gamblings.shape)
#
# # LoanStats3a = pd.concat([x_services[:100000], x_exchanges[:100000], x_gamblings[:100000]])
# LoanStats3a = pd.concat([x_services, x_exchanges, x_gamblings])
# LoanStats3a.to_csv('BIGDATASET.csv')
LoanStats3a = pd.read_csv('BIGDATASET.csv')
LoanStats3a = LoanStats3a.sample(frac=0.2).reset_index(drop=True)
print(LoanStats3a.shape)

X, y = LoanStats3a[['in_len', 'number', 'vout_num_origin', 'value_origin', 'out_len',
                    'out_value', 'out_number', 'is_coinbase']], LoanStats3a[['label']]
# x_rematch, y_rematch = x_rematch[['in_len', 'number', 'vout_num_origin', 'value_origin',
#                                   'out_len', 'out_value', 'out_number', 'is_coinbase']], x_rematch[['label']]
print(X.shape, y.shape)

X = np.array(X)
Y = np.array(y)

# x_rematch = np.array(x_rematch)
# y_rematch = np.array(y_rematch)
# y_rematch = y_rematch.ravel()

Y=Y.ravel()
print('提取后的矩阵维度',X.shape)

X_std = X

X_train, X_test, y_train, y_test = train_test_split(
    X_std, Y, test_size=0.1, random_state=110)
print('训练集数据：\n', X_train,'\n', y_train,'\n测试集数据\n', X_test,'\n', y_test)

print('services', np.sum(y_test=='services'), np.sum(y_test=='services')/len(y_test))
print('exchanges', np.sum(y_test=='exchanges'), np.sum(y_test=='exchanges')/len(y_test))
print('gambling', np.sum(y_test=='gambling'), np.sum(y_test=='gambling')/len(y_test))
y_all = y_test.copy()
y_all[:] = 'services'

names = 'Random_Forest'

parameters = {'max_depth': [5, 10, 15],
              'n_estimators': [10, 100, 1000],
              'max_features': [1, 3, 5],
              'min_samples_split': [0.2, 0.6, 0.8],
              'min_samples_leaf': [1, 3, 5],
              # 'class_weight': ["balanced"]
               }
xlf = RandomForestClassifier(max_depth=5, n_estimators=100, max_features=2, min_samples_leaf=1, n_jobs=-1, class_weight="balanced")

from sklearn.model_selection import GridSearchCV
print('Grid Searching...')
gsearch = GridSearchCV(xlf, param_grid=parameters, scoring='accuracy', cv=2, verbose=1, n_jobs=-1)
#
time_start = time.time()
gsearch.fit(X_train, y_train)
time_train = time.time()
#
#
print("Best score: %0.4f" % gsearch.best_score_)
print("Best parameters set:")
joblib.dump(gsearch.best_estimator_, 'RandomForest.m')
# joblib.dump(gsearch, 'RF_FIXED.m')
best_parameters = gsearch.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

y_pred = gsearch.predict(X_test)
time_predict = time.time()
y_test = y_test
print('training time', time_train-time_start)
print('predicting time', time_predict-time_start)
print("\tAccuracy: %1.3f" % accuracy_score(y_test, y_pred))
print("\tPrecision: %1.3f" % precision_score(y_test, y_pred, average='macro'))
print("\tRecall: %1.3f" % recall_score(y_test, y_pred, average='macro'))
print("\tF1: %1.3f\n" % f1_score(y_test, y_pred, average='macro'))
print("\tConfusion matrix: \n" , confusion_matrix(y_test, y_pred))
target_names = ['services', 'exchanges', 'gambling']
print(classification_report(y_test, y_pred, target_names=target_names))

# from sklearn.metrics import precision_recall_fscore_support as score
#
# precision, recall, fscore, support = score(y_test, y_pred)
#
# print('precision: {}'.format(precision))
# print('recall: {}'.format(recall))
# print('fscore: {}'.format(fscore))
# print('support: {}'.format(support))
