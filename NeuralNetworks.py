import pandas as pd
from sklearn.decomposition import PCA
from sklearn import decomposition
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error
import sklearn.externals.joblib as joblib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

#读取数据
LoanStats3a=pd.read_csv('preliminary_train_dataset.csv')
x_services = LoanStats3a[LoanStats3a['label']=='services']
x_exchanges = LoanStats3a[LoanStats3a['label']=='exchanges']
x_gamblings = LoanStats3a[LoanStats3a['label']=='gambling']
print(x_services.shape, x_exchanges.shape, x_gamblings.shape)
LoanStats3a = pd.concat([x_services,x_exchanges],axis=0,ignore_index=True)

LoanStats3a=LoanStats3a[~(
    LoanStats3a['in_len'].isin([0]) & LoanStats3a['number'].isin([0])& LoanStats3a['vout_num_origin'].isin([0])& LoanStats3a['value_origin'].isin([0])
    & LoanStats3a['out_len'].isin([0]) & LoanStats3a['out_value'].isin([0]) & LoanStats3a['out_number'].isin([0]) & LoanStats3a['is_coinbase'].isin([0])
                          )]
print(LoanStats3a.shape)
X, y = LoanStats3a[['in_len', 'number', 'vout_num_origin', 'value_origin', 'out_len', 'out_value', 'out_number', 'is_coinbase']], LoanStats3a[['label']]
# X['in_num_ave'] = X['number'] / X['in_len']
# X['in_voutnum_ave'] = X['vout_num_origin'] / X['in_len']
# X['in_value_ave'] = X['value_origin'] / X['in_len']
# X['out_value_ave'] = X['out_value'] / X['out_len']
# X['out_number_ave'] = X['out_number'] / X['out_len']
# X = X.fillna(0)
print(X.shape, y.shape)
# y_all = y
# y_all['label'] = 'services'

X = np.array(X)
Y = np.array(y)
Y=Y.ravel()

#查看特征的维度
print('提取后的矩阵维度',X.shape)
# sc = StandardScaler()
# X_std = sc.fit_transform(X)
X_std = X

# joblib.dump(sc, 'standarsc')
#查看标准化后的特征数据
# print('standardization\n',X_std)

# 创建PCA对象，n_components为主成分维度——列 12 0.817, 3 0.856, 4 0.868, 5 0.871, 6 0.887, 7 0.887
# pca = decomposition.PCA(n_components=7)
# #使用PCA对特征进行降维
# X_std = pca.fit_transform(X_std)

X_train, X_test, y_train, y_test = train_test_split(
    X_std, Y, test_size=0.3, random_state=0)
print('训练集数据：\n', X_train,'\n', y_train,'\n测试集数据\n', X_test,'\n', y_test)

print('services', np.sum(y_test=='services'), np.sum(y_test=='services')/len(y_test))
print('exchanges', np.sum(y_test=='exchanges'), np.sum(y_test=='exchanges')/len(y_test))
print('gambling', np.sum(y_test=='gambling'), np.sum(y_test=='gambling')/len(y_test))
y_all = y_test.copy()
y_all[:] = 'services'
# print('all services', np.sum(y_all=='services'), np.sum(y_all=='services')/len(y_all))

tt = "rf"
if tt == "all":
    names = ["Nearest Neighbors", "GBDT",
             "Decision Tree", "Random Forest", "Neural Net", "AdaBoost", "Naive Bayes"]

    classifiers = [
        KNeighborsClassifier(3),
        # SVC(kernel="linear", C=0.025),
        # SVC(gamma=2, C=1),
        GradientBoostingClassifier(n_estimators=100, learning_rate=0.01, max_depth=1, random_state=0),
        # GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=10, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1, max_iter=500),
        AdaBoostClassifier(),
        GaussianNB()]
    score = []
    # # iterate over classifiers
    for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        # clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric="merror", early_stopping_rounds=10,
        #           verbose=True)
        tmp = clf.score(X_test, y_test)
        # clf.save_model('XGBoost_{:.3f}.m'.format(tmp))
        # clf.load_model('XGBoost')
        # scores = cross_val_score(clf, X_test, y_test, cv=5)
        # print('cross validation', scores)
        # tmp = np.mean(scores)
        # print('predict accuracy', accuracy_score(clf.predict(X_test), y_test))
        # joblib.dump(clf, 'XGBoost_%s.m'%tmp)
        # print('predict "services" for all test sample:', accuracy_score(y_all, y_test))
        score.append(tmp)
        print(name, '%.3f' % tmp)

elif tt == "XGB":
    # names = ["Random Forest"]
    # classifiers = [RandomForestClassifier(max_depth=15, n_estimators=300, max_features=5)]
    # names = ["GB DT"]
    # classifiers = [GradientBoostingClassifier(n_estimators=50, learning_rate=1, max_depth=10, random_state=0)]
    names = 'XGBoost'
    classifiers = [XGBClassifier(learning_rate=0.1,
                       n_estimators=1000,         # 树的个数--1000棵树建立xgboost
                       max_depth=10,               # 树的深度
                       min_child_weight = 1,      # 叶子节点最小权重
                      gamma=0.,                  # 惩罚项中叶子结点个数前的参数
                      subsample=0.8,             # 随机选择80%样本建立决策树
                      colsample_btree=0.5,       # 随机选择80%特征建立决策树
                      objective='multi:softmax', # 指定损失函数
                      scale_pos_weight=1,        # 解决样本个数不平衡的问题
                       random_state=27            # 随机数
                      )]
    xlf = XGBClassifier(max_depth=10,
                        learning_rate=0.01,
                        n_estimators=2000,
                        silent=True,
                        objective='multi:softmax',
                        nthread=-1,
                        gamma=0,
                        min_child_weight=1,
                        max_delta_step=0,
                        subsample=0.85,
                        colsample_bytree=0.7,
                        colsample_bylevel=1,
                        reg_alpha=0,
                        reg_lambda=1,
                        scale_pos_weight=1,
                        seed=1440,
                        missing=None)

    # parameters = {
    #               'max_depth': [5, 10, 15, 20, 25],
    #               'learning_rate': [0.01, 0.02, 0.05, 0.1],
    #               'n_estimators': [500, 1000, 2000, 3000],
    #               'min_child_weight': [0, 2, 5, 10, 20],
    #               'max_delta_step': [0, 0.2, 1, 2],
    #               'subsample': [0.6, 0.7, 0.8, 0.85],
    #               'colsample_bytree': [0.5, 0.6, 0.7, 0.8],
    #               'reg_alpha': [0, 0.25, 0.5, 1],
    #               'reg_lambda': [0.2, 0.4, 0.6, 1],
    #               'scale_pos_weight': [0.2, 0.4, 0.8, 1]
    # }
    parameters = {
                  'max_depth': [5, 10, 15, 20, 25],
                  'n_estimators': [10, 100, 1000, 2000],
                  'min_child_weight': [0, 2, 5, 10, 20],
                  'scale_pos_weight': [0.2, 0.4, 0.6, 0.8, 1]
}
elif tt == "rf":
    names = 'Random Forest'
    classifiers = [RandomForestClassifier(max_depth=10, n_estimators=10, max_features=1)]
    parameters = {'max_depth': [5, 8, 10, 15],
                  'n_estimators': [100, 150, 200, 500],
                  'max_features': [1, 3, 5, 8],
                  # 'min_samples_split': [0.2, 0.5, 0.8, 1],
                    'min_samples_leaf': [1, 3, 5, 8],
                   }
    xlf = RandomForestClassifier(max_depth=10, n_estimators=10, max_features=1)


# 有了gridsearch我们便不需要fit函数
from sklearn.model_selection import GridSearchCV
print('Grid Searching...')
gsearch = GridSearchCV(xlf, param_grid=parameters, scoring='accuracy', cv=3, verbose=1, n_jobs=-1)
gsearch.fit(X_train, y_train)

print("Best score: %0.3f" % gsearch.best_score_)
print("Best parameters set:")
joblib.dump(gsearch, names+'_GridSearch_%.4f.m'%gsearch.best_score_)
best_parameters = gsearch.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

y_pred = gsearch.predict(X_test)
print("\tPrecision: %1.3f" % precision_score(y_test, y_pred, average='macro'))
print("\tRecall: %1.3f" % recall_score(y_test, y_pred, average='macro'))
print("\tF1: %1.3f\n" % f1_score(y_test, y_pred, average='macro'))

from sklearn.metrics import precision_recall_fscore_support as score

precision, recall, fscore, support = score(y_test, y_pred)

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))


# score = []
# # # iterate over classifiers
# for name, clf in zip(names, classifiers):
#     clf.fit(X_train, y_train)
#     # clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric="merror", early_stopping_rounds=10,
#     #           verbose=True)
#     tmp = clf.score(X_test, y_test)
#     # clf.save_model('XGBoost_{:.3f}.m'.format(tmp))
#     # clf.load_model('XGBoost')
#     # scores = cross_val_score(clf, X_test, y_test, cv=5)
#     # print('cross validation', scores)
#     # tmp = np.mean(scores)
#     # print('predict accuracy', accuracy_score(clf.predict(X_test), y_test))
#     # joblib.dump(clf, 'XGBoost_%s.m'%tmp)
#     print('predict "services" for all test sample:', accuracy_score(y_all, y_test))
#     score.append(tmp)
#     print(name, '%.3f'%tmp)

