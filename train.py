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

#读取数据
LoanStats3a=pd.read_csv('processed_train_dataset17k.csv', nrows=17000)
# print(LoanStats3a.columns.values.tolist())
X, y = LoanStats3a[['in_len', 'number', 'vout_num_origin', 'value_origin', 'out_len', 'out_value', 'out_number', 'is_coinbase']], LoanStats3a[['label']]
print(X.shape, y.shape)

X = np.array(X)
Y = np.array(y)
Y=Y.ravel()

#查看特征的维度
print('提取后的矩阵维度',X.shape)
sc = StandardScaler()
X_std = sc.fit_transform(X)

joblib.dump(sc, 'standarsc')
#查看标准化后的特征数据
print('standardization\n',X_std)

# 创建PCA对象，n_components为主成分维度——列
# pca = decomposition.PCA(n_components=1)
# # #使用PCA对特征进行降维
# xGeoPCA = pca.fit_transform(X_std)

X_train, X_test, y_train, y_test = train_test_split(
    X_std, Y, test_size=0.3, random_state=23375)
print('训练集数据：\n', X_train,'\n', y_train,'\n测试集数据\n', X_test,'\n', y_test)

tt = "single"
if tt == "all":
    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "GBDT",
             "Decision Tree", "Random Forest", "Neural Net", "AdaBoost", "Naive Bayes"]

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        GradientBoostingClassifier(n_estimators=100, learning_rate=0.01, max_depth=1, random_state=0),
        # GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=10, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1, max_iter=500),
        AdaBoostClassifier(),
        GaussianNB()]
elif tt == "single":
    names = ["Random Forest"]
    classifiers = [RandomForestClassifier(max_depth=10, n_estimators=50, max_features=8)]

score = []
# iterate over classifiers
for name, clf in zip(names, classifiers):
    clf.fit(X_train, y_train)
    tmp = clf.score(X_test, y_test)
    print(clf.predict(X_test), y_test)
    # joblib.dump(clf, 'RF_10_50_8_%s.m'%tmp)
    score.append(tmp)
    print(name, tmp)

# xxx = np.logspace(-4, -1, 4)
# zzz = [50, 100, 200]
# yyy = [1, 2, 3, 4]
# print(xxx)
# # xxx=[1.2]
#
# a = 100
# a1 = 10000
#
# scores = []
# scores1 = []
# count = 0
# aaaa = np.ones((len(xxx),len(xxx),len(zzz)), dtype=np.float64)
# aaaa = aaaa*1000
#
# for zz, vz in zip(range(len(zzz)), zzz):
#     for yy in yyy:
#         for ii, vi in zip(range(len(xxx)), xxx):
#             for jj, vj in zip(range(len(xxx)), xxx):
#                 print('循环次数', zz, yy, ii, jj)
#                 # clf = KNeighborsRegressor(int(ii), weights='uniform')
#                 clf1 = MLPRegressor(hidden_layer_sizes=(vz, yy), alpha=vi, solver='sgd', learning_rate_init=vj, max_iter=1000)
#                 clf1.fit(x_train, y_train)
#                 y_predict = clf1.predict(x_test)
#                 if np.isnan(y_predict).any() == True:
#                     continue
#                 MAE_TEMP1 = mean_absolute_error(y_predict, y_test)
#                 # aaaa[ii][jj][zz] = MAE_TEMP1
#                 print(MAE_TEMP1)
#                 if MAE_TEMP1 < a1:
#                     joblib.dump(clf1, 'clf_NN_3D_1214.m')
#                     a1 = MAE_TEMP1
#
#         # MAE_TEMP = cross_val_score(clf, XXX, Y, cv=5, scoring='neg_mean_absolute_error')
#         # MAE_TEMP1 = cross_val_score(clf1, x_train, y_train, cv=5, scoring='neg_mean_absolute_error')
#         # # MAE_TEMP = -MAE_TEMP
#         # MAE_TEMP1 = -MAE_TEMP1
#         # print(ii, MAE_TEMP1, MAE_TEMP1)
#         # # scores.append(MAE_TEMP.mean())
#         # scores1.append(MAE_TEMP1.mean())
#         # if MAE_TEMP1.mean() < a1:
#         #     clf1.fit(x_train, y_train)
#         #     joblib.dump(clf1, 'clf_NN_25D_CV5_all.m')
#         #     a1 = MAE_TEMP1.mean()
#
#         count += 1
#
# # print(scores)
# print(aaaa)
# print('最低MAE',a, a1)
# # scores_np = np.array(scores, dtype=float)
# scores_np1 = np.array(scores1,dtype=float)
# # print(scores_np, scores_np1)
# # np.savetxt('uniform.txt',scores_np)
# # np.savetxt('distance.txt',aaaa)
#
#
# clf_best = joblib.load('clf_NN_3D_1214.m') #加载离线预测模型
# print(clf_best.get_params())
# y_predict1=clf_best.predict(XXX)
# print(y_predict1)
# print(Y)
# print('all MSE', mean_absolute_error(y_predict1, Y))


# import matplotlib.pyplot as plt
# from matplotlib import cm
# from mpl_toolkits.mplot3d import axes3d
#
# # x = range(1, len(scores_np1)+1)
# # # plt.plot(x, scores_np,  marker='^', c='r', linewidth=1, label='Weights = "uniform"')
# # plt.plot(x, scores_np1,  marker='o', c='b', linewidth=1, label='Weights = "distance"')
# # plt.legend()
# # plt.xlabel('n_neighbors')
# # plt.ylabel('MAE')
# # plt.grid(linestyle=':')
# # plt.show()
# X,Y=np.meshgrid(xxx,xxx)
# fig=plt.figure()
# ax1 = fig.add_subplot(141, projection='3d')
# ax1.gca(projection='3d')
# # ax.plot_surface(X,Y,zzz,cmap=cm.hot)
# ax1.plot_surface(X, Y, aaaa[:,:,0], cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)
# ax2 = fig.add_subplot(142)
# ax2.plot(xxx, )
#
# plt.show()


# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FormatStrFormatter
# import numpy as np
#
# fig = plt.figure()
# ax = fig.gca(projection='3d')
#
# # Make data.
# X = np.arange(-5, 5, 0.25)
# Y = np.arange(-5, 5, 0.25)
# X, Y = np.meshgrid(X, Y)
# R = np.sqrt(X ** 2 + Y ** 2)
# Z = np.sin(R)
#
# # Plot the surface.
# surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)
#
# # Customize the z axis.
# ax.set_zlim(-1.01, 1.01)
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
#
# # Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)
#
# plt.show()
