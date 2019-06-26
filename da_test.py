import pandas as pd
import numpy as np
import sklearn.externals.joblib as joblib
from sklearn.metrics import accuracy_score

# rfclassifier = joblib.load('RF_BEST.m')
# parameters = {'max_depth': [5, 8, 10, 15],
#               'n_estimators': [100, 150, 200, 500],
#               'max_features': [1, 3, 5, 8],
#               # 'min_samples_split': [0.2, 0.5, 0.8, 1],
#               'min_samples_leaf': [1, 3, 5, 8],
#               }
# print("Best parameters set:")
# best_parameters = rfclassifier.best_estimator_.get_params()
# print(best_parameters)
# for param_name in sorted(parameters.keys()):
#     print("\t%s: %r" % (param_name, best_parameters[param_name]))

# standard_answer = pd.read_csv('preliminary_test_submission.csv')
# classifier = joblib.load('RF_FIXED.m')
# sa_np = np.array(standard_answer.loc[:, 'label'])
# dataset = pd.read_csv('processed_test_datasetALL.csv')
# X = dataset[['in_len', 'number', 'vout_num_origin', 'value_origin', 'out_len', 'out_value', 'out_number', 'is_coinbase']]
# X = np.array(X)
# check = classifier.predict(X)
# print('accuracy', accuracy_score(check, sa_np))


# df = pd.read_csv('new1.csv')
# submission = pd.read_csv('rematch_test_submission_server.csv')
# # alist = df.ix[:, [1, 2]]
# alist = np.array(df.ix[:, [1, 2]])
# # print(alist.shape)
# for i in alist:
#     submission.loc[i[0], ['label']] = 'exchanges'
# submission.to_csv('rematch_test_submission0605.csv', index=False)

submission = 'TRAINDATASET.csv'
# result = pd.read_csv('TRAINDATASET.csv')
# result = result[['address', 'label']]
# result.to_csv(submission, index=False)
csv_list = ['TRAINDATASET25_50.csv', 'TRAINDATASET50_75.csv', 'TRAINDATASET75_100.csv']
for inputfile in csv_list:
    a = pd.read_csv(inputfile)  # header=None表示原始文件数据没有列索引，这样的话read_csv会自动加上列索引
    # a = a[['address', 'label']]
    a.to_csv(submission, mode='a', index=False, header=False)
    # header=0表示不保留列名，index=False表示不保留行索引，mode='a'表示附加方式写入，文件原有内容不会被清除

print('done!')
