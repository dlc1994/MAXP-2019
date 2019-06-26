import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# btc_vin.csv记录了训练集和测试集地址相关的所有输入（inputs）记录，字段解释：
# tx_id:交易id
# tx_id_origin:引用的输出的交易id
# timestamp:时间戳
# number:交易的第几条输入
# vout_num_origin:引用的第几条输出
# address_origin:引用的输出的地址
# value_origin:引用的输出的原始金额

# btc_vout.csv记录了训练集和测试集地址相关的所有输出（outputs）记录，字段解释：
# tx_id:交易id
# value:金额
# address:地址
# timestamp:时间戳
# number:交易的第几条输出
# is_coinbase:是否是挖矿交易

DataSet_DIR = r'./address_classfication_rematch.jdcloud/test.csv'
# DataSet_DIR = r'./address_classfication-dataset.jdcloud/train.csv'
BTC_VIN_DIR = r'./address_classfication-dataset.jdcloud/btc_vin.csv'
BTC_VOUT_DIR = r'./address_classfication-dataset.jdcloud/btc_vout.csv'
n_samples = 0
if n_samples>0:
    print('Reading data...')
    train_df = pd.read_csv(DataSet_DIR, nrows=n_samples)
    print('Loading train dataset completed!')
    train_vin_df = pd.read_csv(BTC_VIN_DIR, nrows=n_samples)
    print('Loading vin completed!')
    train_vout_df = pd.read_csv(BTC_VOUT_DIR, nrows=n_samples)
    train_vout_df.to_csv('btc_vout1.csv')
    print("Loading vout completed!")
else:
    print('Reading data...')
    train_df = pd.read_csv(DataSet_DIR)
    print('Loading train dataset completed!')
    train_vin_df = pd.read_csv(BTC_VIN_DIR)
    print('Loading vin completed!')
    train_vout_df = pd.read_csv(BTC_VOUT_DIR)
    # train_vout_df.to_csv('btc_vout1.csv')
    print("Loading vout completed!")

print('train dataset shape', train_df.shape)
print('vin shape', train_vin_df.shape)
print('vout shape', train_vout_df.shape)

for i in range(265001, train_df.shape[0]):
    if i%10 == 0: print('iterations', i)
    address = train_df.loc[i, 'address']
    tmp = train_vin_df[train_vin_df['address_origin'].isin([address])]
    # print(tmp)
    if len(tmp)>0:
        # print('tmp', tmp)
        train_df.loc[i, 'in_len'] = len(tmp)
        train_df.loc[i, 'number'] = sum(tmp['number'])
        train_df.loc[i, 'vout_num_origin'] = sum(tmp['vout_num_origin'])
        train_df.loc[i, 'value_origin'] = sum(tmp['value_origin'])
        # train_df.loc[i, 'average'] = sum(tmp['value_origin'])/len(tmp)
    vout_tmp = train_vout_df[train_vout_df['address'].isin([address])]
    if len(vout_tmp) > 0:
        # tx_ids = vout_tmp.loc[:, ['tx_id']]
        # out_df = train_vout_df[train_vout_df['tx_id'].isin(tx_ids)]
        train_df.loc[i, 'out_len'] = len(vout_tmp)
        train_df.loc[i, 'out_value'] = sum(vout_tmp['value'])
        train_df.loc[i, 'out_number'] = sum(vout_tmp['number'])
        train_df.loc[i, 'is_coinbase'] = sum(vout_tmp['is_coinbase'])

    if i%1000==0:   train_df.to_csv('rematch_test_all2.csv', na_rep=0)
        # for j in range(len(tmp)):
        #     tx_id = tmp.iloc[j]['tx_id']
        #     tx_tmp = train_vout_df[train_vout_df['tx_id'].isin([tx_id])]
        #     if len(tx_tmp)>0:
        #         print('tx_tmp', tx_tmp)
        #         train_df[i, 'out_len'] = len(tx_tmp)
        #         train_df[i, 'out_value'] = tx_tmp['value']
        #         train_df[i, 'out_number'] = tx_tmp['number']
        #         train_df[i, 'is_coinbase'] = tx_tmp['is_coinbase']
        #     else:
        #         train_df[i, 'out_number'] = 0
    #
    #
    # else:
    #     train_df.loc[i, 'number'] = 0
    #     train_df.loc[i, 'total'] = 0
    #     train_df.loc[i, 'average'] = 0
    #     train_df.loc[i, 'out_number'] = 0
# train_df.fillna(0)

# train_df.to_csv('processed_train_dataset.csv',na_rep=0)