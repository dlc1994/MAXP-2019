# encoding: utf-8
'''
@author: Lingcheng Dai
@contact: 2013210288@bupt.edu.cn
@file: cluster.py
@time: 2019/4/22 15:30
'''
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import multiprocessing

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

# DataSet_DIR = r'./address_classfication_preliminary.jdcloud/test.csv'
DataSet_DIR = r'train.csv'
BTC_VIN_DIR = r'btc_vin.csv'
BTC_VOUT_DIR = r'btc_vout.csv'


if __name__ == "__main__":
   

    print('Reading data...')
    train_df = pd.read_csv(DataSet_DIR)
    print('Loading train dataset completed!')
    train_vin_df = pd.read_csv(BTC_VIN_DIR)
    print('Loading vin completed!')
    # train_vout_df = pd.read_csv(BTC_VOUT_DIR)
    # train_vout_df.to_csv('btc_vout1.csv')
    print("Loading vout completed!")

    ds = pd.DataFrame(columns=['address', 'label'])


    for i in range(train_df.shape[0]):
        if i%100 ==0: print('iterations', i)
        address, label = train_df.loc[i, 'address'], train_df.loc[i, 'label']
        ds = ds.append({'address': address, 'label': label}, ignore_index=True)
        tmp = train_vin_df[train_vin_df['address_origin']==address]
        if len(tmp) > 0:
            for t in tmp.index:
                tx_id = tmp.loc[t, 'tx_id']
                in_txid = train_vin_df[train_vin_df['tx_id'] == tx_id]
                if len(in_txid) > 0:
                    print('vin', len(in_txid))
                    in_txid = in_txid[['address_origin']]
                    in_txid.rename(columns={'address_origin': 'address'}, inplace=True)
                    in_txid['label'] = label
                    ds = ds.append(in_txid, ignore_index=True)
		
        if i % 100 == 0:
			ds = ds.drop_duplicates()
			ds.to_csv('newDATASET_rear.csv', index=False)

