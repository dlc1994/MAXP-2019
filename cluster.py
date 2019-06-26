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
DataSet_DIR = r'./address_classfication-dataset.jdcloud/train.csv'
BTC_VIN_DIR = r'./address_classfication-dataset.jdcloud/btc_vin.csv'
BTC_VOUT_DIR = r'./address_classfication-dataset.jdcloud/btc_vout.csv'
#
# def save_dataset1(x):
#     print('Reading data...')
#     train_df = pd.read_csv(DataSet_DIR)
#     print('Loading train dataset completed!')
#     train_vin_df = pd.read_csv(BTC_VIN_DIR, nrows=1000)
#     print('Loading vin completed!')
#     train_vout_df = pd.read_csv(BTC_VOUT_DIR, nrows=1000)
#     # train_vout_df.to_csv('btc_vout1.csv')
#     print("Loading vout completed!")
#     ds = pd.DataFrame(columns=['address', 'label'])
#     end = int(train_df.shape[0]*0.25)
#     for i in range(end):
#         if i%1000 != 0: print(1, 'iterations', i)
#         address, label = train_df.loc[i, 'address'], train_df.loc[i, 'label']
#         ds = ds.append({'address':address, 'label':label}, ignore_index=True)
#         tmp = train_vin_df[train_vin_df['address_origin'].isin([address])]
#         if len(tmp)>0:
#             for t in tmp.index:
#                 tx_id = tmp.loc[t, 'tx_id']
#                 out_tmp = train_vout_df[train_vout_df['tx_id'].isin([tx_id])]
#                 if len(out_tmp) > 0:
#                     ds = ds.append({'address':out_tmp['address'], 'label':label}, ignore_index=True)
#
#                 tx_id_origin = tmp.loc[t, 'tx_id_origin']
#                 out_origin_tmp = train_vout_df[train_vout_df['tx_id'].isin([tx_id_origin])]
#                 if len(out_origin_tmp) > 0:
#                     # print('out_origin', out_origin_tmp)
#                     out_origin_tmp.loc[:, 'label'] = label
#                     ds = ds.append(out_origin_tmp.loc[:, ['address', 'label']], ignore_index=True)
#         if i % 1000 == 0:   train_df.to_csv('ds1.csv')
#
# def save_dataset2(x):
#     print('Reading data...')
#     train_df = pd.read_csv(DataSet_DIR)
#     print('Loading train dataset completed!')
#     train_vin_df = pd.read_csv(BTC_VIN_DIR, nrows=1000)
#     print('Loading vin completed!')
#     train_vout_df = pd.read_csv(BTC_VOUT_DIR, nrows=1000)
#     # train_vout_df.to_csv('btc_vout1.csv')
#     print("Loading vout completed!")
#     ds = pd.DataFrame(columns=['address', 'label'])
#     for i in range(int(train_df.shape[0]*0.25), int(train_df.shape[0]*0.5)):
#         if i%1000 != 0: print('iterations', 2, i)
#         address, label = train_df.loc[i, 'address'], train_df.loc[i, 'label']
#         ds = ds.append({'address':address, 'label':label}, ignore_index=True)
#         tmp = train_vin_df[train_vin_df['address_origin'].isin([address])]
#         if len(tmp)>0:
#             for t in tmp.index:
#                 tx_id = tmp.loc[t, 'tx_id']
#                 out_tmp = train_vout_df[train_vout_df['tx_id'].isin([tx_id])]
#                 if len(out_tmp) > 0:
#                     ds = ds.append({'address':out_tmp['address'], 'label':label}, ignore_index=True)
#
#                 tx_id_origin = tmp.loc[t, 'tx_id_origin']
#                 out_origin_tmp = train_vout_df[train_vout_df['tx_id'].isin([tx_id_origin])]
#                 if len(out_origin_tmp) > 0:
#                     # print('out_origin', out_origin_tmp)
#                     out_origin_tmp.loc[:, 'label'] = label
#                     ds = ds.append(out_origin_tmp.loc[:, ['address', 'label']], ignore_index=True)
#         if i % 1000 == 0:   train_df.to_csv('ds2.csv')
#
# def save_dataset3(x):
#     print('Reading data...')
#     train_df = pd.read_csv(DataSet_DIR)
#     print('Loading train dataset completed!')
#     train_vin_df = pd.read_csv(BTC_VIN_DIR, nrows=1000)
#     print('Loading vin completed!')
#     train_vout_df = pd.read_csv(BTC_VOUT_DIR, nrows=1000)
#     # train_vout_df.to_csv('btc_vout1.csv')
#     print("Loading vout completed!")
#     ds = pd.DataFrame(columns=['address', 'label'])
#     for i in range(int(train_df.shape[0]*0.5), int(train_df.shape[0]*0.75)):
#         if i%1000 != 0: print('iterations', 3, i)
#         address, label = train_df.loc[i, 'address'], train_df.loc[i, 'label']
#         ds = ds.append({'address':address, 'label':label}, ignore_index=True)
#         tmp = train_vin_df[train_vin_df['address_origin'].isin([address])]
#         if len(tmp)>0:
#             for t in tmp.index:
#                 tx_id = tmp.loc[t, 'tx_id']
#                 out_tmp = train_vout_df[train_vout_df['tx_id'].isin([tx_id])]
#                 if len(out_tmp) > 0:
#                     ds = ds.append({'address':out_tmp['address'], 'label':label}, ignore_index=True)
#
#                 tx_id_origin = tmp.loc[t, 'tx_id_origin']
#                 out_origin_tmp = train_vout_df[train_vout_df['tx_id'].isin([tx_id_origin])]
#                 if len(out_origin_tmp) > 0:
#                     # print('out_origin', out_origin_tmp)
#                     out_origin_tmp.loc[:, 'label'] = label
#                     ds = ds.append(out_origin_tmp.loc[:, ['address', 'label']], ignore_index=True)
#         if i % 1000 == 0:   train_df.to_csv('ds3.csv')
#
# def save_dataset4(x):
#     print('Reading data...')
#     train_df = pd.read_csv(DataSet_DIR)
#     print('Loading train dataset completed!')
#     train_vin_df = pd.read_csv(BTC_VIN_DIR, nrows=1000)
#     print('Loading vin completed!')
#     train_vout_df = pd.read_csv(BTC_VOUT_DIR, nrows=1000)
#     # train_vout_df.to_csv('btc_vout1.csv')
#     print("Loading vout completed!")
#     ds = pd.DataFrame(columns=['address', 'label'])
#     for i in range(int(train_df.shape[0]*0.75), train_df.shape[0]):
#         if i%1000 != 0: print('iterations', 4, i)
#         address, label = train_df.loc[i, 'address'], train_df.loc[i, 'label']
#         ds = ds.append({'address':address, 'label':label}, ignore_index=True)
#         tmp = train_vin_df[train_vin_df['address_origin'].isin([address])]
#         if len(tmp)>0:
#             for t in tmp.index:
#                 tx_id = tmp.loc[t, 'tx_id']
#                 out_tmp = train_vout_df[train_vout_df['tx_id'].isin([tx_id])]
#                 if len(out_tmp) > 0:
#                     ds = ds.append({'address':out_tmp['address'], 'label':label}, ignore_index=True)
#
#                 tx_id_origin = tmp.loc[t, 'tx_id_origin']
#                 out_origin_tmp = train_vout_df[train_vout_df['tx_id'].isin([tx_id_origin])]
#                 if len(out_origin_tmp) > 0:
#                     # print('out_origin', out_origin_tmp)
#                     out_origin_tmp.loc[:, 'label'] = label
#                     ds = ds.append(out_origin_tmp.loc[:, ['address', 'label']], ignore_index=True)
#         if i % 1000 == 0:   train_df.to_csv('ds4.csv')

if __name__ == "__main__":
    # pool = multiprocessing.Pool(processes=4)
    # #
    # # n_samples = 2000
    # # if n_samples > 0:
    # #     print('Reading data...')
    # #     train_df = pd.read_csv(DataSet_DIR, nrows=n_samples)
    # #     print('Loading train dataset completed!')
    # #     train_vin_df = pd.read_csv(BTC_VIN_DIR, nrows=n_samples)
    # #     print('Loading vin completed!')
    # #     train_vout_df = pd.read_csv(BTC_VOUT_DIR, nrows=n_samples)
    # #     # train_vout_df.to_csv('btc_vout1.csv')
    # #     print("Loading vout completed!")
    # # else:
    # #     print('Reading data...')
    # #     train_df = pd.read_csv(DataSet_DIR)
    # #     print('Loading train dataset completed!')
    # #     train_vin_df = pd.read_csv(BTC_VIN_DIR)
    # #     print('Loading vin completed!')
    # #     train_vout_df = pd.read_csv(BTC_VOUT_DIR)
    # #     # train_vout_df.to_csv('btc_vout1.csv')
    # #     print("Loading vout completed!")
    #
    # # print('train dataset shape', train_df.shape, train_df.head())
    # # print('vin shape', train_vin_df.shape, train_vin_df.head())
    # # print('vout shape', train_vout_df.shape, train_vout_df.head())
    #
    #
    # pool.apply_async(save_dataset1, (1,))
    # pool.apply_async(save_dataset2, (2,))
    # pool.apply_async(save_dataset3, (3,))
    # pool.apply_async(save_dataset4, (4,))
    #
    # pool.close()
    # pool.join()
    #
    # for inputfile in ['ds1.csv','ds2.csv','ds3.csv','ds4.csv']:
    #     a = pd.read_csv(inputfile) # header=None表示原始文件数据没有列索引，这样的话read_csv会自动加上列索引
    #     a.to_csv('submission_0502.csv', mode='a', index=False,
    #               header=False)  # header=0表示不保留列名，index=False表示不保留行索引，mode='a'表示附加方式写入，文件原有内容不会被清除
    # print('done!')

    print('Reading data...')
    train_df = pd.read_csv(DataSet_DIR)
    print('Loading train dataset completed!')
    train_vin_df = pd.read_csv(BTC_VIN_DIR)
    print('Loading vin completed!')
    # train_vout_df = pd.read_csv(BTC_VOUT_DIR)
    # train_vout_df.to_csv('btc_vout1.csv')
    print("Loading vout completed!")

    ds = pd.DataFrame(columns=['address', 'label'])
    # for i in range(train_df.shape[0]):
    #     print('iterations', i)
    #     address, label = train_df.loc[i, 'address'], train_df.loc[i, 'label']
    #     ds = ds.append({'address': address, 'label': label}, ignore_index=True)
    #     tmp = train_vin_df[train_vin_df['address_origin']==address]
    #     if len(tmp) > 0:
    #         for t in tmp.index:
    #             tx_id = tmp.loc[t, 'tx_id']
    #             out_tmp = train_vout_df[train_vout_df['tx_id']==tx_id]
    #             if len(out_tmp) > 0:
    #                 ds = ds.append(out_tmp.loc[:, ['address', 'label']], ignore_index=True)
    #             tx_id_origin = tmp.loc[t, 'tx_id_origin']
    #             out_origin_tmp = train_vout_df[train_vout_df['tx_id']==tx_id_origin]
    #             if len(out_origin_tmp) > 0:
    #                 # print('out_origin', out_origin_tmp)
    #                 out_origin_tmp.loc[:, 'label'] = label
    #                 ds = ds.append(out_origin_tmp.loc[:, ['address', 'label']], ignore_index=True)
    #     if i % 100 == 0:   train_df.to_csv('DATASET.csv', index=False)

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
                    print('vin', in_txid)
                    in_txid = in_txid[['address_origin']]
                    in_txid.rename(columns={'address_origin': 'address'}, inplace=True)
                    in_txid['label'] = label
                    ds = ds.append(in_txid, ignore_index=True)
        #
        # out_tmp = train_vout_df[train_vout_df['address'] == address]
        # if len(out_tmp) > 0:
        #     for t in out_tmp.index:
        #         tx_id = tmp.loc[t, 'tx_id']
        #         in_txid = train_vin_df[train_vin_df['tx_id'] == tx_id]
        #         if len(in_txid) > 0:
        #             print('vin', in_txid)
        #             in_txid = in_txid[in_txid['address_origin']]
        #             in_txid.rename(columns={'address_origin': 'address'}, inplace=True)
        #             in_txid['label'] = label
        #             ds = ds.append(in_txid, ignore_index=True)
        #
        #
        #         out_tmp = train_vout_df[train_vout_df['tx_id']==tx_id]
        #         if len(out_tmp) > 0:
        #             ds = ds.append(out_tmp.loc[:, ['address', 'label']], ignore_index=True)
        #         tx_id_origin = tmp.loc[t, 'tx_id_origin']
        #         out_origin_tmp = train_vout_df[train_vout_df['tx_id']==tx_id_origin]
        #         if len(out_origin_tmp) > 0:
        #             # print('out_origin', out_origin_tmp)
        #             out_origin_tmp.loc[:, 'label'] = label
        #             ds = ds.append(out_origin_tmp.loc[:, ['address', 'label']], ignore_index=True)
        if i % 100 == 0:
            ds = ds.drop_duplicates()
            ds.to_csv('newDATASET.csv', index=False)

