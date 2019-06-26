#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import sklearn
import sklearn.externals.joblib as joblib
from multiprocessing import *
import multiprocessing.sharedctypes as sharedctypes
import ctypes

def fun(df, ns):
    train_df = ns.df2
    vin_df = ns.df1
    alist = ns.vlist
    m[0] = 1
    m.append(20)
    find = True
    def findlabel(addr):
        address = addr['address']
        # print('1', addr._values[0], address, end=' ')
        tmp = train_df[train_df['address'].isin([address])]
        if not tmp.empty:
            # print('found in dataset!')
            return tmp.loc[:, 'label']._values[0]
        tmp_in = vin_df[vin_df['address_origin']==address]
        if not tmp_in.empty:
            # print('found in vin!')
            for t in tmp_in.index:
                tx_id = tmp_in.loc[t, 'tx_id']
                in_txid = vin_df[vin_df['tx_id'] == tx_id]
                addr_list = in_txid.loc[:, 'address_origin'].tolist()
                for al in set(addr_list):
                    al_in = train_df[train_df['address'] == al]
                    if not al_in.empty:
                        al_in = al_in.loc[:, 'label']
                        # print('found by searching vin', al_in._values[0])
                        return al_in._values[0]
            find = False
        else:
            # print('return directly!')
            print('1', addr._values[0], address)
            alist.append(addr._values[0])
            return 'services'
        if not find:
            # print('return directly!')
            print('1', addr._values[0], address)
            alist.append(addr._values[0])
            return 'services'

    df['label'] = df.apply(findlabel, axis=1)
    del train_df, vin_df
    df = df[['address', 'label']]
    df.to_csv('df1.csv', index=False)


def fun2(df, ns):
    train_df = ns.df2
    vin_df = ns.df1
    alist = ns.vlist
    find = True
    def findlabel(addr):
        address = addr['address']
        # print('2', addr._values[0], address, end=' ')
        tmp = train_df[train_df['address'].isin([address])]
        if not tmp.empty:
            #print('found in dataset!')
            return tmp.loc[:, 'label']._values[0]
        tmp_in = vin_df[vin_df['address_origin']==address]
        if not tmp_in.empty:
            # print('found in vin!', end=' ')
            for t in tmp_in.index:
                tx_id = tmp_in.loc[t, 'tx_id']
                in_txid = vin_df[vin_df['tx_id'] == tx_id]
                addr_list = in_txid.loc[:, 'address_origin'].tolist()
                for al in set(addr_list):
                    al_in = train_df[train_df['address'] == al]
                    if not al_in.empty:
                        al_in = al_in.loc[:, 'label']
                        # print('found by searching vin', al_in._values[0])
                        return al_in._values[0]
            find = False
        else:
            # print('return directly!')
            print('2', addr._values[0], address)
            alist.append(addr._values[0])
            return 'services'
        if not find:
            # print('return directly!')
            print('2', addr._values[0], address)
            alist.append(addr._values[0])
            return 'services'

    df['label'] = df.apply(findlabel, axis=1)
    del train_df, vin_df
    df = df[['address', 'label']]
    df.to_csv('df2.csv', index=False)


def fun3(df, ns):
    train_df = ns.df2
    vin_df = ns.df1
    alist = ns.vlist
    find = True
    def findlabel(addr):
        address = addr['address']
        # print('3', addr._values[0], address, end=' ')
        tmp = train_df[train_df['address'].isin([address])]
        if not tmp.empty:
            #print('found in dataset!')
            return tmp.loc[:, 'label']._values[0]
        tmp_in = vin_df[vin_df['address_origin']==address]
        if not tmp_in.empty:
            # print('found in vin!', end=' ')
            for t in tmp_in.index:
                tx_id = tmp_in.loc[t, 'tx_id']
                in_txid = vin_df[vin_df['tx_id'] == tx_id]
                addr_list = in_txid.loc[:, 'address_origin'].tolist()
                for al in set(addr_list):
                    al_in = train_df[train_df['address'] == al]
                    if not al_in.empty:
                        al_in = al_in.loc[:, 'label']
                        # print('found by searching vin', al_in._values[0])
                        return al_in._values[0]
            find = False
        else:
            # print('return directly!')
            print('3', addr._values[0], address)
            alist.append(addr._values[0])
            return 'services'
        if not find:
            # print('return directly!')
            print('3', addr._values[0], address)
            alist.append(addr._values[0])
            return 'services'

    df['label'] = df.apply(findlabel, axis=1)
    del train_df, vin_df
    df = df[['address', 'label']]
    df.to_csv('df3.csv', index=False)


def fun4(df, ns):
    train_df = ns.df2
    vin_df = ns.df1
    alist = ns.vlist
    find = True
    def findlabel(addr):
        address = addr['address']
        # print('4', addr._values[0], address, end=' ')
        tmp = train_df[train_df['address'].isin([address])]
        if not tmp.empty:
            # print('found in dataset!')
            return tmp.loc[:, 'label']._values[0]
        tmp_in = vin_df[vin_df['address_origin']==address]
        if not tmp_in.empty:
            # print('found in vin!', end=' ')
            for t in tmp_in.index:
                tx_id = tmp_in.loc[t, 'tx_id']
                in_txid = vin_df[vin_df['tx_id'] == tx_id]
                addr_list = in_txid.loc[:, 'address_origin'].tolist()
                for al in set(addr_list):
                    al_in = train_df[train_df['address'] == al]
                    if not al_in.empty:
                        al_in = al_in.loc[:, 'label']
                        # print('found by searching vin', al_in._values[0])
                        return al_in._values[0]
            find = False
        else:
            # print('return directly!')
            print('4', addr._values[0], address)
            alist.append(addr._values[0])
            return 'services'
        if not find:
            # print('return directly!')
            print('4', addr._values[0], address)
            alist.append(addr._values[0])
            return 'services'

    df['label'] = df.apply(findlabel, axis=1)
    del train_df, vin_df
    df = df[['address', 'label']]
    df.to_csv('df4.csv', index=False)

def fun5(df, ns):
    train_df = ns.df2
    vin_df = ns.df1
    alist = ns.vlist
    find = True
    def findlabel(addr):
        address = addr['address']
        # print('5', addr._values[0], address, end=' ')
        tmp = train_df[train_df['address'].isin([address])]
        if not tmp.empty:
            # print('found in dataset!')
            return tmp.loc[:, 'label']._values[0]
        tmp_in = vin_df[vin_df['address_origin']==address]
        if not tmp_in.empty:
            # print('found in vin!', end=' ')
            for t in tmp_in.index:
                tx_id = tmp_in.loc[t, 'tx_id']
                in_txid = vin_df[vin_df['tx_id'] == tx_id]
                addr_list = in_txid.loc[:, 'address_origin'].tolist()
                for al in set(addr_list):
                    al_in = train_df[train_df['address'] == al]
                    if not al_in.empty:
                        al_in = al_in.loc[:, 'label']
                        # print('found by searching vin', al_in._values[0])
                        return al_in._values[0]
            find = False
        else:
            # print('return directly!')
            print('5', addr._values[0], address)
            alist.append(addr._values[0])
            return 'services'
        if not find:
            # print('return directly!')
            print('5', addr._values[0], address)
            alist.append(addr._values[0])
            return 'services'

    df['label'] = df.apply(findlabel, axis=1)
    del train_df, vin_df
    df = df[['address', 'label']]
    df.to_csv('df5.csv', index=False)

def fun6(df, ns):
    train_df = ns.df2
    vin_df = ns.df1
    alist = ns.vlist
    find = True
    def findlabel(addr):
        address = addr['address']
        # print('6', addr._values[0], address, end=' ')
        tmp = train_df[train_df['address'].isin([address])]
        if not tmp.empty:
            # print('found in dataset!')
            return tmp.loc[:, 'label']._values[0]
        tmp_in = vin_df[vin_df['address_origin']==address]
        if not tmp_in.empty:
            # print('found in vin!', end=' ')
            for t in tmp_in.index:
                tx_id = tmp_in.loc[t, 'tx_id']
                in_txid = vin_df[vin_df['tx_id'] == tx_id]
                addr_list = in_txid.loc[:, 'address_origin'].tolist()
                for al in set(addr_list):
                    al_in = train_df[train_df['address'] == al]
                    if not al_in.empty:
                        al_in = al_in.loc[:, 'label']
                        # print('found by searching vin', al_in._values[0])
                        return al_in._values[0]
            find = False
        else:
            # print('return directly!')
            print('6', addr._values[0], address)
            alist.append(addr._values[0])
            return 'services'
        if not find:
            # print('return directly!')
            print('6', addr._values[0], address)
            alist.append(addr._values[0])
            return 'services'

    df['label'] = df.apply(findlabel, axis=1)
    del train_df, vin_df
    df = df[['address', 'label']]
    df.to_csv('df6.csv', index=False)

def fun7(df, ns):
    train_df = ns.df2
    vin_df = ns.df1
    alist = ns.vlist
    find = True
    def findlabel(addr):
        address = addr['address']
        # print('7', addr._values[0], address, end=' ')
        tmp = train_df[train_df['address'].isin([address])]
        if not tmp.empty:
            # print('found in dataset!')
            return tmp.loc[:, 'label']._values[0]
        tmp_in = vin_df[vin_df['address_origin']==address]
        if not tmp_in.empty:
            # print('found in vin!', end=' ')
            for t in tmp_in.index:
                tx_id = tmp_in.loc[t, 'tx_id']
                in_txid = vin_df[vin_df['tx_id'] == tx_id]
                addr_list = in_txid.loc[:, 'address_origin'].tolist()
                for al in set(addr_list):
                    al_in = train_df[train_df['address'] == al]
                    if not al_in.empty:
                        al_in = al_in.loc[:, 'label']
                        # print('found by searching vin', al_in._values[0])
                        return al_in._values[0]
            find = False
        else:
            # print('return directly!')
            print('7', addr._values[0], address)
            alist.append(addr._values[0])
            return 'services'
        if not find:
            # print('return directly!')
            print('7', addr._values[0], address)
            alist.append(addr._values[0])
            return 'services'

    df['label'] = df.apply(findlabel, axis=1)
    del train_df, vin_df
    df = df[['address', 'label']]
    df.to_csv('df7.csv', index=False)

def fun8(df, ns):
    train_df = ns.df2
    vin_df = ns.df1
    alist = ns.vlist
    find = True
    def findlabel(addr):
        address = addr['address']
        # print('8', addr._values[0], address, end=' ')
        tmp = train_df[train_df['address'].isin([address])]
        if not tmp.empty:
            # print('found in dataset!')
            return tmp.loc[:, 'label']._values[0]
        tmp_in = vin_df[vin_df['address_origin']==address]
        if not tmp_in.empty:
            # print('found in vin!', end=' ')
            for t in tmp_in.index:
                tx_id = tmp_in.loc[t, 'tx_id']
                in_txid = vin_df[vin_df['tx_id'] == tx_id]
                addr_list = in_txid.loc[:, 'address_origin'].tolist()
                for al in set(addr_list):
                    al_in = train_df[train_df['address'] == al]
                    if not al_in.empty:
                        al_in = al_in.loc[:, 'label']
                        # print('found by searching vin', al_in._values[0])
                        return al_in._values[0]
            find = False
        else:
            # print('return directly!')
            print('7', addr._values[0], address)
            alist.append(addr._values[0])
            return 'services'
        if not find:
            # print('return directly!')
            print('7', addr._values[0], address)
            alist.append(addr._values[0])
            return 'services'

    df['label'] = df.apply(findlabel, axis=1)
    del train_df, vin_df
    df = df[['address', 'label']]
    df.to_csv('df8.csv', index=False)

def fun9(df, ns):
    train_df = ns.df2
    vin_df = ns.df1
    alist = ns.vlist
    find = True
    def findlabel(addr):
        address = addr['address']
        # print('9', addr._values[0], address, end=' ')
        tmp = train_df[train_df['address'].isin([address])]
        if not tmp.empty:
            # print('found in dataset!')
            return tmp.loc[:, 'label']._values[0]
        tmp_in = vin_df[vin_df['address_origin']==address]
        if not tmp_in.empty:
            # print('found in vin!', end=' ')
            for t in tmp_in.index:
                tx_id = tmp_in.loc[t, 'tx_id']
                in_txid = vin_df[vin_df['tx_id'] == tx_id]
                addr_list = in_txid.loc[:, 'address_origin'].tolist()
                for al in set(addr_list):
                    al_in = train_df[train_df['address'] == al]
                    if not al_in.empty:
                        al_in = al_in.loc[:, 'label']
                        # print('found by searching vin', al_in._values[0])
                        return al_in._values[0]
            find = False
        else:
            # print('return directly!')
            print('9', addr._values[0], address)
            alist.append(addr._values[0])
            return 'services'
        if not find:
            # print('return directly!')
            print('9', addr._values[0], address)
            alist.append(addr._values[0])
            return 'services'

    df['label'] = df.apply(findlabel, axis=1)
    del train_df, vin_df
    df = df[['address', 'label']]
    df.to_csv('df9.csv', index=False)

def fun10(df, ns):
    train_df = ns.df2
    vin_df = ns.df1
    alist = ns.vlist
    find = True
    def findlabel(addr):
        address = addr['address']
        # print('10', addr._values[0], address, end=' ')
        tmp = train_df[train_df['address'].isin([address])]
        if not tmp.empty:
            # print('found in dataset!')
            return tmp.loc[:, 'label']._values[0]
        tmp_in = vin_df[vin_df['address_origin']==address]
        if not tmp_in.empty:
            # print('found in vin!', end=' ')
            for t in tmp_in.index:
                tx_id = tmp_in.loc[t, 'tx_id']
                in_txid = vin_df[vin_df['tx_id'] == tx_id]
                addr_list = in_txid.loc[:, 'address_origin'].tolist()
                for al in set(addr_list):
                    al_in = train_df[train_df['address'] == al]
                    if not al_in.empty:
                        al_in = al_in.loc[:, 'label']
                        # print('found by searching vin', al_in._values[0])
                        return al_in._values[0]
            find = False
        else:
            # print('return directly!')
            print('10', addr._values[0], address)
            alist.append(addr._values[0])
            return 'services'
        if not find:
            # print('return directly!')
            print('10', addr._values[0], address)
            alist.append(addr._values[0])
            return 'services'

    df['label'] = df.apply(findlabel, axis=1)
    del train_df, vin_df
    df = df[['address', 'label']]
    df.to_csv('df10.csv', index=False)

def fun11(df, ns):
    train_df = ns.df2
    vin_df = ns.df1
    alist = ns.vlist
    find = True
    def findlabel(addr):
        address = addr['address']
        # print('11', addr._values[0], address, end=' ')
        tmp = train_df[train_df['address'].isin([address])]
        if not tmp.empty:
            # print('found in dataset!')
            return tmp.loc[:, 'label']._values[0]
        tmp_in = vin_df[vin_df['address_origin']==address]
        if not tmp_in.empty:
            # print('found in vin!', end=' ')
            for t in tmp_in.index:
                tx_id = tmp_in.loc[t, 'tx_id']
                in_txid = vin_df[vin_df['tx_id'] == tx_id]
                addr_list = in_txid.loc[:, 'address_origin'].tolist()
                for al in set(addr_list):
                    al_in = train_df[train_df['address'] == al]
                    if not al_in.empty:
                        al_in = al_in.loc[:, 'label']
                        # print('found by searching vin', al_in._values[0])
                        return al_in._values[0]
            find = False
        else:
            # print('return directly!')
            print('11', addr._values[0], address)
            alist.append(addr._values[0])
            return 'services'
        if not find:
            # print('return directly!')
            print('11', addr._values[0], address)
            alist.append(addr._values[0])
            return 'services'

    df['label'] = df.apply(findlabel, axis=1)
    del train_df, vin_df
    df = df[['address', 'label']]
    df.to_csv('df11.csv', index=False)

def fun12(df, ns):
    train_df = ns.df2
    vin_df = ns.df1
    alist = ns.vlist
    find = True
    def findlabel(addr):
        address = addr['address']
        # print('12', addr._values[0], address, end=' ')
        tmp = train_df[train_df['address'].isin([address])]
        if not tmp.empty:
            # print('found in dataset!')
            return tmp.loc[:, 'label']._values[0]
        tmp_in = vin_df[vin_df['address_origin']==address]
        if not tmp_in.empty:
            # print('found in vin!', end=' ')
            for t in tmp_in.index:
                tx_id = tmp_in.loc[t, 'tx_id']
                in_txid = vin_df[vin_df['tx_id'] == tx_id]
                addr_list = in_txid.loc[:, 'address_origin'].tolist()
                for al in set(addr_list):
                    al_in = train_df[train_df['address'] == al]
                    if not al_in.empty:
                        al_in = al_in.loc[:, 'label']
                        # print('found by searching vin', al_in._values[0])
                        return al_in._values[0]
            find = False
        else:
            # print('return directly!')
            print('12', addr._values[0], address)
            alist.append(addr._values[0])
            return 'services'
        if not find:
            # print('return directly!')
            print('12', addr._values[0], address)
            alist.append(addr._values[0])
            return 'services'

    df['label'] = df.apply(findlabel, axis=1)
    del train_df, vin_df
    df = df[['address', 'label']]
    df.to_csv('df12.csv', index=False)

if __name__=='__main__':

    import time
    from multiprocessing import Process, Array
    time_start = time.time()
    m = Array('i', 200)

    TEST_DATASET = r'test.csv'
    DataSet_DIR = 'train.csv'
    BTC_VIN_DIR = r'btc_vin.csv'

    mgr = Manager()
    ns = mgr.Namespace()

    print('Reading train data...')
    ns.df2 = pd.read_csv(DataSet_DIR)
    print('Reading test data...')
    test_df = pd.read_csv(TEST_DATASET)
    test_df.to_csv('rematch_test.csv')
    del test_df
    test_df = pd.read_csv('rematch_test.csv')
    print('Loading vin data...')
    vin_df = pd.read_csv(BTC_VIN_DIR)
    vin_df = vin_df[['tx_id', 'address_origin']]
    ns.df1 = vin_df
    del vin_df
    ns.vlist = []
    print('Loading all dataset completed!')

    df1 = test_df[0: 30000]
    df2 = test_df[30000: 60000]
    df3 = test_df[60000: 90000]
    df4 = test_df[90000: 120000]
    df5 = test_df[120000: 150000]
    df6 = test_df[150000: 180000]
    df7 = test_df[210000: 240000]
    df8 = test_df[240000: 270000]
    df9 = test_df[270000: 300000]
    df10 = test_df[300000: 330000]
    df11 = test_df[330000: 360000]
    df12 = test_df[360000:]
    del test_df
    # del vin_df, train_df, test_df

    print('begin multprocessing...')

    process1 = Process(target=fun, args=(df1, ns,))
    process2 = Process(target=fun2, args=(df2, ns,))
    process3 = Process(target=fun3, args=(df3, ns,))
    process4 = Process(target=fun4, args=(df4, ns,))
    process5 = Process(target=fun5, args=(df5, ns,))
    process6 = Process(target=fun6, args=(df6, ns,))
    process7 = Process(target=fun7, args=(df7, ns,))
    process8 = Process(target=fun8, args=(df8, ns,))
    process9 = Process(target=fun9, args=(df9, ns,))
    process10 = Process(target=fun10, args=(df10, ns,))
    process11 = Process(target=fun11, args=(df11, ns,))
    process12 = Process(target=fun12, args=(df12, ns,))

    datatime = time.time()-time_start

    process1.start()
    process2.start()
    process3.start()
    process4.start()
    process5.start()
    process6.start()
    process7.start()
    process8.start()
    process9.start()
    process10.start()
    process11.start()
    process12.start()

    process1.join()
    process2.join()
    process3.join()
    process4.join()
    process5.join()
    process6.join()
    process7.join()
    process8.join()
    process9.join()
    process10.join()
    process11.join()
    process12.join()

    print('data reading time (s)', datatime)
    print('classification time (s)', time.time()-time_start)
    vnp = np.array(m)
    np.save('vnp.npy', vnp)

    submission = 'rematch_test_submission_searchNOPREDICT.csv'
    result = pd.read_csv('df1.csv')
    result = result[['address', 'label']]
    result.to_csv(submission, index=False)
    csv_list = ['df2.csv', 'df3.csv', 'df4.csv', 'df5.csv', 'df6.csv', 'df7.csv', 'df8.csv', 'df9.csv', 'df10.csv', 'df11.csv', 'df12.csv']
    for inputfile in csv_list:
        a = pd.read_csv(inputfile) # header=None表示原始文件数据没有列索引，这样的话read_csv会自动加上列索引
        a = a[['address', 'label']]
        a.to_csv(submission, mode='a', index=False, header=False)
        # header=0表示不保留列名，index=False表示不保留行索引，mode='a'表示附加方式写入，文件原有内容不会被清除

    print('done!')





