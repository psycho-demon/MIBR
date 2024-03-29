import sys

sys.path.insert(0, '..')
import pickle as pkl
import pandas as pd
import random
import numpy as np
import torch
from deepctr_torch.inputs import (DenseFeat, SparseFeat, VarLenSparseFeat,
                                  get_feature_names)

RAW_DATA_FILE = '../data/alipay/ijcai2016_taobao.csv'


def to_df(file_name):
    df = pd.read_csv(RAW_DATA_FILE)
    # df = pd.read_csv(RAW_DATA_FILE, nrows=100000)
    df = df[df['act_ID'] == 0]
    print(len(df))
    return df


def remap(df, MAX_LEN_ITEM):
    # 特征id化
    user_key = sorted(df['use_ID'].unique().tolist())
    user_len = len(user_key)
    user_map = dict(zip(user_key, range(1, user_len + 1)))
    df['use_ID'] = df['use_ID'].map(lambda x: user_map[x])

    sel_key = sorted(df['sel_ID'].unique().tolist())
    sel_len = len(sel_key)
    sel_map = dict(zip(sel_key, range(1, sel_len + 1)))
    df['sel_ID'] = df['sel_ID'].map(lambda x: sel_map[x])

    item_key = sorted(df['ite_ID'].unique().tolist())
    item_len = len(item_key)
    item_map = dict(zip(item_key, range(1, item_len + 1)))
    df['ite_ID'] = df['ite_ID'].map(lambda x: item_map[x])

    cate_key = sorted(df['cat_ID'].unique().tolist())
    cate_len = len(cate_key)
    cate_map = dict(zip(cate_key, range(1, cate_len + 1)))
    df['cat_ID'] = df['cat_ID'].map(lambda x: cate_map[x])

    btag_key = sorted(df['act_ID'].unique().tolist())
    btag_len = len(btag_key)
    btag_map = dict(zip(btag_key, range(1, btag_len + 1)))
    df['act_ID'] = df['act_ID'].map(lambda x: btag_map[x])

    print("remap completed")
    print(f"The number of users:{user_len}")
    print(f"The number of items:{item_len}")
    print(f"The number of cates:{cate_len}")
    print(f"The number of sellers:{sel_len}")
    print(f"The number of btags:{btag_len}")

    len_data = (user_len, item_len, cate_len, sel_len)
    if MAX_LEN_ITEM == 256:
        np.save('../data/alipay/alipay_256_len_data.npy', len_data)
    if MAX_LEN_ITEM == 16:
        np.save('../data/alipay/alipay_16_len_data.npy', len_data)
    if MAX_LEN_ITEM == 128:
        np.save('../data/alipay/alipay_128_len_data.npy', len_data)

    return df, user_len, item_len, cate_len, sel_len


def gen_user_item_group(df):
    # 根据uid、time排序， uid分组
    # 根据iid、time排序， iid分组
    user_df = df.sort_values(['use_ID', 'time']).groupby('use_ID')
    item_df = df.sort_values(['ite_ID', 'time']).groupby('ite_ID')

    print("group completed")
    return user_df, item_df


def gen_dataset(user_df, item_df, item_cnt, MAX_LEN_ITEM):
    # uid + seller + target_item + target_item_cate + label + item_list + cat_list
    train_uid_array = []
    train_iid_array = []
    train_icate_array = []
    train_seller_array = []
    train_label_array = []
    train_hist_iid_array = []
    train_hist_icate_array = []
    train_hist_seller_array = []
    train_behavior_length = []

    test_uid_array = []
    test_iid_array = []
    test_icate_array = []
    test_seller_array = []
    test_label_array = []
    test_hist_iid_array = []
    test_hist_icate_array = []
    test_hist_seller_array = []
    test_behavior_length = []

    cnt = 0
    for uid, hist in user_df:
        if len(hist) < 5:
            continue
        cnt += 1
        item_hist = hist['ite_ID'].tolist()
        cate_hist = hist['cat_ID'].tolist()
        seller_hist = hist['sel_ID'].tolist()

        target_item_time = hist['time'].tolist()[-1]
        target_item = item_hist[-1]
        target_item_cate = cate_hist[-1]
        target_seller = seller_hist[-1]
        test = random.random() > 0.7
        label = 1

        # neg sampling
        neg = random.randint(0, 1)
        # 50%概率为负样本
        if neg == 1:
            label = 0
            while target_item == item_hist[-1]:
                target_item = random.randint(1, item_cnt)
                target_item_cate = item_df.get_group(target_item)['cat_ID'].tolist()[0]
                target_seller = item_df.get_group(target_item)['sel_ID'].tolist()[0]

        # the item history part of the sample
        item_part = []
        for i in range(len(item_hist) - 1):
            item_part.append([uid, item_hist[i], cate_hist[i], seller_hist[i]])

        behavoir_len = min(len(item_part), MAX_LEN_ITEM)

        # padding history with 0
        if len(item_part) <= MAX_LEN_ITEM:
            item_part_pad = [[0] * 4] * (MAX_LEN_ITEM - len(item_part)) + item_part
        else:
            item_part_pad = item_part[len(item_part) - MAX_LEN_ITEM:len(item_part)]

        if test:
            cat_list = []
            item_list = []
            seller_list = []
            for i in range(len(item_part_pad)):
                item_list.append(item_part_pad[i][1])
                cat_list.append(item_part_pad[i][2])
                seller_list.append(item_part_pad[i][3])

            test_uid_array.append(uid)
            test_iid_array.append(target_item)
            test_icate_array.append(target_item_cate)
            test_seller_array.append(target_seller)
            test_label_array.append(label)
            test_hist_iid_array.append(item_list)
            test_hist_icate_array.append(cat_list)
            test_hist_seller_array.append(seller_list)
            test_behavior_length.append(behavoir_len)
        else:
            cat_list = []
            item_list = []
            seller_list = []
            for i in range(len(item_part_pad)):
                item_list.append(item_part_pad[i][1])
                cat_list.append(item_part_pad[i][2])
                seller_list.append(item_part_pad[i][3])

            train_uid_array.append(uid)
            train_iid_array.append(target_item)
            train_icate_array.append(target_item_cate)
            train_seller_array.append(target_seller)
            train_label_array.append(label)
            train_hist_iid_array.append(item_list)
            train_hist_icate_array.append(cat_list)
            train_hist_seller_array.append(seller_list)
            train_behavior_length.append(behavoir_len)

    train_uid_array = np.array(train_uid_array)
    train_iid_array = np.array(train_iid_array)
    train_icate_array = np.array(train_icate_array)
    train_seller_array = np.array(train_seller_array)
    train_label_array = np.array(train_label_array)
    train_hist_iid_array = np.array(train_hist_iid_array)
    train_hist_icate_array = np.array(train_hist_icate_array)
    train_hist_seller_array = np.array(train_hist_seller_array)
    train_behavior_length = np.array(train_behavior_length)

    test_uid_array = np.array(test_uid_array)
    test_iid_array = np.array(test_iid_array)
    test_icate_array = np.array(test_icate_array)
    test_seller_array = np.array(test_seller_array)
    test_label_array = np.array(test_label_array)
    test_hist_iid_array = np.array(test_hist_iid_array)
    test_hist_icate_array = np.array(test_hist_icate_array)
    test_hist_seller_array = np.array(test_hist_seller_array)
    test_behavior_length = np.array(test_behavior_length)

    train_feature_dict = {'user': train_uid_array, 'item': train_iid_array, 'item_cate': train_icate_array,
                          'hist_item': train_hist_iid_array, 'hist_item_cate': train_hist_icate_array,
                          "seq_length": train_behavior_length,
                          "label_array": train_label_array,
                          'seller': train_seller_array, 'hist_seller': train_hist_seller_array}
    # train_x = {name: train_feature_dict[name] for name in get_feature_names(feature_columns)}
    # train_y = train_label_array

    test_feature_dict = {'user': test_uid_array, 'item': test_iid_array, 'item_cate': test_icate_array,
                         'hist_item': test_hist_iid_array, 'hist_item_cate': test_hist_icate_array,
                         "seq_length": test_behavior_length,
                         "label_array": test_label_array,
                         'seller': test_seller_array, 'hist_seller': test_hist_seller_array}
    # test_x = {name: test_feature_dict[name] for name in get_feature_names(feature_columns)}
    # test_y = test_label_array

    print(f"The number of users (behavior > 5):{cnt}")
    print("train, valid, test sample completed")

    if MAX_LEN_ITEM == 256:
        np.save('../data/alipay/alipay_256_train.npy', train_feature_dict)
        np.save('../data/alipay/alipay_256_test.npy', test_feature_dict)
    if MAX_LEN_ITEM == 16:
        np.save('../data/alipay/alipay_16_train.npy', train_feature_dict)
        np.save('../data/alipay/alipay_16_test.npy', test_feature_dict)
    if MAX_LEN_ITEM == 128:
        np.save('../data/alipay/alipay_128_train.npy', train_feature_dict)
        np.save('../data/alipay/alipay_128_test.npy', test_feature_dict)


if __name__ == "__main__":
    random.seed(2023)
    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:1'
    MAX_LEN_ITEM = 16
    print(f'MAX_LEN_ITEM: {MAX_LEN_ITEM}')

    df = to_df(RAW_DATA_FILE)
    df, user_len, item_len, cate_len, seller_len = remap(df, MAX_LEN_ITEM)
    user_df, item_df = gen_user_item_group(df)
    gen_dataset(user_df, item_df, item_len, MAX_LEN_ITEM)
