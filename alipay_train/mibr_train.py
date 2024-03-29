import sys

sys.path.insert(0, '..')
import pickle as pkl
import pandas as pd
import random
import numpy as np
import torch
from deepctr_torch.inputs import (DenseFeat, SparseFeat, VarLenSparseFeat,
                                  get_feature_names)
from deepctr_torch.models.din import DIN
from deepctr_torch.models.mibr import MIBR

if __name__ == "__main__":
    random.seed(2023)
    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'
    max_sequence_len = 256
    batch_size = 256

    train_feature_dict = np.load('../data/alipay/alipay_train.npy', allow_pickle=True).item()
    test_feature_dict = np.load('../data/alipay/alipay_test.npy', allow_pickle=True).item()
    [user_length, item_length, cate_length, seller_length] = np.load('../data/alipay/alipay_len_data.npy',
                                                                     allow_pickle=True)

    behavior_feature_list = ["item", "item_cate", "seller"]
    feature_columns = []
    feature_columns += [SparseFeat('user', user_length + 1, embedding_dim=16),
                        SparseFeat('item', item_length + 1, embedding_dim=16),
                        SparseFeat('item_cate', cate_length + 1, embedding_dim=16),
                        SparseFeat('seller', seller_length + 1, embedding_dim=16)]
    feature_columns += [VarLenSparseFeat(SparseFeat('hist_item', item_length + 1, embedding_dim=16), max_sequence_len,
                                         length_name="seq_length"),
                        VarLenSparseFeat(SparseFeat('hist_item_cate', cate_length + 1, embedding_dim=16),
                                         max_sequence_len,
                                         length_name="seq_length"),
                        VarLenSparseFeat(SparseFeat('hist_seller', seller_length + 1, embedding_dim=16),
                                         max_sequence_len,
                                         length_name="seq_length")]

    train_x = {name: train_feature_dict[name] for name in get_feature_names(feature_columns)}
    train_y = train_feature_dict['label_array']
    test_x = {name: test_feature_dict[name] for name in get_feature_names(feature_columns)}
    test_y = test_feature_dict['label_array']
    print(f'train_len: {len(train_y)}')
    print(f'test_len: {len(test_y)}')

    model = MIBR(feature_columns, behavior_feature_list, device=device, att_weight_normalization=True, dnn_use_bn=True,
                 short_long_length=(16, 256), hash_bits=32, retrieval=16)
    model.compile('adam', 'binary_crossentropy',
                  metrics=['auc', 'logloss'])

    history = model.fit(train_x, train_y, batch_size=batch_size, epochs=1, verbose=2,
                        validation_data=(test_x, test_y), shuffle=True)


