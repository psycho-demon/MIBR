# =========================================================================
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

import torch
from torch import nn
from pandas.core.common import flatten
from .basemodel import BaseModel
from ..inputs import *
from ..layers import *
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from sklearn.metrics import *
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from tensorflow.python.keras.callbacks import CallbackList
except ImportError:
    from tensorflow.python.keras._impl.keras.callbacks import CallbackList

from ..inputs import build_input_features, SparseFeat, DenseFeat, VarLenSparseFeat, get_varlen_pooling_list, \
    create_embedding_matrix, varlen_embedding_lookup, embedding_lookup
from ..layers import PredictionLayer
from ..layers.utils import slice_arrays
from ..callbacks import History


class MIBR(BaseModel):
    def __init__(self,
                 dnn_feature_columns,
                 history_feature_list,
                 dnn_use_bn=False,
                 dnn_hidden_units=(200, 80), dnn_activation='prelu', att_hidden_size=(80, 40),
                 att_activation='sigmoid', att_weight_normalization=False, l2_reg_dnn=0.0,
                 l2_reg_embedding=1e-6, dnn_dropout=0, init_std=0.1,
                 seed=1024, task='binary', device='cpu', gpus=None,
                 short_long_length=(16, 256),
                 hash_bits=4,
                 retrieval=8,
                 ):
        super(MIBR, self).__init__([], dnn_feature_columns, l2_reg_linear=0, l2_reg_embedding=l2_reg_embedding,
                                   init_std=init_std, seed=seed, task=task, device=device, gpus=gpus)

        self.sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns)) if dnn_feature_columns else []
        self.varlen_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat), dnn_feature_columns)) if dnn_feature_columns else []

        self.history_feature_list = history_feature_list

        self.history_feature_columns = []
        self.sparse_varlen_feature_columns = []
        self.history_fc_names = list(map(lambda x: "hist_" + x, history_feature_list))

        for fc in self.varlen_sparse_feature_columns:
            feature_name = fc.name
            if feature_name in self.history_fc_names:
                self.history_feature_columns.append(fc)
            else:
                self.sparse_varlen_feature_columns.append(fc)

        self.att_emb_dim = self._compute_interest_dim()

        # actiavtion module
        self.attention = AttentionSequencePoolingLayer(att_hidden_units=att_hidden_size,
                                                       embedding_dim=self.att_emb_dim,
                                                       att_activation=att_activation,
                                                       return_score=False,
                                                       supports_masking=False,
                                                       weight_normalization=att_weight_normalization)
        # Final MLP
        self.dnn = DNN(inputs_dim=self.compute_input_dim(dnn_feature_columns),
                       hidden_units=dnn_hidden_units,
                       activation=dnn_activation,
                       dropout_rate=dnn_dropout,
                       l2_reg=l2_reg_dnn,
                       use_bn=dnn_use_bn)
        self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False).to(device)
        self.short_length = short_long_length[0]
        self.long_length = short_long_length[1]
        self.gru = nn.GRU(self.att_emb_dim, self.att_emb_dim, batch_first=True).to(device)

        # SimHash related
        self.hash_bits = hash_bits
        self.retrieval = retrieval
        self.random_rotation = nn.Parameter(torch.randn(self.att_emb_dim, self.hash_bits), requires_grad=False)

        # multi-granularity interest refinement module
        self.fft_block = MIRM(self.att_emb_dim * 3, num_channel=3)
        self.aspect_weight = nn.Parameter(torch.randn(3, 1), requires_grad=True)
        self.to(device)

    def forward(self, X):
        _, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns, self.embedding_dict)

        # sequence pooling part
        query_emb_list = embedding_lookup(X, self.embedding_dict, self.feature_index, self.sparse_feature_columns,
                                          return_feat_list=self.history_feature_list, to_list=True)
        keys_emb_list = embedding_lookup(X, self.embedding_dict, self.feature_index, self.history_feature_columns,
                                         return_feat_list=self.history_fc_names, to_list=True)
        dnn_input_emb_list = embedding_lookup(X, self.embedding_dict, self.feature_index, self.sparse_feature_columns,
                                              to_list=True)

        sequence_embed_dict = varlen_embedding_lookup(X, self.embedding_dict, self.feature_index,
                                                      self.sparse_varlen_feature_columns)

        sequence_embed_list = get_varlen_pooling_list(sequence_embed_dict, X, self.feature_index,
                                                      self.sparse_varlen_feature_columns, self.device)

        dnn_input_emb_list += sequence_embed_list
        deep_input_emb = torch.cat(dnn_input_emb_list, dim=-1)

        # concatenate
        query_emb = torch.cat(query_emb_list, dim=-1)  # [B, 1, E]
        keys_emb = torch.cat(keys_emb_list, dim=-1)  # [B, T, E]
        B, T, E = keys_emb.size()

        keys_length_feature_name = [feat.length_name for feat in self.varlen_sparse_feature_columns if
                                    feat.length_name is not None]
        keys_length = torch.squeeze(maxlen_lookup(X, self.feature_index, keys_length_feature_name), 1)  # [B, 1]
        keys_masks = torch.arange(T, device=keys_length.device, dtype=keys_length.dtype).repeat(B, 1)  # [B, T]
        keys_length = T - keys_length  # update
        keys_masks = keys_masks >= keys_length.view(-1, 1)  # 0, 1 mask  # update  [B, T]

        # short
        short_keys_length = torch.where(keys_length > self.short_length, self.short_length, keys_length)
        short_keys_emb = keys_emb[:, -self.short_length:, :]

        # long
        long_keys_length = keys_length - self.short_length
        long_keys_length = torch.where(long_keys_length > 0, long_keys_length, 0)
        long_keys_emb = keys_emb[:, :-self.short_length, :]

        # target-aware search unit
        query_target = query_emb
        token1, _ = self.search_topk_by_lsh(query_target, keys_emb, keys_length)

        # local-aware search unit
        query_local = self.query_local_modeling(query_emb, short_keys_emb)
        token2, _ = self.search_topk_by_lsh(query_local, keys_emb, keys_length)

        # global-aware search unit
        query_global = self.query_global_modeling(X, query_emb)
        token3, _ = self.search_topk_by_lsh(query_global, keys_emb, keys_length)

        # cat
        token_emb = torch.cat((token1, token2, token3), dim=-1)

        # refinement
        token_input = token_emb
        token_output = self.fft_block(token_input).reshape(B, self.retrieval, 3, E)
        token_emb = torch.matmul(token_output.permute(0, 1, 3, 2), self.aspect_weight).squeeze(-1)

        # din
        hist = self.attention(query_emb, token_emb, 3 * self.retrieval * torch.ones(B).to(self.device))

        # deep part
        deep_input_emb = torch.cat((deep_input_emb, hist), dim=-1)
        deep_input_emb = deep_input_emb.view(deep_input_emb.size(0), -1)

        dnn_input = combined_dnn_input([deep_input_emb], dense_value_list)
        dnn_output = self.dnn(dnn_input)
        dnn_logit = self.dnn_linear(dnn_output)

        y_pred = self.out(dnn_logit)
        return y_pred

    def query_local_modeling(self, query, key):
        B, T, E = key.size()
        key = torch.cat((key, query), dim=1)
        order_dependent = torch.mean(key, dim=1, keepdim=True)
        output, hn = self.gru(key)
        order_sensitive = hn.reshape(B, 1, -1)
        new_query = (order_dependent + order_sensitive) * 0.5
        return new_query

    def query_global_modeling(self, X, query_emb):
        # X: [B, uid + iid + other feas]
        item_ids = X[:, [1]].squeeze(-1).long()  # [B]

        center_ids = self.assign[item_ids]  # [B]
        center_emb = self.C[center_ids]  # [B, E]
        center_emb = center_emb.unsqueeze(1)

        embedding_dim = self.embedding_dict['item'].embedding_dim
        cate_emb = query_emb[:, :, embedding_dim:]
        new_query_emb = torch.cat((center_emb, cate_emb), dim=-1)
        return new_query_emb

    def _compute_interest_dim(self):
        interest_dim = 0
        for feat in self.sparse_feature_columns:
            if feat.name in self.history_feature_list:
                interest_dim += feat.embedding_dim
        return interest_dim

    def search_topk_by_lsh(self, query, key, length):
        def lsh(input):
            input_hash = input @ self.random_rotation
            input_hash = torch.relu(torch.sign(input_hash))
            return input_hash

        query_hash = lsh(query)
        key_hash = lsh(key)
        hamming_distance = (key_hash - query_hash).abs().sum(-1)
        distance_inf = self.hash_bits + 1

        max_length = self.long_length
        mask = torch.arange(max_length, device=self.device).repeat(len(length), 1)
        mask_length = max_length - length
        mask = mask >= mask_length.view(-1, 1)
        hamming_distance = torch.where(mask, hamming_distance, distance_inf)

        k = self.retrieval
        _, index = torch.topk(hamming_distance, k, dim=-1, largest=False)
        index = torch.sort(index)[0]

        topk_key = key.gather(1, index.unsqueeze(-1).expand(index.shape[0], index.shape[1], key.shape[-1]))
        topk_length = (index >= mask_length.unsqueeze(dim=-1)).sum(-1)
        return topk_key, topk_length

    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, initial_epoch=0, validation_split=0.,
            validation_data=None, shuffle=True, callbacks=None):
        """

        :param x: Numpy array of training data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs).If input layers in the model are named, you can also pass a
            dictionary mapping input names to Numpy arrays.
        :param y: Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs).
        :param batch_size: Integer or `None`. Number of samples per gradient update. If unspecified, `batch_size` will default to 256.
        :param epochs: Integer. Number of epochs to train the model. An epoch is an iteration over the entire `x` and `y` data provided. Note that in conjunction with `initial_epoch`, `epochs` is to be understood as "final epoch". The model is not trained for a number of iterations given by `epochs`, but merely until the epoch of index `epochs` is reached.
        :param verbose: Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
        :param initial_epoch: Integer. Epoch at which to start training (useful for resuming a previous training run).
        :param validation_split: Float between 0 and 1. Fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch. The validation data is selected from the last samples in the `x` and `y` data provided, before shuffling.
        :param validation_data: tuple `(x_val, y_val)` or tuple `(x_val, y_val, val_sample_weights)` on which to evaluate the loss and any model metrics at the end of each epoch. The model will not be trained on this data. `validation_data` will override `validation_split`.
        :param shuffle: Boolean. Whether to shuffle the order of the batches at the beginning of each epoch.
        :param callbacks: List of `deepctr_torch.callbacks.Callback` instances. List of callbacks to apply during training and validation (if ). See [callbacks](https://tensorflow.google.cn/api_docs/python/tf/keras/callbacks). Now available: `EarlyStopping` , `ModelCheckpoint`

        :return: A `History` object. Its `History.history` attribute is a record of training loss values and metrics values at successive epochs, as well as validation loss values and validation metrics values (if applicable).
        """
        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]

        do_validation = False
        if validation_data:
            do_validation = True
            if len(validation_data) == 2:
                val_x, val_y = validation_data
                val_sample_weight = None
            elif len(validation_data) == 3:
                val_x, val_y, val_sample_weight = validation_data  # pylint: disable=unpacking-non-sequence
            else:
                raise ValueError(
                    'When passing a `validation_data` argument, '
                    'it must contain either 2 items (x_val, y_val), '
                    'or 3 items (x_val, y_val, val_sample_weights), '
                    'or alternatively it could be a dataset or a '
                    'dataset or a dataset iterator. '
                    'However we received `validation_data=%s`' % validation_data)
            if isinstance(val_x, dict):
                val_x = [val_x[feature] for feature in self.feature_index]

        elif validation_split and 0. < validation_split < 1.:
            do_validation = True
            if hasattr(x[0], 'shape'):
                split_at = int(x[0].shape[0] * (1. - validation_split))
            else:
                split_at = int(len(x[0]) * (1. - validation_split))
            x, val_x = (slice_arrays(x, 0, split_at),
                        slice_arrays(x, split_at))
            y, val_y = (slice_arrays(y, 0, split_at),
                        slice_arrays(y, split_at))

        else:
            val_x = []
            val_y = []
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)

        train_tensor_data = Data.TensorDataset(
            torch.from_numpy(
                np.concatenate(x, axis=-1)),
            torch.from_numpy(y))
        if batch_size is None:
            batch_size = 256

        model = self.train()
        loss_func = self.loss_func
        optim = self.optim

        if self.gpus:
            print('parallel running on these gpus:', self.gpus)
            model = torch.nn.DataParallel(model, device_ids=self.gpus)
            batch_size *= len(self.gpus)  # input `batch_size` is batch_size per gpu
        else:
            print(self.device)

        train_loader = DataLoader(
            dataset=train_tensor_data, shuffle=shuffle, batch_size=batch_size)

        sample_num = len(train_tensor_data)
        steps_per_epoch = (sample_num - 1) // batch_size + 1

        # configure callbacks
        callbacks = (callbacks or []) + [self.history]  # add history callback
        callbacks = CallbackList(callbacks)
        callbacks.set_model(self)
        callbacks.on_train_begin()
        callbacks.set_model(self)
        if not hasattr(callbacks, 'model'):  # for tf1.4
            callbacks.__setattr__('model', self)
        callbacks.model.stop_training = False

        # Train
        print("Train on {0} samples, validate on {1} samples, {2} steps per epoch".format(
            len(train_tensor_data), len(val_y), steps_per_epoch))
        for epoch in range(initial_epoch, epochs):
            callbacks.on_epoch_begin(epoch)
            epoch_logs = {}
            start_time = time.time()
            loss_epoch = 0
            total_loss_epoch = 0
            train_result = {}
            try:
                with tqdm(enumerate(train_loader), disable=False) as t:
                    for iter, (x_train, y_train) in t:
                        x = x_train.to(self.device).float()
                        y = y_train.to(self.device).float()

                        if iter == 0:
                            # cluster
                            item_emb = self.embedding_dict["item"]
                            item_len = item_emb.num_embeddings
                            item_emb = item_emb(torch.LongTensor(range(item_len)).to(self.device))
                            self.C, self.assign, assign_m, kmeans_loss = self.cluster(item_emb, 500, 100)

                        y_pred = model(x).squeeze()

                        optim.zero_grad()
                        if isinstance(loss_func, list):
                            assert len(loss_func) == self.num_tasks, \
                                "the length of `loss_func` should be equal with `self.num_tasks`"
                            loss = sum(
                                [loss_func[i](y_pred[:, i], y[:, i], reduction='sum') for i in range(self.num_tasks)])
                        else:
                            loss = loss_func(y_pred, y.squeeze(), reduction='sum')
                        reg_loss = self.get_regularization_loss()

                        total_loss = loss + reg_loss + self.aux_loss

                        loss_epoch += loss.item()
                        total_loss_epoch += total_loss.item()
                        total_loss.backward()
                        optim.step()

                        if verbose > 0:
                            for name, metric_fun in self.metrics.items():
                                if name not in train_result:
                                    train_result[name] = []
                                train_result[name].append(metric_fun(
                                    y.cpu().data.numpy(), y_pred.cpu().data.numpy().astype("float64")))

            except KeyboardInterrupt:
                t.close()
                raise
            t.close()

            # Add epoch_logs
            epoch_logs["loss"] = total_loss_epoch / sample_num
            for name, result in train_result.items():
                epoch_logs[name] = np.sum(result) / steps_per_epoch

            if do_validation:
                eval_result = self.evaluate(val_x, val_y, batch_size)
                for name, result in eval_result.items():
                    epoch_logs["val_" + name] = result
            # verbose
            if verbose > 0:
                epoch_time = int(time.time() - start_time)
                print('Epoch {0}/{1}'.format(epoch + 1, epochs))

                eval_str = "{0}s - loss: {1: .4f}".format(
                    epoch_time, epoch_logs["loss"])

                for name in self.metrics:
                    eval_str += " - " + name + \
                                ": {0: .4f}".format(epoch_logs[name])

                if do_validation:
                    for name in self.metrics:
                        eval_str += " - " + "val_" + name + \
                                    ": {0: .4f}".format(epoch_logs["val_" + name])
                print(eval_str)
            callbacks.on_epoch_end(epoch, epoch_logs)
            if self.stop_training:
                break

        callbacks.on_train_end()
        if self.save_param:
            torch.save(self.embedding_dict, "../data/taobao/embedding.pth")

        return self.history

    def cluster(self, X, K_or_center=100, max_iter=30, verbose=False):
        N = X.size(0)
        if isinstance(K_or_center, int):
            K = K_or_center
            C = X[torch.randperm(N)[:K]]
        else:
            K = K_or_center.size(0)
            C = K_or_center
        prev_loss = np.inf
        for iter in range(max_iter):
            # dist = torch.sum(X * X, dim=-1, keepdim=True) - 2 * (X @ C.T) + torch.sum(C * C, dim=-1).unsqueeze(0)
            dist = torch.cdist(X, C)
            assign = dist.argmin(-1)
            assign_m = torch.zeros(N, K).to(self.device)
            assign_m[(range(N), assign)] = 1
            loss = torch.sum(torch.square(X - C[assign, :])).item()
            if verbose:
                print(f'step:{iter:<3d}, loss:{loss:.3f}')
            if (prev_loss - loss) < prev_loss * 1e-3:
                break
            prev_loss = loss
            cluster_count = assign_m.sum(0)
            C = (assign_m.T @ X) / cluster_count.unsqueeze(-1)
            empty_idx = cluster_count < .5
            ndead = empty_idx.sum().item()
            C[empty_idx] = X[torch.randperm(N)[:ndead]]
        return C, assign, assign_m, loss


class MIRM(nn.Module):
    def __init__(self, hidden_size, num_channel=3, sparsity_threshold=0.01):
        super().__init__()
        assert hidden_size % num_channel == 0, f"hidden_size {hidden_size} should be divisble by num_blocks {num_blocks}"

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_channel
        self.block_size = self.hidden_size // self.num_blocks
        self.scale = 0.02
        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

    def forward(self, x):
        bias = x
        dtype = x.dtype
        x = x.float()
        B, L, E = x.shape
        k = self.num_blocks
        e = E // k
        x = x.reshape(B, L, k, e)
        x = torch.fft.rfft(x, dim=1, norm="ortho")  # 对L做FFT变换
        # 0初始化线性组合后的结果，后面可以看出0初始化只需要部分赋值即可实现截断
        o1_real = torch.zeros([B, x.shape[1], k, e], device=x.device)
        o1_imag = torch.zeros([B, x.shape[1], k, e], device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        o1_real = F.relu(
            torch.einsum('...bi,bio->...bo', x.real, self.w1[0]) - \
            torch.einsum('...bi,bio->...bo', x.imag, self.w1[1]) + \
            self.b1[0]
        )

        o1_imag = F.relu(
            torch.einsum('...bi,bio->...bo', x.imag, self.w1[0]) + \
            torch.einsum('...bi,bio->...bo', x.real, self.w1[1]) + \
            self.b1[1]
        )

        o2_real = (
                torch.einsum('...bi,bio->...bo', o1_real, self.w2[0]) - \
                torch.einsum('...bi,bio->...bo', o1_imag, self.w2[1]) + \
                self.b2[0]
        )

        o2_imag = (
                torch.einsum('...bi,bio->...bo', o1_imag, self.w2[0]) + \
                torch.einsum('...bi,bio->...bo', o1_real, self.w2[1]) + \
                self.b2[1]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(B, x.shape[1], E)
        x = torch.fft.irfft(x, n=L, dim=1, norm='ortho')
        x = x.type(dtype)
        return x + bias
