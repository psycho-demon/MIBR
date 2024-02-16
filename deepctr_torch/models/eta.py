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


class ETA(BaseModel):
    def __init__(self,
                 dnn_feature_columns,
                 history_feature_list,
                 short_long_length=(16, 256),
                 hash_bits=8,
                 retrieval_k=8,
                 dnn_use_bn=False,
                 dnn_hidden_units=(200, 80), dnn_activation='prelu', att_hidden_size=(80, 40),
                 att_activation='sigmoid', att_weight_normalization=False, l2_reg_dnn=0.0,
                 l2_reg_embedding=1e-6, dnn_dropout=0, init_std=0.1,
                 seed=1024, task='binary', device='cpu', gpus=None):
        super(ETA, self).__init__([], dnn_feature_columns, l2_reg_linear=0, l2_reg_embedding=l2_reg_embedding,
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

        att_emb_dim = self._compute_interest_dim()
        self.short_attention = AttentionSequencePoolingLayer(att_hidden_units=att_hidden_size,
                                                             embedding_dim=att_emb_dim,
                                                             att_activation=att_activation,
                                                             return_score=False,
                                                             supports_masking=False,
                                                             weight_normalization=att_weight_normalization)
        self.long_attention = AttentionSequencePoolingLayer(att_hidden_units=att_hidden_size,
                                                            embedding_dim=att_emb_dim,
                                                            att_activation=att_activation,
                                                            return_score=False,
                                                            supports_masking=False,
                                                            weight_normalization=att_weight_normalization)

        self.dnn = DNN(inputs_dim=self.compute_input_dim(dnn_feature_columns) + att_emb_dim,
                       hidden_units=dnn_hidden_units,
                       activation=dnn_activation,
                       dropout_rate=dnn_dropout,
                       l2_reg=l2_reg_dnn,
                       use_bn=dnn_use_bn)
        self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False).to(device)
        self.short_length = short_long_length[0]
        self.long_length = short_long_length[1]

        # LSH related
        self.hash_bits = hash_bits
        self.retrieval_k = retrieval_k
        self.random_rotation = nn.Parameter(torch.randn(att_emb_dim, self.hash_bits), requires_grad=False)
        self.to(device)

    def search_topk_by_lsh(self, query, key, length):
        def lsh(input):
            input_hash = input @ self.random_rotation
            input_hash = torch.relu(torch.sign(input_hash))
            return input_hash

        query_hash = lsh(query)
        key_hash = lsh(key)
        hamming_distance = (key_hash - query_hash).abs().sum(-1)
        distance_inf = self.hash_bits + 1

        max_length = self.long_length - self.short_length
        mask_hamming = torch.arange(max_length, device=self.device).repeat(len(length), 1)
        mask_length = max_length - length
        mask_hamming = mask_hamming >= mask_length.view(-1, 1)
        hamming_distance = torch.where(mask_hamming, hamming_distance, distance_inf)

        k = self.retrieval_k
        _, index = torch.topk(hamming_distance, k, dim=-1, largest=False)
        index = torch.sort(index)[0]

        topk_key = key.gather(1, index.unsqueeze(-1).expand(index.shape[0], index.shape[1], key.shape[-1]))
        topk_length = (index >= mask_length.unsqueeze(dim=-1)).sum(-1)
        return topk_key, topk_length

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

        keys_length_feature_name = [feat.length_name for feat in self.varlen_sparse_feature_columns if
                                    feat.length_name is not None]
        keys_length = torch.squeeze(maxlen_lookup(X, self.feature_index, keys_length_feature_name), 1)  # [B, 1]

        short_keys_length = torch.where(keys_length > self.short_length, self.short_length, keys_length)
        short_keys_emb = keys_emb[:, -self.short_length:, :]
        long_keys_length = keys_length - self.short_length
        long_keys_length = torch.where(long_keys_length > 0, long_keys_length, 0)
        long_keys_emb = keys_emb[:, :-self.short_length, :]

        # short
        short_hist = self.short_attention(query_emb, short_keys_emb, short_keys_length)

        # long
        long_keys_emb_topk, long_keys_length_topk = self.search_topk_by_lsh(query_emb, long_keys_emb, long_keys_length)
        long_hist = self.long_attention(query_emb, long_keys_emb_topk, long_keys_length_topk)

        # deep part
        deep_input_emb = torch.cat((deep_input_emb, short_hist, long_hist), dim=-1)
        deep_input_emb = deep_input_emb.view(deep_input_emb.size(0), -1)

        dnn_input = combined_dnn_input([deep_input_emb], dense_value_list)
        dnn_output = self.dnn(dnn_input)
        dnn_logit = self.dnn_linear(dnn_output)

        y_pred = self.out(dnn_logit)
        return y_pred

    def _compute_interest_dim(self):
        interest_dim = 0
        for feat in self.sparse_feature_columns:
            if feat.name in self.history_feature_list:
                interest_dim += feat.embedding_dim
        return interest_dim
