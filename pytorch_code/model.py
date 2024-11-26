#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
import pickle
from metrics import FocalLoss
from entmax import entmax_bisect

from main import logger

directory = './'
spc = pickle.load(open('spc.txt', 'rb'))


class GNN(Module):
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden


class SessionSimilarityAggregation(Module):
    def __init__(self, hidden_size):
        super(SessionSimilarityAggregation, self).__init__()
        self.hidden_size = hidden_size
        self.topk = 3  # 每个session应该找到的最近邻的数量，不同的数据集有不同的设置，例如Diginetica:3; Tmall: 7; Nowplaying: 4
        self.dropout = nn.Dropout(0.40)  # 防止过拟合

    def compute_sim(self, sess_emb):  # 计算session之间的余弦相似度
        norm_emb = F.normalize(sess_emb, p=2, dim=1)  # [batch_size, hidden_size]
        cos_sim = torch.matmul(norm_emb, norm_emb.t())  # [batch_size, batch_size]
        return cos_sim

    # 返回每个session的最近邻的隐藏表示，其shape与sess_emb相同
    def forward(self, sess_emb):
        # 计算session之间的余弦相似度
        cos_sim = self.compute_sim(sess_emb)  # [batch_size, batch_size]
        # 获取每个session的Top K相似session
        topk = min(self.topk, cos_sim.size(0))
        cos_topk, topk_indice = torch.topk(cos_sim, k=topk, dim=1)  # 找到每个session的k_v 个最大余弦相似度和索引，分别存储在 cos_topk 和 topk_indice 中。
        # 对Top K相似度应用Softmax
        cos_topk = F.softmax(cos_topk, dim=1)  # [batch_size, K]
        sess_topk = sess_emb[topk_indice]  # 通过索引 topk_indice 从 sess_emb 中选出每个session的最近邻的隐藏表示。[batch_size, K, hidden_size]
        # 扩展相似度权重以匹配sess_topk的维度
        cos_sim = cos_topk.unsqueeze(2).expand(-1, -1, self.hidden_size)  # [batch_size, K, hidden_size]
        # 计算加权和
        sess_sim = torch.sum(cos_sim * sess_topk, dim=1)  # 计算每个session的最近邻的隐藏表示的加权和，即每个session的邻居的隐藏表示。[batch_size, hidden_size]
        sess_sim = self.dropout(sess_sim)  # [batch_size, hidden_size]
        return sess_sim

# 会话全局相似性学习模块
class GlobalSimilarityLearning(Module):
    def __init__(self, batch_size, hidden_size=100):
        super(GlobalSimilarityLearning, self).__init__()
        self.batch_size = batch_size
        self.dim = hidden_size
        # self.w_f = nn.Linear(2 * hidden_size, hidden_size)
        self.alpha_w = nn.Linear(self.dim, 1)
        self.atten_w0 = nn.Parameter(torch.Tensor(1, self.dim))
        self.atten_w1 = nn.Parameter(torch.Tensor(self.dim, self.dim))
        self.atten_w2 = nn.Parameter(torch.Tensor(self.dim, self.dim))
        self.atten_bias = nn.Parameter(torch.Tensor(self.dim))

    def get_alpha(self, x=None):
        # x[b,1,d]
        alpha_global = torch.sigmoid(self.alpha_w(x)) + 1  # [b,1,1]
        alpha_global = self.add_value(alpha_global)
        return alpha_global  # [b,1,1]

    def add_value(self, value):
        mask_value = (value == 1).float()
        value = value.masked_fill(mask_value == 1, 1.00001)
        return value

    def intention_attention_global(self, intention, k, v, alpha_ent=1):
        alpha = torch.matmul(
            torch.relu(k.matmul(self.atten_w1) + intention.matmul(self.atten_w2) + self.atten_bias),
            self.atten_w0.t()
        )
        alpha = entmax_bisect(alpha, alpha_ent, dim=1)
        c = torch.matmul(alpha.transpose(1, 2), v)
        return c

    # item_embedding是item的嵌入，items是item序列，A是会话全局相似性图的相似度矩阵，D是会话全局相似性图的度矩阵，intention_embedding是意图表示intention的嵌入
    def forward(self, item_embedding, items, A, D, intention_embedding):
        # 通过图卷积操作，利用相似性矩阵和度矩阵对会话序列嵌入聚合
        seq_h = []
        for i in torch.arange(items.shape[0]):  # items的shape是[batch_size, seq_length]
            seq_h.append(torch.index_select(item_embedding, 0, items[i]))  # [b,s,d]
        seq_h1 = trans_to_cuda(torch.tensor([item.cpu().detach().numpy() for item in seq_h]))  # 数组越界报错
        len = seq_h1.shape[1]
        similarity_emb_gcn = torch.sum(seq_h1, 1)  # [b,d]
        DA = torch.mm(D, A).float()  # [b,b]
        similarity_emb_gcn = torch.mm(DA, similarity_emb_gcn)  # [b,d]
        similarity_emb_gcn = similarity_emb_gcn.unsqueeze(1).expand(similarity_emb_gcn.shape[0], len,
                                                                similarity_emb_gcn.shape[1])  # [b,s,d]
        # 基于意图表示，利用意图注意力机制降噪
        intention_emb = intention_embedding
        alpha_line = self.get_alpha(x=intention_emb)
        q = intention_emb  # [b,1,d]
        k = similarity_emb_gcn  # [b,1,d]
        v = similarity_emb_gcn  # [b,1,d]

        line_c = self.intention_attention_global(q, k, v, alpha_ent=alpha_line)  # [b,1,d]
        c = torch.selu(line_c).squeeze()
        l_c = (c / torch.norm(c, dim=-1).unsqueeze(1))

        return l_c  # [b,d]


# 会话局部相似性学习模块
class LocalSimilarityLearning(Module):
    # 实例初始化方法
    def __init__(self, batch_size, hidden_size=100):
        super(LocalSimilarityLearning, self).__init__()
        self.batch_size = batch_size
        self.dim = hidden_size
        # self.w_f = nn.Linear(2 * hidden_size, hidden_size)
        self.alpha_w = nn.Linear(self.dim, 1)
        self.atten_w0 = nn.Parameter(torch.Tensor(1, self.dim))
        self.atten_w1 = nn.Parameter(torch.Tensor(self.dim, self.dim))
        self.atten_w2 = nn.Parameter(torch.Tensor(self.dim, self.dim))
        self.atten_bias = nn.Parameter(torch.Tensor(self.dim))

    def get_alpha(self, x=None):
        # x[b,1,d]
        alpha_global = torch.sigmoid(self.alpha_w(x)) + 1  # [b,1,1]
        alpha_global = self.add_value(alpha_global)
        return alpha_global  # [b,1,1]

    def add_value(self, value):
        mask_value = (value == 1).float()
        value = value.masked_fill(mask_value == 1, 1.00001)
        return value

    def intention_attention_local(self, intention, k, v, alpha_ent=1):
        alpha = torch.matmul(torch.relu(k.matmul(self.atten_w1) + intention.matmul(self.atten_w2) + self.atten_bias),
                             self.atten_w0.t())
        alpha = entmax_bisect(alpha, alpha_ent, dim=1)
        c = torch.matmul(alpha.transpose(1, 2), v)
        return c

    # item_embedding是item的嵌入，items是item序列，A是会话局部相似性图的重叠矩阵，D是会话局部相似性图的度矩阵，intention_embedding是意图表示intention的嵌入
    def forward(self, item_embedding, items, A, D, intention_embedding):
        seq_h = []
        for i in torch.arange(items.shape[0]):  # items的shape是[batch_size, seq_length]
            seq_h.append(torch.index_select(item_embedding, 0, items[i]))  # [b,s,d]
        seq_h1 = trans_to_cuda(torch.tensor([item.cpu().detach().numpy() for item in seq_h]))  # 数组越界报错
        len = seq_h1.shape[1]
        similarity_emb_gcn = torch.sum(seq_h1, 1)  # [b,d]
        DA = torch.mm(D, A).float()  # [b,b]
        similarity_emb_gcn = torch.mm(DA, similarity_emb_gcn)  # [b,d]
        similarity_emb_gcn = similarity_emb_gcn.unsqueeze(1).expand(similarity_emb_gcn.shape[0], len,
                                                                similarity_emb_gcn.shape[1])  # [b,s,d]

        intention_emb = intention_embedding
        alpha_line = self.get_alpha(x=intention_emb)
        q = intention_emb  # [b,1,d]
        k = similarity_emb_gcn  # [b,1,d]
        v = similarity_emb_gcn  # [b,1,d]

        line_c = self.intention_attention_local(q, k, v, alpha_ent=alpha_line)  # [b,1,d]
        c = torch.selu(line_c).squeeze()
        l_c = (c / torch.norm(c, dim=-1).unsqueeze(1))

        return l_c  # [b,d]


class SessionGraph(Module):
    def __init__(self, opt, n_node, pretrained_embeddings=None):
        super(SessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.len_max = opt.len_max
        self.batch_size = opt.batchSize
        self.nonhybrid = opt.nonhybrid
        self.last_num = opt.last_num

        # 如果有预训练的嵌入，则加载；否则使用nn.Embedding随机初始化
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
        else:
            self.embedding = nn.Embedding(self.n_node, self.hidden_size)

        self.gnn = GNN(self.hidden_size, step=opt.step)
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.pos_emb = Parameter(torch.Tensor(self.len_max + 1, self.hidden_size))  # 位置编码，用于区分不同位置的item
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)

        # 根据opt.loss_function的值来选择预测损失函数
        self.pred_loss_function = opt.loss_function == 'cross_entropy' and nn.CrossEntropyLoss() or FocalLoss(gamma=2)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.dropout = nn.Dropout(0.1)

        # 对比学习
        self.negative_samples = None  # 初始化 negative_samples 属性
        self.temperature = opt.temperature
        self.temperature_decay = opt.temperature_decay
        self.min_temperature = opt.min_temperature
        # self.residual_weight = opt.residual_weight

        # 新增：意图向量的数量
        self.intention_num = opt.intention_num

        # 新增：多头注意力层
        if self.intention_num > 1:
            self.multi_head_attention = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=self.intention_num)
            # 初始化查询向量，每个意图对应一个查询
            self.intent_queries = nn.Parameter(torch.randn(self.intention_num, self.hidden_size))
            # 注意力加权池化层
            self.attention_pooling = nn.Linear(self.hidden_size, 1)
        else:
            self.multi_head_attention = None  # 保持兼容性

        self.s = 16
        self.m = 0.1
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m
        # self.register_buffer('t', torch.FloatTensor([0.8]))
        self.t = 0.8
        self.spc = spc

        # Session Similarity Aggregation Module
        self.SessionSimilarityAggregation = SessionSimilarityAggregation(self.hidden_size)
        self.w_ne = opt.w_ne
        self.gama = opt.gama

        # Global Similarity Learning Module
        self.GlobalSimilarityLearning = GlobalSimilarityLearning(self.batch_size, self.hidden_size)
        # Local Similarity Learning Module
        self.LocalSimilarityLearning = LocalSimilarityLearning(self.batch_size, self.hidden_size)

        self.sub_len = opt.sub_len
        self.alpha = opt.alpha

        # # 新增：注意力融合模块
        # self.attention_fusion = AttentionFusion(self.hidden_size)

        # # **新增：多头注意力融合模块**
        # self.MultiHeadAttentionFusion = MultiHeadAttentionFusion(self.hidden_size, num_heads=4)
    
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


    def update_temperature(self):
        """
        根据衰减率更新温度参数，但不低于最小温度。
        """
        new_temperature = self.temperature * self.temperature_decay
        if new_temperature < self.min_temperature:
            new_temperature = self.min_temperature
        self.temperature = new_temperature
        print(f'Updated temperature: {self.temperature:.4f}')  # 可选：打印更新后的温度
        # 如果使用日志记录，可以取消注释以下行
        logger.info(f'Updated temperature: {self.temperature:.4f}')


    # inputs是session序列，其shape是(batch_size, max_n_node)
    # A是session图的邻接矩阵，其shape是(batch_size, max_n_node, 2 * max_n_node)
    # alias_inputs是session序列中每个item的索引，其shape是(batch_size, len_max)
    # A_hat是session关系图的重叠矩阵，D_hat是session关系图的度矩阵，二者的shape都是[batch_size, batch_size]
    def forward(self, inputs, A, alias_inputs, A_hat, D_hat, A_hat_local, D_hat_local):
        # 对输入的session序列进行嵌入，得到序列的隐藏表示，其shape为[batch_size, max_n_node, hidden_size]
        hidden = self.embedding(inputs)  # 对输入的item序列进行嵌入，得到item序列的隐藏表示
        # 对隐藏表示进行归一化，采纳NISER论文中的建议。这里不会影响self.embedding.weight。hidden的shape为[batch_size, max_n_node, hidden_size]
        hidden = self.dropout(F.normalize(hidden, dim=-1))  # 对隐藏表示进行归一化，采纳NISER论文中的建议。这里不会影响self.embedding.weight
        # 通过GNN模型，输入item序列的隐藏表示和邻接矩阵，输出序列的隐藏表示，shape为[batch_size, max_n_node, hidden_size]
        hidden = self.dropout(self.gnn(A, hidden))

        # 从隐藏表示中取出每个item的隐藏表示
        get = lambda i: hidden[i][alias_inputs[i]]
        # 将每个item的隐藏表示拼接起来，得到整个序列的隐藏表示，其shape为[batch_size, len_max, hidden_size]
        sequence_emb = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])

        # 添加位置编码
        sequence_emb = sequence_emb + self.pos_emb[:sequence_emb.size(1)]  # [batch_size, len_max, hidden_size]

        if self.intention_num > 1 and self.multi_head_attention is not None:
            # 准备多头注意力的输入
            # PyTorch的MultiheadAttention期望输入为 [seq_len, batch_size, embed_dim]
            sequence_emb_permuted = sequence_emb.permute(1, 0, 2)  # [len_max, batch_size, hidden_size]

            # 准备查询向量，每个意图对应一个查询
            queries = self.intent_queries.unsqueeze(1).repeat(1, sequence_emb.size(0), 1)  # [intention_num, batch_size, hidden_size]

            # 应用多头注意力
            attn_output, attn_weights = self.multi_head_attention(queries, sequence_emb_permuted, sequence_emb_permuted)
            # attn_output: [intention_num, batch_size, hidden_size]
            attn_output = attn_output.permute(1, 0, 2)  # [batch_size, intention_num, hidden_size]

            # 最大池化聚合意图向量
            h_target, _ = torch.max(attn_output, dim=1)  # [batch_size, hidden_size]

            # # 注意力加权池化
            # # 计算每个意图向量的权重
            # attention_weights = torch.sigmoid(self.attention_pooling(attn_output))  # [batch_size, intention_num, 1]
            # attention_weights = F.softmax(attention_weights, dim=1)  # 确保权重和为1

            # # 加权求和
            # h_target = torch.sum(attention_weights * attn_output, dim=1)  # [batch_size, hidden_size]
        else:
            # 如果 intention_num = 1，则使用原有的平均池化
            h_target = torch.mean(sequence_emb, 1)  # [batch_size, hidden_size]

        # MSGAT论文中是通过引入一个附加的空白节点来表示目标item，但是我们计划通过对当前batch中的每个session序列进行平均池化来表示目标item，其shape为[batch_size, 1, hidden_size]
        # target_emb = torch.mean(hidden, 1, True)
        # target_emb = torch.mean(sequence_emb, 1, True)  # shape 为[batch_size, 1, hidden_size]
        # 调用GlobalSimilarityLearning的forward()函数，输入分别是：item的嵌入、item序列、session关系图的重叠矩阵和度矩阵、目标item的嵌入
        # 返回的global_relation_emb是Session之间的关系表示，其shape为[batch_size, hidden_size]
        global_relation_emb = self.GlobalSimilarityLearning(self.embedding.weight, inputs, A_hat, D_hat, h_target.unsqueeze(1))
        # 调用LocalSimilarityLearning的forward()函数，输入分别是：item的嵌入、item序列、最后一个item相同的所有session的集合、目标item的嵌入
        # 返回的local_relation_emb是Session之间的局部关系表示，其shape为[batch_size, hidden_size]
        local_relation_emb = self.LocalSimilarityLearning(self.embedding.weight, inputs, A_hat_local, D_hat_local, h_target.unsqueeze(1))
        return hidden, global_relation_emb, local_relation_emb  # hidden的shape为[batch_size, max_n_node, hidden_size]，relation_emb的shape为[batch_size, hidden_size]

    # seq_hidden是session序列的隐藏表示，mask是序列长度，targets是目标item
    def compute_scores(self, sequence_emb, mask, targets, global_relation_emb, local_relation_emb, is_predict=False, h_sequence=None):
        # 如果 h_sequence 已经计算，则不再重复计算
        if h_sequence is None:
            sequence_emb = sequence_emb + self.pos_emb[:mask.shape[1]]  # 为session序列嵌入添加位置编码
            # ht是序列的最后一个item的隐藏表示，q1是ht的线性变换，q2是序列的隐藏表示的线性变换，alpha是注意力权重，a是注意力权重加权后的隐藏表示（即序列的隐藏表示）
            ht = sequence_emb[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # shape 为[batch_size, hidden_size]
            q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # shape 为[batch_size, 1, hidden_size]
            q2 = self.linear_two(sequence_emb)  # shape 为[batch_size, len_max, hidden_size]
            alpha = F.softmax(self.linear_three(torch.sigmoid(q1 + q2)) + (1 - mask).unsqueeze(-1) * (-9999),
                              dim=1)  # shape 为[batch_size, len_max, 1]
            h_sequence = torch.sum(alpha * sequence_emb * mask.view(mask.shape[0], -1, 1).float(),
                                   1)  # shape 为[batch_size, hidden_size]
            if not self.nonhybrid:  # 如果是混合模型，代表要结合session的长短期兴趣，其中ht是短期局部兴趣（序列中最后一个item），a是长期全局兴趣（通过注意力机制得到的整个序列的隐藏表示）
                h_sequence = F.normalize(self.linear_transform(torch.cat([h_sequence, ht], 1)), dim=-1)

        # 计算Session关系图中的Session之间的相似度（传入的是Session本身的隐藏表示与邻居Session的隐藏表示之和）
        h_fused = h_sequence + global_relation_emb + local_relation_emb
        # **使用注意力机制融合 h_global、h_local 和 h_sequence**
        # h_fused = self.attention_fusion(global_relation_emb, local_relation_emb, h_sequence)  # [batch_size, hidden_size]
        # **使用多头注意力机制融合 h_global、h_local 和 h_sequence**
        # h_fused = self.MultiHeadAttentionFusion(global_relation_emb, local_relation_emb, h_sequence)  # [batch_size, hidden_size]
        # 将融合后的表示传递给 SessionSimilarityAggregation 模块
        h_similarity = self.SessionSimilarityAggregation(h_fused)  # shape 为[batch_size, hidden_size]
        h_similarity = F.normalize(h_similarity, dim=-1)

        # # 保存原始嵌入以引入残差连接
        # original_h_sequence = h_sequence.clone()
        # original_h_similarity = h_similarity.clone()

        # 对比学习（仅在训练时进行）
        if not is_predict:
            # 计算对比损失
            contrastive_loss = self.compute_contrastive_loss(h_sequence, h_similarity, self.negative_samples)
        else:
            contrastive_loss = None

        # # 引入残差连接，将原始嵌入叠加回来
        # residual_weight = self.residual_weight  # 残差权重，可根据需要调整
        # h_sequence = h_sequence + original_h_sequence * residual_weight
        # h_sequence = F.normalize(h_sequence, dim=-1)

        # # 如果需要，也可以对 h_similarity 引入残差连接
        # h_similarity = h_similarity + original_h_similarity * residual_weight
        # h_similarity = F.normalize(h_similarity, dim=-1)

        # 得到session序列表示与session相似性表示的和，即最终的session序列表示
        h_sequence = h_sequence + h_similarity
        h_sequence = F.normalize(h_sequence, dim=-1)  # shape 为[batch_size, hidden_size]

        # items_emb是所有item归一化后的的隐藏表示，shape为[n_items, hidden_size]
        items_emb = F.normalize(self.embedding.weight[1:], dim=-1)  # n_nodes x latent_size

        # 计算当前session序列与各个item之间的预测得分
        if not is_predict:
            # 训练时采用余弦相似度来代替内积，对应NISER论文中的建议，计算session序列的隐藏表示和各个item的隐藏表示之间的余弦相似度
            cos_theta = torch.matmul(h_sequence, items_emb.transpose(1, 0))

            target_logit = cos_theta[torch.arange(0, cos_theta.size(0)), targets - 1].view(-1, 1)

            sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2)).clamp(0, 1)
            cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)
            mask = cos_theta > cos_theta_m
            final_target_logit = torch.where(target_logit > self.th, cos_theta_m, target_logit - self.mm)

            hard_example = cos_theta[mask] * self.t
            cos_theta[mask] = hard_example
            cos_theta.scatter_(1, (targets - 1).view(-1, 1).long(), final_target_logit)
            scores = cos_theta * self.s
        else:
            # 预测时采用内积，计算session序列的隐藏表示和各个item的隐藏表示之间的内积
            scores = torch.matmul(h_sequence, items_emb.transpose(1, 0))  # shape为[batch_size, n_nodes]

        # scores是当前batch中每个session序列与各个item之间的预测得分，shape为[batch_size, n_nodes]
        return scores, contrastive_loss

    def compute_contrastive_loss(self, h_sequence, h_similarity, negative_samples):
        """
        计算InfoNCE对比学习损失
    
        :param h_sequence: 当前 session 的序列嵌入，shape [batch_size, hidden_size]
        :param h_similarity: 与当前 session 相关的相似嵌入，shape [batch_size, hidden_size]
        :param negative_samples: 负样本的嵌入，shape [batch_size, num_negatives, hidden_size]
        :return: 对比损失标量
        """
        batch_size = h_sequence.size(0)
        num_negatives = negative_samples.size(1)

        # 正样本相似度
        positive_sim = F.cosine_similarity(h_sequence, h_similarity)  # [batch_size]

        # 负样本相似度
        negative_sim = F.cosine_similarity(
            h_sequence.unsqueeze(1), negative_samples, dim=-1
        )  # [batch_size, num_negatives]

        # Logits: 正样本在第一列，负样本在后面
        logits = torch.cat([positive_sim.unsqueeze(1), negative_sim], dim=1)  # [batch_size, 1 + num_negatives]
        logits = logits / self.temperature

        # Labels: 正样本为0
        labels = torch.zeros(batch_size).long().to(h_sequence.device)  # [batch_size]

        # InfoNCE 损失
        contrastive_loss = F.cross_entropy(logits, labels)

        return contrastive_loss

    def save(self):
        torch.save(self.state_dict(), directory + 'CFL.pth')
        print("====================================")
        print("Model has been saved...")
        print("====================================")
        logger.info("Model has been saved...")

    def load(self):
        self.load_state_dict(torch.load(directory + 'CFL.pth'))
        # self.critic.load_state_dict(torch.load(directory + 'critic.pth'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")
        logger.info("model has been loaded...")


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda(1)
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


# model表示SessionGraph模型，i表示当前batch的索引列表，其shape是[batch_size,]，data是Data对象，is_predict表示是否是预测
def forward(model, i, data, sub_len, alpha, is_predict=False):
    # alias_inputs:序列中每个item的索引，其shape是(batch_size, len_max)
    # A:序列中每个item的邻接矩阵，其shape是(batch_size, max_n_node, 2 * max_n_node)。因为每个序列涉及的不同item个数不会超过max_n_node，所以邻接矩阵的大小是(max_n_node, 2 * max_n_node)
    # items:session序列，其shape是(batch_size, max_n_node)
    # mask:序列mask，其shape是(batch_size, len_max)
    # targets:目标item，其shape是(batch_size,)
    alias_inputs, A, items, mask, targets = data.get_slice(i)
    # A_hat是session关系图的重叠矩阵，其shape是[batch_size, batch_size]。
    # D_hat是session关系图的度矩阵，其shape是[batch_size, batch_size]。
    A_hat, D_hat = data.get_global_overlap(items)
    A_hat_local, D_hat_local = data.get_local_overlap(items, model.last_num)
    # A_hat, D_hat = data.get_overlap_1(items, A)
    # A_hat, D_hat = data.get_overlap_mix(items, sub_len, alpha)
    # A_hat, D_hat = data.get_overlap_sub(items, sub_len)
    # A_hat, D_hat = data.get_overlap_popularity(items)
    # last_item_sessions = data.build_last_item_session_mapping(alias_inputs)  # 最后一个item相同的所有session的集合

    A_hat = trans_to_cuda(torch.Tensor(A_hat))
    D_hat = trans_to_cuda(torch.Tensor(D_hat))
    A_hat_local = trans_to_cuda(torch.Tensor(A_hat_local))
    D_hat_local = trans_to_cuda(torch.Tensor(D_hat_local))
    # last_item_sessions = trans_to_cuda(torch.LongTensor(last_item_sessions))
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    A = trans_to_cuda(torch.Tensor(A).float())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    targets = trans_to_cuda(torch.LongTensor(targets))  # shape 为[batch_size, 1]

    # 调用model的forward()函数，输入的是item序列和邻接矩阵，经过SessionGraph模型，输出的是序列的隐藏表示、Session之间的关系表示
    # hidden的shape为[batch_size, max_n_node, hidden_size]，relation_emb的shape为[batch_size, hidden_size]
    hidden, global_relation_emb, local_relation_emb = model(items, A, alias_inputs, A_hat, D_hat, A_hat_local, D_hat_local)
    # 从隐藏表示中取出每个item的隐藏表示
    get = lambda i: hidden[i][alias_inputs[i]]
    # 将每个item的隐藏表示拼接起来，得到整个序列的隐藏表示，其shape为[batch_size, len_max, hidden_size]
    sequence_emb = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])

    # 添加位置编码并计算 h_sequence
    sequence_emb = sequence_emb + model.pos_emb[:mask.shape[1]]
    ht = sequence_emb[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]
    q1 = model.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])
    q2 = model.linear_two(sequence_emb)
    alpha = F.softmax(model.linear_three(torch.sigmoid(q1 + q2)) + (1 - mask).unsqueeze(-1) * (-1e9), dim=1)
    h_sequence = torch.sum(alpha * sequence_emb * mask.view(mask.shape[0], -1, 1).float(), 1)
    if not model.nonhybrid:
        h_sequence = F.normalize(model.linear_transform(torch.cat([h_sequence, ht], 1)), dim=-1)

    # 将 h_sequence 转为 numpy 数组，并在 CPU 上处理
    h_sequence_np = h_sequence.detach().cpu().numpy()

    # target_emb = torch.mean(sequence_emb, 1, True)  # shape 为[batch_size, 1, hidden_size]

    # 负样本生成
    if not is_predict:
        negative_samples = data.get_hard_negative_samples(i, model.num_negatives, h_sequence_np)
        negative_samples = torch.tensor(negative_samples).to(h_sequence.device)
        model.negative_samples = negative_samples

    # 计算scores和对比损失，即当前batch中每个session序列与各个item之间的预测得分
    scores, contrastive_loss = model.compute_scores(
        sequence_emb, mask, targets, global_relation_emb, local_relation_emb, is_predict, h_sequence)
    return targets, scores, contrastive_loss  # targets的shape为[batch_size, 1]，scores的shape为[batch_size, n_nodes]


def train_test(model, train_data, test_data):
    model.scheduler.step()
    print('start training: ', datetime.datetime.now())
    logger.info('start training: %s' % datetime.datetime.now())
    model.train()
    total_loss = 0.0
    total_contrastive_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    for i, j in zip(slices, np.arange(len(slices))):
        model.optimizer.zero_grad()
        targets, scores, contrastive_loss = forward(model, i, train_data, model.sub_len, model.alpha)
        pred_loss = model.pred_loss_function(scores, targets - 1)
        if contrastive_loss is not None:
            loss = pred_loss + model.contrastive_weight * contrastive_loss
            total_contrastive_loss += contrastive_loss.item()
        else:
            loss = pred_loss
        loss.backward()
        model.optimizer.step()
        # total_loss += loss
        total_loss += loss.item()
        if j % int(len(slices) / 5 + 1) == 0:
            print('[%d/%d] Loss: %.4f, Contrastive Loss: %.4f' % (j, len(slices), loss.item(), contrastive_loss.item() if contrastive_loss is not None else 0))
            logger.info('[%d/%d] Loss: %.4f, Contrastive Loss: %.4f' % (j, len(slices), loss.item(), contrastive_loss.item() if contrastive_loss is not None else 0))
    print('\tLoss:\t%.3f, Contrastive Loss:\t%.3f' % (total_loss, total_contrastive_loss))
    logger.info('Loss: %.3f, Contrastive Loss: %.3f' % (total_loss, total_contrastive_loss))

    print('start predicting: ', datetime.datetime.now())
    logger.info('start predicting: %s' % datetime.datetime.now())
    model.eval()

    # Initialize metrics
    metrics = {k: {'hit': [], 'mrr': [], 'precision': [], 'recall': [], 'ndcg': []} for k in [5, 10, 20]}
    slices = test_data.generate_batch(model.batch_size)

    for i in slices:
        targets, scores, _ = forward(model, i, test_data, model.sub_len, model.alpha, is_predict=True)
        scores = trans_to_cpu(scores).detach().numpy()
        targets = trans_to_cpu(targets).detach().numpy()

        for k in [5, 10, 20]:
            sub_scores = np.argsort(-scores, axis=1)[:, :k]  # 取前K个得分最高的item
            for score, target in zip(sub_scores, targets):
                metrics[k]['hit'].append(np.isin(target - 1, score))
                metrics[k]['precision'].append(np.isin(target - 1, score).sum() / k)
                metrics[k]['recall'].append(np.isin(target - 1, score).sum() / 1)  # 单一目标时 recall 与 hit 相同
                if len(np.where(score == target - 1)[0]) == 0:
                    metrics[k]['mrr'].append(0)
                else:
                    metrics[k]['mrr'].append(1 / (np.where(score == target - 1)[0][0] + 1))
                # 计算 NDCG
                if target - 1 in score:
                    rank = np.where(score == target - 1)[0][0] + 1
                    metrics[k]['ndcg'].append(1 / np.log2(rank + 1))
                else:
                    metrics[k]['ndcg'].append(0)

    # Calculate average for each metric
    results = {}
    for k in [5, 10, 20]:
        results[f'HitRate@{k}'] = np.mean(metrics[k]['hit']) * 100
        results[f'MRR@{k}'] = np.mean(metrics[k]['mrr']) * 100
        results[f'Precision@{k}'] = np.mean(metrics[k]['precision']) * 100
        results[f'Recall@{k}'] = np.mean(metrics[k]['recall']) * 100
        results[f'NDCG@{k}'] = np.mean(metrics[k]['ndcg']) * 100

    return results
