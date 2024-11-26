#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import networkx as nx
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import torch
import torch.nn as nn
from tqdm import tqdm


# 将item的元数据转换为自然语言描述
def json2sentence(meta):
    # 提取元数据字段
    title = meta.get("title", "No Title")
    brand = meta.get("brand", "No Brand")
    category = ", ".join(meta.get("category", [])) or "Uncategorized"
    description_list = meta.get("description", [])
    description_text = " ".join(description_list) if description_list else "No description available."
    rank = meta.get("rank", "No rank information")
    price = meta.get("price", "No price available")
    features = ", ".join(meta.get("feature", [])) if meta.get("feature", []) else "No additional features"
    details = meta.get("details", {})

    # 确保details字段存在
    product_dimensions = details.get("Product Dimensions", "No product dimensions")
    shipping_weight = details.get("Shipping Weight", "No shipping weight")
    domestic_shipping = details.get("Domestic Shipping", "No domestic shipping information")
    international_shipping = details.get("International Shipping", "No international shipping information")

    also_buy = ", ".join(meta.get("also_buy", [])) if meta.get("also_buy", []) else "No recommendations"

    # 构建自然语言描述
    description = f"Product: {title} by {brand}. This product belongs to the {category} category."
    description += f" It is priced at {price}. Product description: {description_text}"
    description += f" The product is ranked {rank} in Beauty & Personal Care. "
    description += f" Key features: {features}. "
    description += f" Product dimensions are {product_dimensions}, and it weighs {shipping_weight}. "
    description += f" Domestic shipping information: {domestic_shipping}. "
    description += f" International shipping information: {international_shipping}. "
    description += f" Customers who bought this product also bought: {also_buy}."

    return description


# 使用智谱AI的 Embedding-3 模型生成商品元数据的原始维度嵌入
def load_raw_item_embeddings(meta_file, item_id_mapping_file, client):
    # 使用pickle读取item元数据
    try:
        with open(meta_file, 'rb') as f:  # 使用pickle的二进制读取方式
            item_meta = pickle.load(f)  # 反序列化元数据
            print(f"Loaded {len(item_meta)} item metadata records.")
    except Exception as e:
        print(f"Error reading meta file: {e}")
        return None

    # 使用pickle读取item_id到asin的映射
    try:
        with open(item_id_mapping_file, 'rb') as f:  # 以二进制方式打开文件
            item_id_to_asin = pickle.load(f)  # 反序列化item_id到asin映射
            print(f"Loaded {len(item_id_to_asin)} item_id to ASIN mappings.")
    except Exception as e:
        print(f"Error reading item_id mapping file: {e}")
        return None

    # 确定 embeddings 矩阵大小：根据最大 item_id 确定大小
    max_item_id = max(item_id_to_asin.keys())  # 获取item_id的最大值
    embeddings = torch.zeros((max_item_id + 1, 1))  # 初始化为1列，之后根据生成的嵌入维度调整

    # 准备描述文本和记录无元数据商品
    descriptions = []
    no_meta_items = []
    for item_id, asin in item_id_to_asin.items():
        meta = item_meta.get(asin, None)
        if meta:
            description = json2sentence(meta)
            descriptions.append((item_id, description))  # 保存 item_id 和描述
        else:
            # 如果没有元数据，将该商品ID记录下来
            no_meta_items.append(item_id)

    # 使用智谱AI Embedding-3 逐个生成商品嵌入
    print("Generating embeddings using ZhipuAI Embedding-3 model...")

    original_dim = None
    valid_embeddings = []  # 用于收集有效嵌入的列表

    for item_id, description in tqdm(descriptions, desc="Embedding items"):
        try:
            response = client.embeddings.create(
                model="embedding-3",
                input=[description]  # 逐个输入描述
            )
            embedding = torch.tensor(response.data[0].embedding)

            # 初始化 embeddings 矩阵
            if original_dim is None:
                original_dim = embedding.size(0)
                embeddings = torch.zeros((max_item_id + 1, original_dim))  # 根据最大item_id和嵌入维度调整

            # 保存生成的嵌入
            embeddings[item_id] = embedding
            valid_embeddings.append(embedding)  # 将有效的嵌入保存下来
        except Exception as e:
            print(f"Error generating embedding for item_id {item_id}: {e}")
            no_meta_items.append(item_id)

    # 计算所有有效嵌入的平均值
    if valid_embeddings:
        average_embedding = torch.stack(valid_embeddings).mean(dim=0)
        print(f"Computed average embedding from {len(valid_embeddings)} items.")
    else:
        print("No valid embeddings generated.")
        return None

    # 将没有元数据的商品嵌入设置为平均值并添加随机噪声
    print(f"{len(no_meta_items)} items missing metadata, setting their embeddings to the average embedding with noise...")
    noise_scale = 0.01  # 设置噪声幅度的比例
    for item_id in no_meta_items:
        noise = torch.randn(original_dim) * noise_scale  # 生成随机噪声
        embeddings[item_id] = average_embedding + noise  # 将噪声添加到平均嵌入

    print("Embedding generation complete.")
    return embeddings


# 将原始维度嵌入映射到指定的hiddenSize
def space_projector(raw_embeddings, hidden_size):
    original_dim = raw_embeddings.size(1)
    # 如果原始维度与hiddenSize相同，直接返回
    if original_dim == hidden_size:
        return raw_embeddings
    # 使用线性层将嵌入映射到hiddenSize
    linear_layer = nn.Linear(original_dim, hidden_size)
    mapped_embeddings = linear_layer(raw_embeddings)
    return mapped_embeddings


def build_graph(train_data):
    graph = nx.DiGraph()
    for seq in train_data:
        for i in range(len(seq) - 1):
            if graph.get_edge_data(seq[i], seq[i + 1]) is None:
                weight = 1
            else:
                weight = graph.get_edge_data(seq[i], seq[i + 1])['weight'] + 1
            graph.add_edge(seq[i], seq[i + 1], weight=weight)
    for node in graph.nodes:
        sum = 0
        for j, i in graph.in_edges(node):
            sum += graph.get_edge_data(j, i)['weight']
        if sum != 0:
            for j, i in graph.in_edges(i):
                graph.add_edge(j, i, weight=graph.get_edge_data(j, i)['weight'] / sum)
    return graph


def data_masks(all_usr_pois, item_tail):
    us_lens = [len(upois) for upois in all_usr_pois]  # 每个用户的序列长度
    len_max = max(us_lens)
    us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)]  # 将每个用户的序列填充到相同的长度
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]  # 生成mask，前le个为1，后面为0。目的是为了在计算loss时，只计算前le个的loss。
    # all_usr_pois表示所有的session序列，us_lens表示每个session序列的长度，len_max表示所有session序列中的最大长度
    # item_tail表示用于填充的item，us_pois表示填充后的所有session序列
    # us_msks表示填充后的所有session序列的掩码，前le个为1，后面为0
    return us_pois, us_msks, len_max


def split_validation(train_set, valid_portion):  # 将数据集分为训练集和验证集
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


class Data():
    def __init__(self, data, shuffle=False, graph=None):
        inputs = data[
            0]  # data[0]表示数据集data中的所有session序列，其shape是(数据集大小, 序列实际长度)。data[1]表示数据集data中的所有目标item，其shape是(数据集大小,)。
        inputs, mask, len_max = data_masks(inputs, [0])  # inputs表示填充后的所有session序列，其大小为（数据集大小，最大序列长度）。
        self.inputs = np.asarray(inputs)  # 将inputs转换为numpy数组，其shape是(数据集大小, 最大序列长度)。
        self.mask = np.asarray(mask)
        self.len_max = len_max
        self.targets = np.asarray(data[1])  # 将data[1]转换为numpy数组，其shape是(数据集大小,)。
        self.length = len(inputs)
        self.shuffle = shuffle
        self.graph = graph

        # 新增：计算所有item的流行度
        self.item_popularity = self._compute_item_popularity(data[0])
        # 新增：计算每个session的TF-IDF矩阵
        self.sessions_text = [' '.join(map(str, session)) for session in data[0]]
        self.tfidf_vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=True)
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.sessions_text)

        # 新增：保存每个 session 中的 items 集合，用于负样本采样
        self.session_items = [set(session[session != 0]) for session in self.inputs]  # 确保去除填充的 0

    # 负样本采样方法1（先按照相似度升序排序所有其他会话，再检查是否没有重叠项，若负样本数量不足，则随机采样）
    def get_negative_samples(self, indices, num_negatives, h_sequence):
        """
        根据与当前会话的余弦相似度，选择负样本

        :param indices: 当前批次的索引列表
        :param num_negatives: 负样本数量
        :param h_sequence: 当前批次的序列嵌入，shape [batch_size, hidden_size]
        :return: 负样本的嵌入，shape [batch_size, num_negatives, hidden_size]
        """
        batch_size = len(indices)
        negative_samples = []

        # 当前批次内的所有会话嵌入
        all_h_sequence = h_sequence  # [batch_size, hidden_size]
        all_indices = indices  # 当前批次的索引列表

        # 对于每个会话，找到最不相似的会话作为负样本
        for idx, index in enumerate(indices):
            current_embedding = h_sequence[idx]  # [hidden_size]
            current_items = self.session_items[index]
            # 计算与其他会话的相似度
            similarities = np.dot(all_h_sequence, current_embedding) / (
                    np.linalg.norm(all_h_sequence, axis=1) * np.linalg.norm(current_embedding) + 1e-8)
            # 将自身的相似度置为 -inf，以便排除自身
            similarities[idx] = -np.inf
            # 找到相似度最低的会话索引
            sorted_indices = np.argsort(similarities)
            negatives = []
            for neg_idx in sorted_indices:
                if len(negatives) >= num_negatives:
                    break
                neg_items = self.session_items[all_indices[neg_idx]]
                # 检查是否没有重叠的 items
                if len(current_items.intersection(neg_items)) == 0:
                    negatives.append(all_h_sequence[neg_idx])
            # 如果负样本数量不足，随机从当前批次内采样
            if len(negatives) < num_negatives:
                remaining = num_negatives - len(negatives)
                # 排除自身
                possible_indices = [i for i in range(batch_size) if i != idx]
                if len(possible_indices) > 0:
                    random_indices = np.random.choice(possible_indices, size=remaining, replace=True)
                    for rand_idx in random_indices:
                        neg_items = self.session_items[all_indices[rand_idx]]
                        if len(current_items.intersection(neg_items)) == 0:
                            negatives.append(all_h_sequence[rand_idx])
                        else:
                            negatives.append(np.random.randn(h_sequence.shape[1]))  # 随机生成一个嵌入
                else:
                    # 如果可能的索引为空，直接使用随机嵌入
                    for _ in range(remaining):
                        negatives.append(np.random.randn(h_sequence.shape[1]))
            negatives = negatives[:num_negatives]
            negative_samples.append(negatives)
        negative_samples = np.array(negative_samples)  # [batch_size, num_negatives, hidden_size]
        return negative_samples

    # 负样本采样方法2（在当前批次内完全随机采样）
    def get_negative_samples_random(self, indices, num_negatives, h_sequence):
        """
        完全随机采样的负样本生成方法，排除当前会话自身，且确保负样本与当前会话没有任何重叠项。

        :param indices: 当前批次的索引列表
        :param num_negatives: 负样本数量
        :param h_sequence: 当前批次的序列嵌入，shape [batch_size, hidden_size]
        :return: 负样本的嵌入，shape [batch_size, num_negatives, hidden_size]
        """
        batch_size = len(indices)
        hidden_size = h_sequence.shape[1]
        negative_samples = []

        for idx in range(batch_size):
            current_index = indices[idx]
            current_items = self.session_items[current_index]

            # 定义可能的负样本索引：当前批次的所有索引，除去自身
            possible_indices = list(range(batch_size))
            possible_indices.remove(idx)

            # 从可能的索引中筛选出没有任何重叠项的会话
            non_overlapping_indices = [
                i for i in possible_indices
                if len(self.session_items[indices[i]].intersection(current_items)) == 0
            ]

            # 随机选择负样本
            negatives = []
            if len(non_overlapping_indices) >= num_negatives:
                sampled_indices = np.random.choice(non_overlapping_indices, size=num_negatives, replace=False)
                negatives = h_sequence[sampled_indices]  # [num_negatives, hidden_size]
            else:
                # 如果非重叠会话数量不足，先采样所有非重叠会话
                sampled_indices = non_overlapping_indices.copy()
                negatives = h_sequence[sampled_indices].tolist()

                # 计算还需要多少负样本
                remaining = num_negatives - len(sampled_indices)

                if len(possible_indices) > len(non_overlapping_indices):
                    # 允许有放回地采样剩余的负样本
                    additional_possible = [i for i in possible_indices if i not in sampled_indices]
                    if len(additional_possible) > 0:
                        sampled_extra = np.random.choice(additional_possible, size=remaining, replace=True)
                        negatives.extend(h_sequence[sampled_extra].tolist())
                else:
                    # 当前批次内只有非重叠会话，允许有放回地采样
                    if len(non_overlapping_indices) > 0:
                        sampled_extra = np.random.choice(non_overlapping_indices, size=remaining, replace=True)
                        negatives.extend(h_sequence[sampled_extra].tolist())

                # 转换为 numpy 数组并截取到 num_negatives
                negatives = negatives[:num_negatives]

                # 如果仍然不足，直接从可能的索引中采样（即使可能有重叠项）
                if len(negatives) < num_negatives:
                    remaining = num_negatives - len(negatives)
                    sampled_extra = np.random.choice(possible_indices, size=remaining, replace=True)
                    negatives.extend(h_sequence[sampled_extra].tolist())
                    negatives = negatives[:num_negatives]

            negatives = np.array(negatives)  # [num_negatives, hidden_size]
            negative_samples.append(negatives)

        negative_samples = np.array(negative_samples)  # [batch_size, num_negatives, hidden_size]
        return negative_samples

    # 负样本采样方法3（难负样本采样）
    def get_hard_negative_samples(self, indices, num_negatives, h_sequence):
        """
        难负样本采样方法，选择与当前session相似但不重叠的session作为负样本。
        
        :param indices: 当前批次的索引列表
        :param num_negatives: 负样本数量
        :param h_sequence: 当前批次的序列嵌入，shape [batch_size, hidden_size]，类型为 numpy.ndarray
        :return: 负样本的嵌入，shape [batch_size, num_negatives, hidden_size]
        """
        batch_size = len(indices)
        negative_samples = []

        # 由于 h_sequence 已经是 numpy 数组，无需调用 .cpu().numpy()
        all_h_sequence = h_sequence  # [batch_size, hidden_size]

        # 计算余弦相似度矩阵
        norm = np.linalg.norm(all_h_sequence, axis=1, keepdims=True)
        similarities = np.dot(all_h_sequence, all_h_sequence.T) / (norm * norm.T + 1e-8)

        for idx, index in enumerate(indices):
            current_embedding = all_h_sequence[idx]
            current_items = self.session_items[index]

            # 排除自身和有重叠的session
            valid_indices = [
                i for i in range(batch_size)
                if i != idx and len(self.session_items[indices[i]].intersection(current_items)) == 0
            ]

            # 选择相似度最高的session作为难负样本
            if len(valid_indices) >= num_negatives:
                # 按相似度降序排序有效索引
                sorted_valid_indices = sorted(valid_indices, key=lambda x: similarities[idx][x], reverse=True)
                selected_indices = sorted_valid_indices[:num_negatives]
            else:
                # 如果不足，随机采样
                selected_indices = valid_indices.copy()
                remaining = num_negatives - len(selected_indices)
                if remaining > 0:
                    additional_indices = np.random.choice([i for i in range(batch_size) if i != idx], size=remaining, replace=True)
                    selected_indices.extend(additional_indices.tolist())

            # 获取负样本嵌入
            neg_samples = all_h_sequence[selected_indices]
            negative_samples.append(neg_samples)

        negative_samples = np.array(negative_samples)  # [batch_size, num_negatives, hidden_size]
        return negative_samples


    def _compute_item_popularity(self, all_sessions):
        item_counts = Counter(item for session in all_sessions for item in session if item != 0)  # 统计所有item的出现次数
        total_items = sum(item_counts.values())  # 所有item的总数
        return {item: count / total_items for item, count in item_counts.items()}  # 计算所有item的流行度，即出现次数占总数的比例

    # 计算每个session序列与其他session序列的重叠比例，即两个序列的交集占并集的比例，返回的是重叠矩阵和度矩阵
    def global_similarity_calculation(self, sessions):
        matrix = np.zeros((len(sessions), len(sessions)))
        for i in range(len(sessions)):
            seq_a = set(sessions[i])
            seq_a.discard(0)  # 删除序列中的0
            for j in range(i + 1, len(sessions)):
                seq_b = set(sessions[j])
                seq_b.discard(0)  # 删除序列中的0
                overlap = seq_a.intersection(seq_b)
                ab_set = seq_a | seq_b
                matrix[i][j] = float(len(overlap)) / float(len(ab_set))
                matrix[j][i] = matrix[i][j]
        matrix = matrix + np.diag([1.0] * len(sessions))
        degree = np.sum(np.array(matrix), 1)
        degree = np.diag(1.0 / degree)
        return matrix, degree

    def get_tfidf_weighted_overlap(self, sessions):
        # 将当前批次的session转换为文本形式以计算TF-IDF权重
        current_sessions_text = [' '.join(map(str, session)) for session in sessions]
        current_tfidf_matrix = self.tfidf_vectorizer.transform(current_sessions_text)

        matrix = np.zeros((len(sessions), len(sessions)))
        for i in range(len(sessions)):
            seq_a_tfidf = current_tfidf_matrix[i].toarray()[0]  # 获取第i个session的TF-IDF权重
            seq_a = set(sessions[i])
            seq_a.discard(0)  # 删除序列中的0
            for j in range(i + 1, len(sessions)):
                seq_b_tfidf = current_tfidf_matrix[j].toarray()[0]
                seq_b = set(sessions[j])
                seq_b.discard(0)  # 删除序列中的0

                overlap = seq_a.intersection(seq_b)
                ab_set = seq_a | seq_b
                # 计算加权交集和加权并集
                # 注意：这里的计算方法需要根据TF-IDF的特性调整，直接求和可能不恰当，因为TF-IDF是针对整个session的，我们需要找到一种方式映射到交集和并集的计算上
                weighted_overlap = sum(
                    seq_a_tfidf[k] * seq_b_tfidf[k] for k in overlap if k < len(seq_a_tfidf))  # 仅在TF-IDF矩阵有效范围内计算
                union_weighted_items = sum(seq_a_tfidf[k] for k in ab_set if k < len(seq_a_tfidf)) + sum(
                    seq_b_tfidf[k] for k in ab_set if k < len(seq_b_tfidf))  # 仅在TF-IDF矩阵有效范围内计算
                matrix[i][j] = weighted_overlap / union_weighted_items if union_weighted_items != 0 else 0
                matrix[j][i] = matrix[i][j]
        matrix = matrix + np.diag([1.0] * len(sessions))
        degree = np.sum(np.array(matrix), 1)
        degree = np.diag(1.0 / degree)
        return matrix, degree

    # 考虑item的流行度，计算每个session序列与其他session序列的重叠比例，即两个序列的交集占并集的比例，返回的是重叠矩阵和度矩阵
    def get_overlap_popularity(self, sessions):
        # sessions是当前batch的所有session序列，其shape是(batch_size, max_n_node)
        matrix = np.zeros((len(sessions), len(sessions)))  # 初始化一个全零矩阵，大小为sessions的长度
        for i in range(len(sessions)):
            seq_a = set(sessions[i])  # 获取第i个序列
            seq_a.discard(0)  # 删除序列中的0
            for j in range(i + 1, len(sessions)):  # 遍历i之后的序列
                seq_b = set(sessions[j])  # 获取第j个序列
                seq_b.discard(0)  # 删除序列中的0
                # 计算加权交集、加权并集
                overlap_weight = sum(
                    [self.item_popularity[item] for item in seq_a.intersection(seq_b)])  # 交集的流行度之和
                ab_set_weight = sum([self.item_popularity[item] for item in seq_a | seq_b])  # 并集的流行度之和
                overlap_ratio = overlap_weight / ab_set_weight if ab_set_weight != 0 else 0  # 计算重叠的item部分占所有item部分的比例。# 防止分母为0的情况，这里假设如果并集的流行度和为0，则说明没有有效item参与计算，直接赋值为0
                matrix[i][j] = overlap_ratio  # 将重叠比例赋值给矩阵
                matrix[j][i] = matrix[i][j]  # 对称矩阵，所以对称位置的值相同
        matrix = matrix + np.diag([1.0] * len(sessions))  # 对角线上的值为1，即每个序列与自己的重叠比例为1
        degree = np.sum(np.array(matrix), 1)  # 计算每个序列的度，即每个序列与其他序列的重叠比例之和，大小为sessions的长度。
        degree = np.diag(1.0 / degree)  # 计算度的倒数
        return matrix, degree  # 返回重叠矩阵和度矩阵

    # (效果不好,且复杂度过高)
    # 目前计算session之间相似性的方法由get_overlap函数实现，其思想是统计两个session序列的重叠item占所有item的比例，然后将这个比例作为两个session序列的相似性。
    # 但是，这种方法有一个问题，即它只考虑了session序列之间的共同item，而没有考虑item之间的转换关系（相当于session子序列）。
    # 例如，假设session1为v1,v2,v5,v3,v6，session2为v4,v6,v3，session3为v5,v3,v1,v6,v3,v7。
    # 如果按照get_overlap函数的思想，那么三个session之间都有相关性，因为它们之间都有共同的item。其中session1和session2的相似度为0.33，session1和session3的相似度为0.67，session2和session3的相似度为0.33。
    # 因此，就有两种改进思路：1）替换原本统计重叠item的方法，改为统计session子序列之间的相似性；2）将原本统计重叠item的方法和新的统计session子序列之间的相似性方法结合起来。
    # 针对思路1，可以统计session之间的重叠子序列占所有子序列的比例，其中子序列的长度是一个超参数sub_len，即统计session之间的重叠长度为sub_len的子序列占所有长度为sub_len的子序列的比例。当sub_len=0时，即统计session之间的重叠item占所有item的比例，这时就是get_overlap函数的方法。
    def get_overlap_sub(self, sessions, sub_len):
        matrix = np.zeros((len(sessions), len(sessions)))
        for i in range(len(sessions)):
            seq_a = sessions[i]
            seq_a = [seq_a[j:j + sub_len] for j in range(len(seq_a) - sub_len + 1)]
            for j in range(i + 1, len(sessions)):
                seq_b = sessions[j]
                seq_b = [seq_b[j:j + sub_len] for j in range(len(seq_b) - sub_len + 1)]
                overlap = set(map(tuple, seq_a)).intersection(set(map(tuple, seq_b)))
                ab_set = set(map(tuple, seq_a)).union(set(map(tuple, seq_b)))
                matrix[i][j] = float(len(overlap)) / float(len(ab_set))
                matrix[j][i] = matrix[i][j]
        matrix = matrix + np.diag([1.0] * len(sessions))
        degree = np.sum(np.array(matrix), 1)
        degree = np.diag(1.0 / degree)
        return matrix, degree

    # 针对思路2，可以将两种方法的结果进行加权求和，其中权重是一个超参数alpha，即最终的相似性为alpha * get_overlap函数的结果 + (1 - alpha) * 新的相似性计算方法的结果。
    def get_overlap_mix(self, sessions, sub_len, alpha):
        matrix = np.zeros((len(sessions), len(sessions)))
        for i in range(len(sessions)):
            seq_a = set(sessions[i])
            seq_a.discard(0)
            seq_a_sub = [sessions[i][j:j + sub_len] for j in range(len(sessions[i]) - sub_len + 1)]
            for j in range(i + 1, len(sessions)):
                seq_b = set(sessions[j])
                seq_b.discard(0)
                seq_b_sub = [sessions[j][k:k + sub_len] for k in range(len(sessions[j]) - sub_len + 1)]
                overlap = seq_a.intersection(seq_b)
                ab_set = seq_a.union(seq_b)
                # TypeError: unhashable type: 'list'. 说明seq_a_sub和seq_b_sub是list类型，不能直接用set函数，需要先转换为tuple类型.
                overlap_sub = set(map(tuple, seq_a_sub)).intersection(set(map(tuple, seq_b_sub)))
                ab_set_sub = set(map(tuple, seq_a_sub)).union(set(map(tuple, seq_b_sub)))
                matrix[i][j] = alpha * float(len(overlap)) / float(len(ab_set)) + (1 - alpha) * float(
                    len(overlap_sub)) / float(len(ab_set_sub))
                matrix[j][i] = matrix[i][j]
        matrix = matrix + np.diag([1.0] * len(sessions))
        degree = np.sum(np.array(matrix), 1)
        degree = np.diag(1.0 / degree)
        return matrix, degree

    # 在思路2的基础上，其实公共子序列不一定需要连续，可以容忍一定的间隔（引入超参数gap，小于sub_len），即针对两个长度为sub_len的子序列，可以允许其中间有gap个item不同。
    def get_overlap_mix_gap(self, sessions, sub_len, alpha, gap):
        matrix = np.zeros((len(sessions), len(sessions)))
        for i in range(len(sessions)):
            seq_a = set(sessions[i])
            seq_a.discard(0)
            seq_a_sub = [sessions[i][j:j + sub_len] for j in range(len(sessions[i]) - sub_len + 1)]
            for j in range(i + 1, len(sessions)):
                seq_b = set(sessions[j])
                seq_b.discard(0)
                seq_b_sub = [sessions[j][k:k + sub_len] for k in range(len(sessions[j]) - sub_len + 1)]
                overlap = seq_a.intersection(seq_b)
                ab_set = seq_a.union(seq_b)
                overlap_sub = 0
                ab_set_sub = 0
                for seq_a_sub_item in seq_a_sub:
                    for seq_b_sub_item in seq_b_sub:
                        if len(set(seq_a_sub_item).intersection(set(seq_b_sub_item))) >= sub_len - gap:
                            overlap_sub += 1
                        ab_set_sub += 1
                matrix[i][j] = alpha * float(len(overlap)) / float(len(ab_set)) + (1 - alpha) * float(
                    overlap_sub) / float(ab_set_sub)
                matrix[j][i] = matrix[i][j]
        matrix = matrix + np.diag([1.0] * len(sessions))
        degree = np.sum(np.array(matrix), 1)
        degree = np.diag(1.0 / degree)
        return matrix, degree

    # 此外，针对get_overlap函数的思想，在考虑session之间的相似性时要消除流行度偏差。因为流行的item可能会在多个session中出现，频繁地作为不同session之间的重叠item，导致这些session之间的相似性被高估。因此，我们需要对流行度进行惩罚，以减少这种偏见的影响。
    # 具体地，可以使用余弦相似度来计算session之间的相似性，即将get_overlap函数的结果作为余弦相似度的分子，分母是两个session序列的流行度的乘积的平方根。
    def get_cosine(self, sessions):
        matrix = np.zeros((len(sessions), len(sessions)))
        for i in range(len(sessions)):
            seq_a = set(sessions[i])
            seq_a.discard(0)
            for j in range(i + 1, len(sessions)):
                seq_b = set(sessions[j])
                seq_b.discard(0)
                overlap = seq_a.intersection(seq_b)
                ab_set = seq_a.union(seq_b)
                matrix[i][j] = float(len(overlap)) / np.sqrt(len(seq_a) * len(seq_b))
                matrix[j][i] = matrix[i][j]
        matrix = matrix + np.diag([1.0] * len(sessions))
        degree = np.sum(np.array(matrix), 1)
        degree = np.diag(1.0 / degree)
        return matrix, degree

    # 新增：计算session之间的局部相似性（即只考虑最后一个item是否相同）
    def get_last_item_similarity(self, sessions):
        # 计算每个session最后一个item相同的其他session之间的相似度
        last_items = [session[-1] for session in sessions]
        local_similarity_matrix = np.zeros((len(sessions), len(sessions)))
        for i in range(len(sessions)):
            for j in range(i + 1, len(sessions)):
                if last_items[i] == last_items[j]:
                    local_similarity_matrix[i][j] = 1.0
                    local_similarity_matrix[j][i] = local_similarity_matrix[i][j]
        return local_similarity_matrix

    def build_last_item_session_mapping(self, alias_inputs):
        """
        构建一个字典，键为session中最后一个item的ID，值为具有相同结尾item的所有session的索引列表。

        :param alias_inputs: 二维列表，表示所有session，每个内部列表是一个session中的item序列，并且被填充0到相同长度。shape为(batch_size, len_max)
        :return: 字典，{item_id: [session_index_1, session_index_2, ...]}
        """
        last_item_session_mapping = {}
        for i, session in enumerate(alias_inputs):
            session = session[:list(session).index(0)]  # 截取session中的有效item，即去掉填充的0
            if (len(session) == 0):
                continue
            last_item = session[-1]
            if last_item not in last_item_session_mapping:
                last_item_session_mapping[last_item] = []
            last_item_session_mapping[last_item].append(i)
        return last_item_session_mapping

    def get_global_overlap(self, sessions):
        """
        计算全局会话相似性矩阵。

        参数:
        - sessions (List[List[int]]): 会话列表，每个会话是一个项的列表。

        返回:
        - matrix (np.ndarray): 全局相似性矩阵。
        - degree (np.ndarray): 度矩阵的倒数。
        """
        num_sessions = len(sessions)
        matrix = np.zeros((num_sessions, num_sessions))
        
        # 预处理：将每个会话转换为集合并移除特殊项
        item_sets = [set(session) - {0} for session in sessions]
        
        # 计算全局相似性矩阵（Jaccard相似系数）
        for i in range(num_sessions):
            for j in range(i + 1, num_sessions):
                set_a = item_sets[i]
                set_b = item_sets[j]
                if not set_a and not set_b:
                    similarity = 0.0
                else:
                    overlap = set_a.intersection(set_b)
                    ab_set = set_a.union(set_b)
                    similarity = float(len(overlap)) / float(len(ab_set)) if ab_set else 0.0
                matrix[i][j] = similarity
                matrix[j][i] = similarity  # 对称性
        
        # 添加对角线元素（自相似度）
        np.fill_diagonal(matrix, 1.0)
        
        # 计算度矩阵的倒数
        degree = np.sum(matrix, axis=1)
        # 防止除以零
        degree = np.where(degree != 0, 1.0 / degree, 0.0)
        degree = np.diag(degree)
        
        return matrix, degree

    def get_local_overlap(self, sessions, last_num=1):
        """
        计算局部会话相似性矩阵。

        参数:
        - sessions (List[List[int]]): 会话列表，每个会话是一个项的列表。
        - last_num (int): 考虑每个会话最后的项数，用于计算局部相似性。

        返回:
        - matrix (np.ndarray): 局部相似性矩阵。
        - degree (np.ndarray): 度矩阵的倒数。
        """
        num_sessions = len(sessions)
        matrix = np.zeros((num_sessions, num_sessions))
        
        # 预处理：提取每个会话的最后 `last_num` 个项，并转换为集合
        local_item_sets = []
        for session in sessions:
            # 如果会话长度小于 last_num，则取整个会话
            local_items = session[-last_num:] if len(session) >= last_num else session
            local_set = set(local_items)
            local_set.discard(0)  # 移除特殊项（如 0）如果有需要
            local_item_sets.append(local_set)
        
        # 计算局部相似性矩阵
        for i in range(num_sessions):
            for j in range(i + 1, num_sessions):
                set_a = local_item_sets[i]
                set_b = local_item_sets[j]
                if not set_a and not set_b:
                    similarity = 0.0
                else:
                    overlap = set_a.intersection(set_b)
                    ab_set = set_a.union(set_b)
                    similarity = float(len(overlap)) / float(len(ab_set)) if ab_set else 0.0
                matrix[i][j] = similarity
                matrix[j][i] = similarity  # 对称性
        
        # 添加对角线元素（自相似度）
        np.fill_diagonal(matrix, 1.0)
        
        # 计算度矩阵的倒数
        degree = np.sum(matrix, axis=1)
        # 防止除以零
        degree = np.where(degree != 0, 1.0 / degree, 0.0)
        degree = np.diag(degree)
        
        return matrix, degree
        
    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            self.mask = self.mask[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]
        return slices

    def get_slice(self, i):
        # 获取当前batch的数据（共batch_size条，具体每条数据的索引包含在i列表中，i的shape是(batch_size,)）
        # inputs是session序列，其shape是(batch_size, len_max)，mask是序列mask，其shape是(batch_size, len_max)，targets是目标item，其shape是(batch_size,)
        inputs, mask, targets = self.inputs[i], self.mask[i], self.targets[i]
        # n_node是序列中的item个数，其shape是(batch_size,)
        items, n_node, A, alias_inputs = [], [], [], []
        for u_input in inputs:
            n_node.append(len(np.unique(u_input)))  # n_node记录了各个序列中不同item的个数
        # max_n_node是所有session序列中不同item的最大个数，注意区分它和len_max的区别：len_max是所有session序列的最大序列长度，而max_n_node是所有session序列中不同item的最大个数！
        max_n_node = np.max(n_node)
        for u_input in inputs:
            node = np.unique(u_input)
            items.append(node.tolist() + (max_n_node - len(node)) * [0])
            u_A = np.zeros((max_n_node, max_n_node))
            for i in np.arange(len(u_input) - 1):
                if u_input[i + 1] == 0:
                    break
                u = np.where(node == u_input[i])[0][0]
                v = np.where(node == u_input[i + 1])[0][0]
                u_A[u][v] = 1
            u_sum_in = np.sum(u_A, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A, u_sum_in)
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()
            A.append(u_A)
            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
        # alias_inputs:序列中每个item的索引，其shape是(batch_size, len_max)
        # A:序列中每个item的邻接矩阵，其shape是(batch_size, max_n_node, 2 * max_n_node)。因为每个序列涉及的item个数不会超过max_n_node，所以邻接矩阵的大小是(max_n_node, 2 * max_n_node)
        # items:session序列，其shape是(batch_size, max_n_node)
        # mask:序列mask，其shape是(batch_size, len_max)
        # targets:目标item，其shape是(batch_size,)
        return alias_inputs, A, items, mask, targets
