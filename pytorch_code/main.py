#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import argparse
import pickle
import time
from utils import build_graph, Data, split_validation, load_raw_item_embeddings, space_projector
from model import *
import torch
from zhipuai import ZhipuAI
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='diginetica',
                    help='dataset name: diginetica/yoochoose1_4/yoochoose1_64/sample/luxury_beauty/musical_instruments/prime_pantry')
parser.add_argument('--batchSize', type=int, default=512, help='input batch size')  # 原本是100
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
parser.add_argument('--epoch', type=int, default=50, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=6,
                    help='the number of steps after which the learning rate decay')  # 原本是3
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--patience', type=int, default=3, help='the number of epoch to wait before early stop ')
parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1,
                    help='split the portion of training set as validation set')
parser.add_argument('--loss_function', default='focal', help='loss function, focal or cross_entropy')
parser.add_argument('--w_ne', type=float, default=1.7, help='neighbor weight')  # digi：1.7 Tmall 0.9
parser.add_argument('--gama', type=float, default=1.7, help='cos_sim')  # digi：1.7
parser.add_argument('--sub_len', type=int, default=2, help='session之间的重叠子序列的长度')
parser.add_argument('--alpha', type=float, default=0.5, help='考虑session之间的相似性时，考虑重叠item的权重，（1-alpha）为考虑重叠子序列的权重')
parser.add_argument('--gap', type=int, default=1, help='session之间的重叠子序列所能容忍的不同item的个数')

parser.add_argument('--last_num', type=int, default=3, help='session之间的局部相似性的考察item数目')

# 对比学习相关超参数
parser.add_argument('--temperature', type=float, default=0.5, help='Initial temperature for contrastive learning')
parser.add_argument('--temperature_decay', type=float, default=0.99, help='Decay rate for temperature parameter in contrastive learning')
parser.add_argument('--min_temperature', type=float, default=0.1, help='Minimum temperature value for contrastive learning')
parser.add_argument('--num_negatives', type=int, default=64, help='number of negative samples for contrastive learning, such as 5, 10, 20')  # 5
parser.add_argument('--contrastive_weight', type=float, default=0.3, help='weight for contrastive loss, such as 0.1, 0.2, 0.5')
# parser.add_argument('--residual_weight', type=float, default=0.5, help='')
parser.add_argument('--intention_num', type=int, default=4, help='number of intention vectors for multi-head attention')
opt = parser.parse_args()

# 配置日志
from logger_config import setup_logger

# 获取当前时间戳（整数）
current_time = str(int(time.time()))
log_filename = 'log/' + opt.dataset + '_current_time=' + current_time + '_batchSize=' + str(opt.batchSize) + '_hiddenSize=' + str(opt.hiddenSize) + \
    '_temperature=' + str(opt.temperature) + '_last_num=' + str(opt.last_num) + \
    '_num_negatives=' + str(opt.num_negatives) + '_contrastive_weight=' + str(opt.contrastive_weight) + \
        '_intention_num=' + str(opt.intention_num) +'.log'

logger = setup_logger(log_filename)

print(opt)
logger.info(opt)


# 加载智谱AI的 Embedding-3 模型用于嵌入生成，并保留原始维度
def llm_meta_driven_embedding(dataset):
    meta_file_path = f'../datasets/{dataset}/filtered_item_meta.txt'
    asin_mapping_file = f'../datasets/{dataset}/item_id_to_asin.txt'
    raw_embedding_file = f'../datasets/{dataset}/raw_item_embeddings.pt'

    if os.path.exists(raw_embedding_file):
        print(f"Raw embeddings found for {dataset}, loading...")
        return torch.load(raw_embedding_file)

    print(f"Generating new raw embeddings for {dataset} using ZhipuAI Embedding-3 model...")

    # 使用 ZhipuAI 的 Embedding-3 模型
    client = ZhipuAI(api_key="1f352109c4d23e86cdff130e2502db6c.Mhpnhp3n7gkEZf2B")  # 1bfaae6947c69fc9eb064e13e109ea85.08qo6VjeApAISVBR

    # 处理item元数据，生成嵌入
    embeddings = load_raw_item_embeddings(meta_file_path, asin_mapping_file, client)

    # 保存原始嵌入到文件
    torch.save(embeddings, raw_embedding_file)
    return embeddings


def seed_everything(seed):
    '''
    设置整个开发环境的seed
    :param seed:
    :param device:
    :return:
    '''
    import os
    import random
    import numpy as np

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def main():
    seed_everything(2024)
    train_data = pickle.load(open('../datasets/' + opt.dataset + '/train.txt', 'rb'))
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open('../datasets/' + opt.dataset + '/test.txt', 'rb'))
    train_data = Data(train_data, shuffle=True)
    test_data = Data(test_data, shuffle=False)
    opt.len_max = max(train_data.len_max, test_data.len_max)

    if opt.dataset in ['diginetica', 'yoochoose1_4', 'yoochoose1_64']:
        n_node = 43098 if opt.dataset == 'diginetica' else 37484
        raw_embeddings = None
        pretrained_embeddings = None
    elif opt.dataset in ['luxury_beauty', 'musical_instruments', 'prime_pantry']:
        n_node = 1438 if opt.dataset == 'luxury_beauty' else 10479 if opt.dataset == 'musical_instruments' else 4962
        raw_embeddings = llm_meta_driven_embedding(opt.dataset)
        # 将原始嵌入映射到 hiddenSize
        pretrained_embeddings = space_projector(raw_embeddings, opt.hiddenSize)
    else:
        n_node = 310
        pretrained_embeddings = None

    model = trans_to_cuda(SessionGraph(opt, n_node, pretrained_embeddings))

    # 设置模型的超参数
    model.temperature = opt.temperature
    model.num_negatives = opt.num_negatives
    model.contrastive_weight = opt.contrastive_weight

    start = time.time()
    best_result = {f'{metric}@{k}': 0 for metric in ['Recall', 'MRR', 'HitRate', 'Precision', 'NDCG'] for k in [5, 10, 20]}
    best_epoch = {key: 0 for key in best_result.keys()}
    bad_counter = 0

    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        logger.info('-------------------------------------------------------')
        logger.info('epoch: %d' % epoch)
        model.t = min(1, 0.8 + epoch / 10)

        results = train_test(model, train_data, test_data)
        flag = 0

        # Compare and save best results
        for key in results.keys():
            if results[key] >= best_result[key]:
                best_result[key] = results[key]
                best_epoch[key] = epoch
                flag = 1
        
        # 更新温度参数
        model.update_temperature()
        
        print('Current Result:')
        for k in [5, 10, 20]:
            print(f'K={k}: HitRate@{k}={results[f"HitRate@{k}"]:.4f}, MRR@{k}={results[f"MRR@{k}"]:.4f}, '
                  f'Precision@{k}={results[f"Precision@{k}"]:.4f}, Recall@{k}={results[f"Recall@{k}"]:.4f}, '
                  f'NDCG@{k}={results[f"NDCG@{k}"]:.4f}')
            logger.info(f'K={k}: HitRate@{k}={results[f"HitRate@{k}"]:.4f}, MRR@{k}={results[f"MRR@{k}"]:.4f}, '
                        f'Precision@{k}={results[f"Precision@{k}"]:.4f}, Recall@{k}={results[f"Recall@{k}"]:.4f}, '
                        f'NDCG@{k}={results[f"NDCG@{k}"]:.4f}')

        print('Best Result:')
        for k in [5, 10, 20]:
            print(f'K={k}: Best HitRate@{k}={best_result[f"HitRate@{k}"]:.4f} at epoch {best_epoch[f"HitRate@{k}"]}, '
                  f'Best MRR@{k}={best_result[f"MRR@{k}"]:.4f} at epoch {best_epoch[f"MRR@{k}"]}, '
                  f'Best Precision@{k}={best_result[f"Precision@{k}"]:.4f} at epoch {best_epoch[f"Precision@{k}"]}, '
                  f'Best Recall@{k}={best_result[f"Recall@{k}"]:.4f} at epoch {best_epoch[f"Recall@{k}"]}, '
                  f'Best NDCG@{k}={best_result[f"NDCG@{k}"]:.4f} at epoch {best_epoch[f"NDCG@{k}"]}')
            logger.info(f'K={k}: Best HitRate@{k}={best_result[f"HitRate@{k}"]:.4f} at epoch {best_epoch[f"HitRate@{k}"]}, '
                        f'Best MRR@{k}={best_result[f"MRR@{k}"]:.4f} at epoch {best_epoch[f"MRR@{k}"]}, '
                        f'Best Precision@{k}={best_result[f"Precision@{k}"]:.4f} at epoch {best_epoch[f"Precision@{k}"]}, '
                        f'Best Recall@{k}={best_result[f"Recall@{k}"]:.4f} at epoch {best_epoch[f"Recall@{k}"]}, '
                        f'Best NDCG@{k}={best_result[f"NDCG@{k}"]:.4f} at epoch {best_epoch[f"NDCG@{k}"]}')

        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break

    print('-------------------------------------------------------')
    logger.info('-------------------------------------------------------')
    end = time.time()
    print("Time Taken To Run：%d h %d m %d s" % ((end - start) // 3600, (end - start) % 3600 // 60, (end - start) % 60))
    logger.info("Time Taken To Run: %d h %d m %d s" % ((end - start) // 3600, (end - start) % 3600 // 60, (end - start) % 60))


if __name__ == '__main__':
    main()
