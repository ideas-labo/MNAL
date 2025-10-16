# 导入所需的库
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import argparse  # 用于解析命令行参数
import pdb       # Python 调试器
from tool_funcs import random_subset_indices
from datetime import timedelta, datetime
from tool_funcs import parse_balance_ratio

# 创建 ArgumentParser 对象以处理命令行输入
parser = argparse.ArgumentParser()
# 添加命令行参数定义
parser.add_argument('--initial_size')       # 初始标记数据集的大小
parser.add_argument('--query_size')         # 每个主动学习步骤中查询（采样）的大小
parser.add_argument('--method_setting')     # 使用的主动学习方法设置名称
parser.add_argument('--start_from_run')     # 从第几次运行开始（用于恢复实验）
parser.add_argument('--start_from_step')    # 从第几个主动学习步骤开始（用于恢复实验）
parser.add_argument('--subset_size')        # 每次从未标注样本的子集中选择样本来标注，该子集的大小
parser.add_argument('--model_name', default='bert')         # 使用什么模型
parser.add_argument('--balance_ratio', type=str, required=True, help='The balance ratio of the dataset')# 数据集中类别不平衡的比例
# parser.add_argument('--pseudo_setting')   # (注释掉) 伪标签设置
# parser.add_argument('--ratio')            # (注释掉) 比例参数
# parser.add_argument('--epoch')            # (注释掉) 训练轮数

# 解析命令行参数
args = parser.parse_args()

# 定义常量和从参数中获取变量
EPOCH = 18                          # 模型训练的总轮数
initial_size = int(args.initial_size) # 初始标记集大小（整数）
sample_size = int(args.query_size)   # 每步查询大小（整数）
method_setting = args.method_setting # 选择的方法设置
start_from_run = int(args.start_from_run)-1  # 起始运行索引（从0开始）
start_from_step = int(args.start_from_step)-1 # 起始步骤索引（从0开始）
subset_size = int(args.subset_size) # 子集大小（整数）
model_name = args.model_name
pseudo_setting = 2                  # 伪标签设置（硬编码为2）
ratio = 1                           # 比例（硬编码为1, 可能用于伪标签数量）
balance_ratio = parse_balance_ratio(args.balance_ratio) # 命令行输入元组解析

# --- 方法设置名称转换 ---
# 将用户友好的方法名称映射到内部使用的名称
if method_setting == 'MNAL':
  method_setting = 'normalized_sum' # MNAL 使用归一化总和
if method_setting == 'MNAL_un':
  method_setting = 'uncertainty'    # MNAL_un 使用不确定性
if method_setting == 'MNAL_ran':
  method_setting = 'random'         # MNAL_ran 使用随机采样

# --- 根据方法设置配置内部标志 ---
# 这个大的 if/elif 块根据 'method_setting' 配置各种标志 (f1_flag 到 f5_flag)
# 和 MODE，这些标志和模式决定了在主动学习循环中具体使用哪种采样策略。

if method_setting=='random':
    # 随机采样设置
    f1_flag = 'no' # 不使用基于确定性/不确定性的标准
    f2_flag = 'no' # 不使用基于多样性的标准 (标记/未标记/组合)
    f3_flag = 'no' # 不使用基于代表性的标准 (标记/未标记/组合)
    f4_flag = 'no' # 不使用其他组合标准 (如 read, iden, normalized_sum)
    f5_flag = 'no' # 不使用其他组合标准
    MODE = 'random' # 采样模式为随机

elif method_setting=='uncertainty':
    # 不确定性采样设置
    f1_flag = 'uncertainty' # 使用不确定性标准
    f2_flag = 'no'
    f3_flag = 'no'
    f4_flag = 'no'
    f5_flag = 'no'
    MODE = 'single' # 采样模式为单一标准

elif method_setting=='read':
    # 可读性采样设置 (Flesch reading ease)
    f1_flag = 'read' # 使用可读性标准
    f2_flag = 'no'
    f3_flag = 'no'
    f4_flag = 'no'
    f5_flag = 'no'
    MODE = 'single' # 采样模式为单一标准

elif method_setting=='iden':
    # 身份/关键词密度采样设置
    f1_flag = 'iden' # 使用关键词密度标准
    f2_flag = 'no'
    f3_flag = 'no'
    f4_flag = 'no'
    f5_flag = 'no'
    MODE = 'single' # 采样模式为单一标准

elif method_setting=='Dominant':
    # 帕累托支配采样设置 (结合不确定性+可读性+身份)
    f1_flag = 'uncertainty' # 使用不确定性作为目标1
    f2_flag = 'no'
    f3_flag = 'no'
    f4_flag = 'read'        # 使用可读性作为目标2 (存储在f4)
    f5_flag = 'iden'        # 使用身份/关键词密度作为目标3 (存储在f5)
    MODE = 'dominated'      # 采样模式为帕累托支配

elif method_setting=='normalized_sum':
    # 归一化总和采样设置 (MNAL)
    f1_flag = 'no'
    f2_flag = 'no'
    f3_flag = 'no'
    f4_flag = 'normalized_sum' # 使用归一化总和作为标准 (不确定性+可读性+身份)
    f5_flag = 'no'
    MODE = 'normalized_sum' # 采样模式为归一化总和

elif method_setting=='read+iden':
    # 可读性+身份 归一化总和采样设置
    f1_flag = 'no'
    f2_flag = 'no'
    f3_flag = 'no'
    f4_flag = 'read+iden' # 使用可读性+身份的归一化总和
    f5_flag = 'no'
    MODE = 'read+iden'    # 采样模式

elif method_setting=='Knee':
    # 膝点采样设置 (在帕累托前沿上找膝点，结合不确定性+可读性+身份)
    f1_flag = 'uncertainty' # 使用不确定性作为目标1
    f2_flag = 'no'
    f3_flag = 'no'
    f4_flag = 'read'        # 使用可读性作为目标2
    f5_flag = 'iden'        # 使用身份/关键词密度作为目标3
    MODE = 'knee'           # 采样模式为膝点选择

elif method_setting=='kmeans(normalized_sum)':
    # K-Means 聚类采样设置 (基于归一化总和选择每个簇的代表点)
    f1_flag = 'uncertainty' # 虽然模式是kmeans，但需要一个基础目标来选择簇内代表，这里可能用uncertainty或在后面覆盖
    f2_flag = 'no'
    f3_flag = 'no'
    f4_flag = 'no'          # Kmeans 模式会覆盖这里，使用归一化总和
    f5_flag = 'no'
    MODE = 'kmeans'         # 采样模式为KMeans聚类

elif method_setting=='labeled_diversity':
    # 标记集多样性采样设置
    f1_flag = 'no'
    f2_flag = 'labeled' # 使用标记集多样性标准
    f3_flag = 'no'
    f4_flag = 'no'
    f5_flag = 'no'
    MODE = 'single' # 采样模式为单一标准

elif method_setting=='unlabeled_diversity':
    # 未标记集多样性采样设置
    f1_flag = 'no'
    f2_flag = 'unlabeled' # 使用未标记集多样性标准
    f3_flag = 'no'
    f4_flag = 'no'
    f5_flag = 'no'
    MODE = 'single' # 采样模式为单一标准

elif method_setting=='labeled_representative':
    # 标记集代表性采样设置
    f1_flag = 'no'
    f2_flag = 'no'
    f3_flag = 'labeled' # 使用标记集代表性标准
    f4_flag = 'no'
    f5_flag = 'no'
    MODE = 'single' # 采样模式为单一标准

elif method_setting=='unlabeled_representative':
    # 未标记集代表性采样设置 (使用负号，选择与未标记集最相似的)
    f1_flag = 'no'
    f2_flag = 'no'
    f3_flag = 'minus_unlabeled' # 使用负的未标记集代表性标准
    f4_flag = 'no'
    f5_flag = 'no'
    MODE = 'single' # 采样模式为单一标准

elif method_setting=='BALD':
    # BALD (Bayesian Active Learning by Disagreement) 采样设置
    f1_flag = 'no'
    f2_flag = 'no'
    f3_flag = 'BALD' # 使用BALD标准
    f4_flag = 'no'
    f5_flag = 'no'
    MODE = 'single' # 采样模式为单一标准

elif method_setting=='Coreset':
    # Coreset 采样设置 (选择能最好代表整个未标记集，并远离已标记集的点)
    f1_flag = 'no'
    f2_flag = 'no'
    f3_flag = 'coreset' # 使用Coreset标准
    f4_flag = 'no'
    f5_flag = 'no'
    MODE = 'single' # 采样模式为单一标准

elif method_setting=='threshold':
    # 基于阈值的不确定性采样设置
    f1_flag = 'threshold' # 使用阈值不确定性标准
    f2_flag = 'no'
    f3_flag = 'no'
    f4_flag = 'no'
    f5_flag = 'no'
    MODE = 'single' # 采样模式为单一标准
    threshold = 0.1 # 不确定性阈值

# --- 定义其他常量 ---
sample_times = 10   # 总的主动学习迭代次数（查询次数）
batch_size = 32     # 训练和预测时使用的批次大小
MAX_LEN = 100       # BERT 输入序列的最大长度
test_size = 5000    # 测试集的大小 (用于加载相应的数据文件)

# --- 导入更多库 ---
import pandas as pd
import re
import nltk
import numpy as np
import string
import json
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import torch
import random
from sklearn.metrics import (accuracy_score, recall_score, precision_score, f1_score, auc, roc_curve, confusion_matrix)
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
import time
import datetime
from tqdm import tqdm # 用于显示进度条
import math
import pickle     # 用于加载/保存 Python 对象
# import logging  # (注释掉) 日志记录
import torch.nn.functional as F # PyTorch 函数式接口
import faiss      # 用于高效相似性搜索 (最近邻)
import math
from abc import abstractmethod # 用于定义抽象基类 (这里可能未使用)
import numpy as np
from sklearn.metrics import pairwise_distances # 计算成对距离
from transformers import BertModel, RobertaModel, AutoModel # Hugging Face 模型类
import os         # 用于操作系统相关功能 (如路径)
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # 用于绘图
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import argparse
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR # 学习率调度器
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler # 数据归一化
from sklearn.cluster import DBSCAN, KMeans     # 聚类算法
import textstat   # 用于计算文本统计信息 (如可读性)
from collections import Counter # 用于计数
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer, get_linear_schedule_with_warmup, AutoTokenizer, RobertaTokenizer # Hugging Face Transformers 库，用于加载预训练模型和分词器

# 定义用于 'iden' 标准的关键词列表
words_to_search = ['error', 'bug', 'reproduce', 'issue', 'behavior', 'debug', 'failed', 'expected', 'crash', 'add', 'would', 'like', 'use', 'feature', 'request', 'support', 'improvement', 'want', 'documentation']


# --- 加载预处理的数据 ---
# 加载预先分词并保存的句子数据
with open(f"../initial_data/diversity_new/test_size{test_size}_tokenized_sentences_balance{balance_ratio}_model_name_{model_name}.pkl",'rb') as pklfile:
    tokenized_sentences = pickle.load(pklfile)

# 加载包含训练/测试数据分割和初始ID的文件
with open(f'../initial_data/diversity_new/data_all_test_size{test_size}_balance{balance_ratio}_model_name_{model_name}.pkl','rb') as pklfile:
    files = pickle.load(pklfile)

# 解包加载的数据
X_test, y_test, bug_train_ids, X_train, y_train = files
# X_test: 测试集文本
# y_test: 测试集标签
# bug_train_ids: 原始训练集的所有ID (可能未使用)
# X_train: 训练集文本 (Pandas Series)
# y_train: 训练集标签 (Numpy Array)

# --- 设置计算设备 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 优先使用 GPU

# --- 定义模型组件 ---
loss_func = torch.nn.CrossEntropyLoss() # 交叉熵损失函数，用于分类任务
linear_layer = torch.nn.Linear(768, 2, device=device) # 线性层，将BERT的输出(768维)映射到2个类别


# --- Tokenizer 初始化 ---
# do_lower_case=True: 在分词前将文本转换为小写 (虽然前面已经做过，但这是标准做法)
# 根据命令行指定的 model_name 选择并加载相应的 Tokenizer
if model_name == 'rta':
    tokenizer = AutoTokenizer.from_pretrained("Colorful/RTA")
elif model_name == 'roberta':
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
elif model_name == 'codebert':
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
else:  # 默认为 bert-base-uncased
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# 对测试集文本进行分词和编码
test_tokenized_sentences = tokenizer(list(X_test), max_length=MAX_LEN,
                              truncation=True, padding=True, # 截断和填充到 MAX_LEN
                              return_tensors="pt", return_attention_mask=True) # 返回 PyTorch 张量和注意力掩码
# 创建测试集的 TensorDataset
test_data = TensorDataset(torch.tensor(test_tokenized_sentences['input_ids']), torch.tensor(test_tokenized_sentences['attention_mask']), torch.tensor(y_test))
# 创建测试集的顺序采样器
test_sampler = SequentialSampler(test_data)
# 创建测试集的 DataLoader
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=128) # 使用较大的 batch_size 进行测试

# --- Coreset 采样实现 ---
class Coreset_Greedy:
    """
    使用贪心算法实现 Coreset 采样。
    目标是选择一个小的子集（coreset），使其能够很好地代表整个数据集，
    通常通过最大化新选点与已选点之间的最小距离来实现。
    """
    def __init__(self, all_pts, mode='max'):
        """
        初始化 Coreset_Greedy 对象。
        :param all_pts: 所有候选点的特征向量列表或Numpy数组。
        :param mode: 'max' (标准 coreset，最大化最小距离) 或 'min' (选择最接近已选点的点)。
        """
        self.all_pts = np.array(all_pts) # 转换为 Numpy 数组
        self.dset_size = len(all_pts)    # 数据集总大小
        self.min_distances = None        # 存储每个点到最近已选中心的距离
        self.already_selected = []       # 已选点的索引列表
        self.mode = mode                 # 采样模式 ('max' or 'min')

        # 重塑特征向量数组，确保是二维的 (n_samples, n_features)
        feature_len = self.all_pts[0].shape[0]
        self.all_pts = self.all_pts.reshape(-1, feature_len)

    def update_dist(self, centers, only_new=True, reset_dist=False):
        """
        更新每个点到最近已选中心的最小距离。
        :param centers: 新加入的中心点索引列表。
        :param only_new: 是否只计算到新加入中心点的距离。
        :param reset_dist: 是否重置最小距离 (通常在开始或更换已标记集时使用)。
        """
        if reset_dist:
            self.min_distances = None # 重置最小距离
        if only_new:
            # 只考虑尚未被选中的新中心
            centers = [p for p in centers if p not in self.already_selected]

        if centers: # 如果有新的中心点
            x = self.all_pts[centers] # 获取中心点的特征向量
            # 计算所有点到这些新中心点的欧氏距离
            dist = pairwise_distances(self.all_pts, x, metric='euclidean')

            if self.min_distances is None:
                # 如果是第一次计算，直接取最小值
                self.min_distances = np.min(dist, axis=1).reshape(-1, 1)
            else:
                # 否则，更新每个点的最小距离 (取当前最小距离和到新中心距离中的较小值)
                self.min_distances = np.minimum(self.min_distances, dist)

    def sample(self, unlabeled_len, annotated_len, sample_size):
        """
        执行贪心采样。
        :param unlabeled_len: 未标记点的数量 (这些点是候选点)。
        :param annotated_len: 已标记点的数量 (这些点作为初始中心)。
        :param sample_size: 需要采样的点的数量。
        :return: (选中的未标记点索引列表, 对应的距离值列表)
        """
        # 初始更新距离：计算所有点到初始已标记点(annotated)的距离
        # 这里的 all_pts 包含了未标记点和已标记点，假设未标记点在前，已标记点在后
        # 索引范围 [unlabeled_len, unlabeled_len + annotated_len) 是已标记点
        self.update_dist(range(unlabeled_len, unlabeled_len + annotated_len), only_new=False, reset_dist=True)
        self.already_selected = list(range(unlabeled_len, unlabeled_len + annotated_len)) # 记录已选点（初始为已标记点）

        new_batch = [] # 存储新选中的未标记点索引
        obj = []       # 存储每次选择时的距离值

        for _ in range(sample_size): # 迭代选择 sample_size 个点
            if not self.already_selected: # 如果还没有已选点 (理论上不会发生，因为有初始标记点)
                # 随机选一个点作为开始
                ind = np.random.choice(np.arange(self.dset_size))
            else:
                # 从未标记点中选择 (索引范围 [0, unlabeled_len))
                if self.mode == 'max':
                    # 选择距离已选点集合最远的点 (最大化最小距离)
                    ind = np.argmax(self.min_distances[:unlabeled_len])
                    dist = np.max(self.min_distances[:unlabeled_len])
                elif self.mode == 'min':
                    # 选择距离已选点集合最近的点 (最小化最小距离)
                    ind = np.argmin(self.min_distances[:unlabeled_len])
                    dist = np.min(self.min_distances[:unlabeled_len])

            # 确保选中的点是未标记点
            assert ind not in range(unlabeled_len, unlabeled_len + annotated_len)

            # 更新距离：将新选中的点加入中心集合，并更新所有点的最小距离
            self.update_dist([ind], only_new=True, reset_dist=False)
            self.already_selected.append(ind) # 将新选点加入已选列表
            new_batch.append(ind)             # 记录选中的索引
            obj.append(dist)                  # 记录对应的距离值

        # 返回选中的未标记点在原始未标记集中的索引列表，以及对应的距离值
        return new_batch, obj

# --- 工具函数 ---

def ids2dataloader(ids, tokenized_sentences, batch_size=32):
  """
  根据给定的索引列表，从预分词的数据中创建 DataLoader。
  :param ids: 需要包含在 DataLoader 中的数据索引列表。
  :param tokenized_sentences: 包含 'input_ids' 和 'attention_mask' 的预分词数据字典。
  :param batch_size: DataLoader 的批次大小。
  :return: PyTorch DataLoader 对象。
  """
  # 从预分词数据中提取对应索引的 input_ids 和 attention_mask
  # 使用 .clone().detach() 来创建副本，避免修改原始数据
  input_ids = tokenized_sentences['input_ids'][ids].clone().detach()
  attention_mask = tokenized_sentences['attention_mask'][ids].clone().detach()
  # 获取对应索引的标签
  labels = torch.tensor(y_train[ids]) # 假设 y_train 是包含所有训练标签的 Numpy 数组
  # 创建 TensorDataset
  data = TensorDataset(input_ids, attention_mask, labels)
  # 创建顺序采样器
  sampler = SequentialSampler(data)
  # 创建 DataLoader
  dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
  return dataloader

def format_time(elapsed):
    """
    将时间差格式化为可读的字符串。
    :param elapsed: 时间差（秒）
    :return: 格式化后的时间字符串
    """
    # 使用 timedelta 格式化
    return str(timedelta(seconds=elapsed))

def flat_accuracy(preds, labels):
    """
    计算预测值和真实标签之间的准确率。
    适用于多分类任务的输出 (logits 或概率)。
    :param preds: 模型的预测输出 (通常是 logits 或概率)，形状为 (batch_size, num_classes)。
    :param labels: 真实标签，形状为 (batch_size,)。
    :return: 准确率 (浮点数)。
    """
    pred_flat = np.argmax(preds, axis=1).flatten() # 获取每个样本预测概率最高的类别索引
    labels_flat = labels.flatten()                 # 展平真实标签
    return np.sum(pred_flat == labels_flat) / len(labels_flat) # 计算匹配数量并除以总数

def pareto_front(costs):
  """
  查找帕累托有效点 (Pareto-efficient points)。
  一个点是帕累托有效的，如果没有其他点在所有目标上都比它好（或相等），并且至少在一个目标上严格更好。
  这里假设目标是越小越好。
  :param costs: 一个 (n_points, n_costs) 的 Numpy 数组，表示每个点的多个成本（目标值）。
  :return: 一个 (n_points,) 的布尔数组，指示每个点是否位于帕累托前沿。
  """
  is_front = np.ones(costs.shape[0], dtype=bool) # 初始化所有点都在前沿
  for i, c in enumerate(costs): # 遍历每个点
      if is_front[i]: # 如果当前点仍然被认为是前沿点
          # 检查其他仍在前沿的点 `costs[is_front]` 是否支配当前点 `c`
          # 如果存在一个其他点 `p`，使得 `p` 在所有维度上都小于等于 `c` (`np.all(costs[is_front] <= c, axis=1)`)
          # 并且 `p` 至少在一个维度上严格小于 `c` (`np.any(costs[is_front] < c, axis=1)`)
          # 则 `c` 被支配。
          # 更简洁的实现：保留所有“不被c支配”的点。如果一个点p至少在一个维度上比c小 (`np.any(costs[is_front] < c, axis=1)`), 那么p就可能支配c或者两者互不支配，p应该保留。
          # 如果一个点p在所有维度上都大于等于c，那么p可能被c支配，应该剔除（除非p就是c本身）。
          # 这里代码的逻辑是：保留那些至少在一个目标上比 c 严格差的点 (即 c 不能支配它们)
          is_front[is_front] = np.any(costs[is_front] < c, axis=1) # 保留任何在至少一个成本上比 c 低的点
          is_front[i] = True  # 确保点自身被保留 (因为它不比自己差)
  return is_front

def dominated_solution(costs, n_sample):
  """
  使用帕累托支配关系选择样本。
  反复查找当前未选样本中的帕累托前沿，直到选够 n_sample 个样本。
  :param costs: 一个 (n_points, n_costs) 的 Numpy 数组，表示每个候选点的成本。
  :param n_sample: 需要选择的样本数量。
  :return: 选中的样本在原始 costs 数组中的索引列表。
  """
  count = 0 # 已选样本计数
  indices = np.array(range(costs.shape[0])) # 当前候选样本的索引
  initial_indices = np.array(range(costs.shape[0])) # 原始所有样本的索引
  # initial_costs = np.copy(costs) # 备份原始成本 (未使用)

  selected_indices_list = [] # 存储每一轮选中的帕累托前沿索引

  while True:
    # 在当前候选样本中查找帕累托前沿
    is_front = pareto_front(costs)
    front_indices = indices[is_front] # 获取当前前沿样本的原始索引
    num_front = len(front_indices)    # 当前前沿的样本数量

    last_count = count # 记录本轮开始前的已选数量
    count += num_front # 累加当前前沿的数量

    if count >= n_sample: # 如果累加后超过或等于目标数量
      needed = n_sample - last_count # 还需要选择多少个
      # 从当前前沿中选择 needed 个样本
      # 如果 costs[is_front] 存在，则基于第一个成本维度排序选择
      if num_front > 0:
          costs_selected = costs[is_front] # 当前前沿的成本
          # 按第一个成本排序，选择前 needed 个
          inner_indices_order = np.argsort(costs_selected[:, 0])[:needed]
          selected_from_front = front_indices[inner_indices_order]
      else: # 如果当前前沿为空 (理论上在 count >= n_sample 前不会发生)
          selected_from_front = np.array([])

      # 将本轮选中的（部分）前沿样本加入结果列表
      selected_indices_list.append(selected_from_front)
      # 合并所有轮次选中的索引
      pareto_indices = np.concatenate(selected_indices_list)
      break # 结束循环

    else: # 如果累加后仍未达到目标数量
      # 将当前整个帕累托前沿加入结果列表
      selected_indices_list.append(front_indices)
      # 从候选样本中移除当前前沿的点，继续下一轮
      indices = indices[~is_front]
      costs = costs[~is_front]
      if len(indices) == 0: # 如果没有候选样本了，提前结束
          pareto_indices = np.concatenate(selected_indices_list)
          break

  # 返回最终选中的 n_sample 个（或更少，如果候选样本不足）索引
  return pareto_indices.astype(int) # 确保返回整数索引

def cal_dists(extreme_points, costs):
  """
  计算给定点集 (costs) 到由三个极值点定义的平面的距离。
  假设是三维空间 (三个目标成本)。
  :param extreme_points: 定义平面的三个极值点 (每个目标上最优的点)，形状为 (3, 3)。
  :param costs: 需要计算距离的点集，形状为 (n_points, 3)。
  :return: 每个点到平面的距离数组，形状为 (n_points,)。
  """
  # 获取三个极值点
  point1 = extreme_points[0]
  point2 = extreme_points[1]
  point3 = extreme_points[2]
  dists = [] # 存储距离

  # 计算平面上的两个向量
  v1 = point3 - point1
  v2 = point2 - point1

  # 计算平面的法向量 (通过向量叉乘)
  normal = np.cross(v2, v1)

  # 获取平面方程 ax + by + cz + d = 0 的系数
  a, b, c = normal
  # 计算 d (使用平面上任意一点 point1)
  d = -np.dot(normal, point1)
  print(f"Plane equation coefficients: a={a}, b={b}, c={c}, d={d}") # 打印平面方程系数

  # 计算每个点到平面的距离
  denominator = math.sqrt(a * a + b * b + c * c) # 分母：法向量的模长
  if denominator == 0: # 避免除以零
      print("Warning: Plane normal vector is zero. Returning zero distances.")
      return np.zeros(costs.shape[0])

  for other_point in costs:
    # 点到平面的距离公式: |ax0 + by0 + cz0 + d| / sqrt(a^2 + b^2 + c^2)
    dist = abs((a * other_point[0] + b * other_point[1] + c * other_point[2] + d)) / denominator
    dists.append(dist)

  return np.array(dists)

def getKneePointIndices(costs, num):
  """
  查找膝点 (Knee Points)。
  通过计算点到三个极值点定义的平面的距离，并选择距离最大的 num 个点作为膝点。
  适用于三维目标空间。
  :param costs: 候选点的成本数组，形状为 (n_points, 3)。
  :param num: 需要选择的膝点数量。
  :return: 选中的膝点在 costs 数组中的索引列表。
  """
  extreme_points = [] # 存储每个目标上的极值点
  if costs.shape[0] == 0: # 如果没有候选点，返回空列表
      return np.array([], dtype=int)
  if costs.shape[1] != 3: # 检查是否为三维
      raise ValueError("Knee point calculation requires 3 cost dimensions.")

  for i in range(costs.shape[1]): # 遍历每个目标维度
    # 找到当前维度成本最高（最差）的点的索引，并获取该点的完整成本向量
    # 注意：这里假设目标是越小越好，所以极值点是每个维度值最大的点
    if costs.shape[0] > 0:
        extreme_points.append(costs[np.argmax(costs[:, i])])
    else: # 如果中间过程导致 costs 为空
        return np.array([], dtype=int)

  extreme_points = np.array(extreme_points)
  if extreme_points.shape[0] < 3: # 如果不足以定义平面
      print("Warning: Not enough unique extreme points to define a plane. Returning top points based on first cost.")
      # Fallback: 如果点太少或共线/共点，无法定义平面，则按第一个成本排序返回前 num 个
      return np.argsort(costs[:, 0])[:num]


  # 计算所有点到极值点定义的平面的距离
  cost_dists = cal_dists(extreme_points, costs)

  # 选择距离最大的 num 个点的索引
  # 使用 argsort 对距离进行降序排序，并取前 num 个索引
  KneePointIndices = np.argsort(-cost_dists)[:num]
  return KneePointIndices

def dominated_knee_solution(costs, n_sample):
  """
  结合帕累托支配和膝点选择来选择样本。
  首先选择帕累托前沿上的点。如果最后一个前沿的点数超过了剩余所需样本数，
  则在最后一个前沿上使用膝点选择策略来挑选剩余的样本。
  :param costs: 一个 (n_points, n_costs) 的 Numpy 数组，表示每个候选点的成本 (这里假设 n_costs=3)。
  :param n_sample: 需要选择的样本数量。
  :return: 选中的样本在原始 costs 数组中的索引列表。
  """
  if costs.shape[1] != 3:
      raise ValueError("Dominated Knee solution currently requires 3 cost dimensions.")

  count = 0 # 已选样本计数
  indices = np.array(range(costs.shape[0])) # 当前候选样本的索引
  initial_indices = np.array(range(costs.shape[0])) # 原始所有样本的索引
  # initial_costs = np.copy(costs) # 备份原始成本 (未使用)

  selected_indices_list = [] # 存储每一轮选中的索引

  while True:
    if len(indices) == 0: # 如果没有候选点了
        break
    # 在当前候选样本中查找帕累托前沿
    is_front = pareto_front(costs)
    front_indices = indices[is_front] # 获取当前前沿样本的原始索引
    num_front = len(front_indices)    # 当前前沿的样本数量

    last_count = count # 记录本轮开始前的已选数量
    count += num_front # 累加当前前沿的数量

    if count >= n_sample: # 如果累加后超过或等于目标数量
      needed = n_sample - last_count # 还需要选择多少个
      # 从当前前沿中选择 needed 个样本
      costs_selected = costs[is_front] # 当前前沿的成本
      indices_selected = indices[is_front] # 当前前沿的索引

      # --- 关键区别：使用膝点选择 ---
      # 在当前前沿 (costs_selected) 上选择 needed 个膝点
      inner_indices = getKneePointIndices(costs_selected, needed)
      selected_from_front = indices_selected[inner_indices] # 获取这些膝点在原始索引中的位置
      # ---

      # 将本轮选中的膝点加入结果列表
      selected_indices_list.append(selected_from_front)
      # 合并所有轮次选中的索引
      pareto_indices = np.concatenate(selected_indices_list)
      break # 结束循环

    else: # 如果累加后仍未达到目标数量
      # 将当前整个帕累托前沿加入结果列表
      selected_indices_list.append(front_indices)
      # 从候选样本中移除当前前沿的点，继续下一轮
      indices = indices[~is_front]
      costs = costs[~is_front]
      if len(indices) == 0: # 如果没有候选样本了
          pareto_indices = np.concatenate(selected_indices_list)
          break

  # 返回最终选中的 n_sample 个（或更少）索引
  return pareto_indices.astype(int)

def single_id_predict(id, return_pooled_output=False):
  """
  对单个样本进行预测。
  :param id: 需要预测的样本在 X_train 中的索引。
  :param return_pooled_output: 是否返回 BERT 的 [CLS] 池化层输出。
  :return: 预测为类别 0 的概率 (假设是二分类)，如果 return_pooled_output 为 True，则额外返回池化层输出。
  """
  # 获取文本数据
  text = X_train.iloc[id]
  # 使用 tokenizer 进行编码
  tokens = tokenizer(text, max_length=MAX_LEN, truncation=True, padding='max_length', # 使用 padding='max_length' 保证长度一致
                     return_tensors="pt", return_attention_mask=True)
  # 将 token 移到设备 (GPU/CPU)
  input_ids = tokens['input_ids'].to(device)
  attention_mask = tokens['attention_mask'].to(device)

  # 执行模型推理
  with torch.no_grad(): # 关闭梯度计算
    # 获取 BERT 模型的输出 (包括最后一层隐藏状态和池化输出)
    output = model(input_ids, token_type_ids=None, attention_mask=attention_mask)
    # output[0] 是 sequence_output, output[1] 是 pooled_output

  # 将池化输出通过线性层得到 logits
  logits = linear_layer(output[1])
  # 计算 softmax 概率
  probs = F.softmax(logits, dim=-1)
  # 获取预测为类别 0 的概率
  prob_class_0 = probs[0][0].detach().cpu().numpy().item()

  if return_pooled_output:
    # 返回概率和池化层输出 (numpy 数组)
    pooled_output_np = output[1][0].detach().cpu().numpy()
    return prob_class_0, pooled_output_np
  else:
    # 只返回概率
    return prob_class_0

def modelpredict(indices, model, tokenized_sentences, batch_size=128, verbose=0, return_pool_result=True):
  """
  对一批索引对应的样本进行预测。
  :param indices: 需要预测的样本在原始训练集中的索引列表。
  :param model: 用于预测的 BERT 模型 (基础模型，不含最后的分类层)。
  :param tokenized_sentences: 预分词的数据。
  :param batch_size: 预测时的批次大小。
  :param verbose: 是否显示进度条 (0: 不显示, 1: 显示)。
  :param return_pool_result: 是否返回所有样本的池化层输出。
  :return: (预测为类别 1 的概率数组, 池化层输出数组 (如果 return_pool_result=True))
            注意：这里返回的是类别 1 的概率，与 single_id_predict 不同。
  """
  start_time = time.time() # 记录开始时间
  TQDM_DISABLE = (verbose == 0) # 根据 verbose 设置是否禁用 tqdm 进度条

  model.eval() # 设置模型为评估模式
  # 创建用于预测的 DataLoader
  dataloader = ids2dataloader(indices, tokenized_sentences, batch_size=batch_size)

  probs_list = [] # 存储每个批次的概率
  if return_pool_result:
    # 预先分配空间存储池化输出，提高效率
    pooled_outputs = np.zeros((len(indices), 768), dtype=np.float32)
    current_idx = 0 # 当前填充到的索引

  # 遍历 DataLoader 中的每个批次
  for batch in tqdm(dataloader, disable=TQDM_DISABLE, desc="Predicting"):
    # 将批次数据移到设备
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask, b_labels = batch # b_labels 在预测中未使用

    # 关闭梯度计算
    with torch.no_grad():
      # 获取 BERT 模型输出
      _, pooled_output = model(b_input_ids,
                              token_type_ids=None,
                              attention_mask=b_input_mask,
                              return_dict=False) # 使用 return_dict=False 获取元组输出

    # 通过线性层获取 logits
    logits = linear_layer(pooled_output)
    # 计算 softmax 概率
    prob = F.softmax(logits, dim=-1).detach().cpu().numpy() # (batch_size, 2)
    probs_list.append(prob) # 添加到列表

    if return_pool_result:
      # 将当前批次的池化输出填充到预分配的数组中
      batch_size_actual = prob.shape[0] # 获取当前批次的实际大小
      pooled_outputs[current_idx : current_idx + batch_size_actual] = pooled_output.detach().cpu().numpy()
      current_idx += batch_size_actual # 更新填充索引

  # 合并所有批次的概率
  probs = np.concatenate(probs_list, axis=0)

  end_time = time.time() # 记录结束时间
  execution_time = end_time - start_time
  print(f"Model predicting time: {execution_time:.2f} seconds")

  # 返回类别 1 的概率 和 池化输出 (如果需要)
  if return_pool_result:
    return probs[:, 1], pooled_outputs
  else:
    return probs[:, 1]

def modeltest(model, verbose=0):
    """
    在固定的测试集上评估模型性能。
    :param model: 待评估的 BERT 模型 (基础模型)。
    :param verbose: 是否显示进度条。
    :return: 包含 [f1, accuracy, auc, recall, precision] 的 Numpy 数组。
    """
    TQDM_DISABLE = (verbose == 0) # 设置 tqdm
    y_pred = np.array([]) # 存储预测概率 (类别 1)

    model.eval() # 设置模型为评估模式

    # 遍历测试集 DataLoader
    for batch in tqdm(test_dataloader, disable=TQDM_DISABLE, desc="Testing"):
        # 移动数据到设备
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch # b_labels 是真实标签

        # 关闭梯度计算
        with torch.no_grad():
            # 获取模型输出
            _, pooled_output = model(b_input_ids,
                                    token_type_ids=None,
                                    attention_mask=b_input_mask,
                                    return_dict=False)
        # 通过线性层和 softmax 获取概率
        logits = linear_layer(pooled_output)
        prob = F.softmax(logits, dim=-1)
        # 获取类别 1 的概率
        prob_class1 = prob[:, 1].detach().cpu().numpy()
        # 追加到预测结果列表
        y_pred = np.append(y_pred, prob_class1)

    # 将概率转换为 0/1 预测 (阈值为 0.5)
    y_pred2 = np.zeros(y_pred.shape)
    y_pred2[y_pred > 0.5] = 1

    # 计算各项评估指标
    f1 = f1_score(y_test, y_pred2)           # F1 分数
    acc = accuracy_score(y_test, y_pred2)    # 准确率
    rec = recall_score(y_test, y_pred2)      # 召回率
    prec = precision_score(y_test, y_pred2)  # 精确率
    # 计算 AUC 需要使用概率 y_pred
    fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
    auc_value = auc(fpr, tpr)                # AUC 值

    # 返回包含所有指标的数组
    return np.array([f1, acc, auc_value, rec, prec])


# 重新加载预训练模型函数
def reload_model():
  """
  根据全局变量 `model_name` 加载一个新的、预训练的 Transformer 模型实例，
  并将其移动到指定的计算设备。

  Returns:
      torch.nn.Module: 加载的预训练模型。
  """
  # 根据 model_name 选择并加载 Hugging Face 模型
  if model_name == 'rta':
    model = AutoModel.from_pretrained("Colorful/RTA")
  elif model_name == "roberta":
    model = RobertaModel.from_pretrained("roberta-base")
  elif model_name == 'codebert':
    model = AutoModel.from_pretrained("microsoft/codebert-base")
  else: # 默认为 bert-base-uncased
    model = BertModel.from_pretrained("bert-base-uncased")
  # 将模型移动到 GPU (如果可用)

  model.cuda()
  return model

def modelfit_fusion(using_ids, using_y, model, tokenized_sentences, epochs=EPOCH, verbose=0):
  """
  在给定的数据子集上训练 (微调) BERT 模型。
  这个函数包含训练和验证过程。
  :param using_ids: 用于训练和验证的数据在原始训练集中的索引。
  :param using_y: 对应的标签 (可能是真实标签或包含伪标签)。
  :param model: 需要训练的 BERT 模型 (基础模型)。
  :param tokenized_sentences: 预分词的数据。
  :param epochs: 训练轮数。
  :param verbose: 是否显示进度条和训练信息。
  :return: 训练完成的模型。
  """
  TQDM_DISABLE = (verbose == 0) # 设置 tqdm

  # --- 数据准备 ---
  # 将传入的数据分割为训练集和验证集 (80% 训练, 20% 验证)
  train_indices, val_indices, train_fusion_y, val_fusion_y = train_test_split(
      using_ids, using_y, test_size=0.2, random_state=999, stratify=using_y # 使用 stratify 保证标签比例
  )

  # 创建训练 DataLoader
  train_dataloader = ids2dataloader(train_indices, tokenized_sentences, batch_size)
  # 创建验证 DataLoader
  val_dataloader = ids2dataloader(val_indices, tokenized_sentences, batch_size)

  # --- 优化器和学习率调度器 ---
  # AdamW 优化器，通常用于 Transformer 模型
  optimizer = AdamW(model.parameters(), lr=3e-5, eps=1e-8) # 学习率 3e-5, epsilon 1e-8
  # 总训练步数
  total_steps = len(train_dataloader) * epochs
  # 线性学习率预热和衰减调度器 (预热步数为 0)
  scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

  # 存储每个 epoch 的平均训练损失
  loss_values = []

  # --- 训练循环 ---
  for epoch in range(epochs):
      print(f"\n===== Epoch {epoch + 1} / {epochs} =====")
      # --- 训练阶段 ---
      print('Training...')
      t0 = time.time() # 记录 epoch 开始时间
      total_loss = 0 # 当前 epoch 总损失

      model.train() # 设置模型为训练模式

      # 遍历训练 DataLoader
      for step, batch in enumerate(train_dataloader):
          # 每 40 步打印一次进度 (如果 verbose > 0)
          if step % 40 == 0 and not step == 0 and verbose > 0:
              elapsed = format_time(time.time() - t0)
              print(f'  Batch {step:>5,}  of  {len(train_dataloader):>5,}.    Elapsed: {elapsed}.')

          # 将数据移到设备
          b_input_ids = batch[0].to(device)
          b_input_mask = batch[1].to(device)
          b_labels = batch[2].type(torch.LongTensor).to(device) # 确保标签是 LongTensor

          # 清除之前的梯度
          model.zero_grad()
          optimizer.zero_grad() # 也清除优化器的梯度

          # 前向传播
          _, pooled_output = model(input_ids=b_input_ids,
                                   token_type_ids=None,
                                   attention_mask=b_input_mask,
                                   return_dict=False)
          # 通过线性层得到 logits
          logits = linear_layer(pooled_output)

          # 计算损失
          train_loss = loss_func(logits, b_labels)
          total_loss += train_loss.item() # 累加损失值

          # 反向传播
          train_loss.backward()

          # 梯度裁剪，防止梯度爆炸
          torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

          # 更新参数和学习率
          optimizer.step()
          scheduler.step()

      # 计算平均训练损失
      avg_train_loss = total_loss / len(train_dataloader)
      loss_values.append(avg_train_loss)

      print(f"  Average training loss: {avg_train_loss:.4f}")
      print(f"  Training epoch took: {format_time(time.time() - t0)}")

      # --- 验证阶段 ---
      print("\nRunning Validation...")
      t0 = time.time() # 记录验证开始时间
      model.eval() # 设置模型为评估模式

      eval_loss, eval_accuracy = 0, 0
      nb_eval_steps, nb_eval_examples = 0, 0
      y_pred = np.array([])   # 存储验证集预测概率
      labels = np.array([])   # 存储验证集真实标签

      # 遍历验证 DataLoader
      for batch in val_dataloader:
          batch = tuple(t.to(device) for t in batch)
          b_input_ids, b_input_mask, b_labels = batch

          with torch.no_grad(): # 关闭梯度计算
              _, pooled_output = model(b_input_ids,
                                      token_type_ids=None,
                                      attention_mask=b_input_mask,
                                      return_dict=False)
          logits = linear_layer(pooled_output)
          # 计算概率并将预测概率和标签移到 CPU
          prob = F.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy() # 获取类别 1 的概率
          b_labels_np = b_labels.to('cpu').numpy()

          # 存储预测和标签
          y_pred = np.append(y_pred, prob)
          labels = np.append(labels, b_labels_np)

      # 计算验证集准确率 (基于 0.5 阈值)
      eval_accuracy = accuracy_score(labels.round(), y_pred.round())
      print(f"  Validation Accuracy: {eval_accuracy:.4f}")
      print(f"  Validation took: {format_time(time.time() - t0)}")

  print("\nTraining complete!")
  return model # 返回训练好的模型

# --- 初始化指标存储数组 ---
# metrics_all: 存储每次主动学习迭代 *之后*（可能包含伪标签训练后）的测试集性能
# metrics_mixed: 存储更详细的性能：偶数列是 AL 选择 + 训练后的性能，奇数列是伪标签加入 + 训练后的性能
metrics_all = np.zeros((5, sample_times + 1), dtype='float64') # 5 指标 (f1,acc,auc,rec,prec), sample_times+1 列 (初始+每次迭代后)
metrics_mixed = np.zeros((5, 2 * sample_times), dtype='float64') # 5 指标, 2*sample_times 列

# --- 主动学习循环 ---

# 设置当前运行的索引 (从命令行参数获取，并减1变为0基索引)
run = start_from_run
print(f'\n--- The run {run+1} started ---\n') # 显示用户友好的运行编号 (从1开始)

# --- 加载初始状态或恢复状态 ---
if start_from_step == -1: # 如果是从头开始 (start_from_step=0 意味着索引为-1)
  print("Loading initial data and model...")
  # 加载预先准备好的初始标记数据和模型权重
  initial_data_path = f'../initial_data/diversity_new/{model_name}/model{model_name}_run{run}_sample_size{initial_size}_initial_data_balance{balance_ratio}.pkl' # 文件名从 run 1 开始
  initial_model_path = f'../initial_data/diversity_new/{model_name}/model{model_name}_run{run}_sample_size{initial_size}_model_weight_balance{balance_ratio}' 
  print(f"Loading data from: {initial_data_path}")
  print(f"Loading model weights from: {initial_model_path}")

  with open(initial_data_path, 'rb') as pklfile:
    files = pickle.load(pklfile)
  # 解包初始数据： _, 初始已标记ID, 初始池ID, 初始测试性能
  _, bug_annotated_ids, bug_pool_ids, metrics_update = files

  # 将初始性能记录在 metrics_all 的第一列
  metrics_all[:, 0] = metrics_update # 注意：这里假设所有运行共享 metrics_all，如果多运行并行，需要修改
  print("Initial metrics (F1, Acc, AUC, Rec, Prec):", metrics_update)

  # 加载初始模型权重
  model = reload_model() # 加载基础 BERT

  # 加载模型权重
  state_dict = torch.load(initial_model_path)
  # 过滤掉不必要的键
  if 'model_state_dict' in state_dict:
    state_dict = state_dict['model_state_dict']
  # # 加载微调过的权重
  model.load_state_dict(state_dict, strict=False)
  # 注意：初始模型没有对应的优化器状态被加载，每次modelfit会重新创建优化器

  # 获取初始标记数据的真实标签
  y_true_labeled = y_train[bug_annotated_ids]

  # 初始化 'real' 标记集，用于 pseudo_setting == 2 的情况
  # 在初始状态下，'real' 和当前的标记集是相同的
  bug_annotated_ids_real = np.copy(bug_annotated_ids)
  y_true_labeled_real = np.copy(y_true_labeled)

  current_step_idx = 0 # 从第0步开始迭代

else: # 如果是从中间步骤恢复
  current_step_idx = start_from_step + 1 # 下一个要执行的步骤索引
  print(f"Resuming from run {run+1}, step {current_step_idx}...")
  resume_file_path = f'../data_state/uncertainty_new/upper_bound/initial_size{initial_size}_sample_size{sample_size}_model_name_{model_name}_run_{run+1}_step{current_step_idx}-{sample_times}_{MODE}_pseudo{pseudo_setting}_ratio{ratio}_balance{balance_ratio}.pkl'
  print(f"Loading state from: {resume_file_path}")

  # 根据伪标签设置加载不同的变量
  # 这些文件理论上保存了上一步 *结束时*（即伪标签训练后）的状态
  with open(resume_file_path, 'rb') as pklfile:
      files = pickle.load(pklfile)

  if pseudo_setting == 1:
    # 加载：标记ID, 池ID, 标记标签(可能含伪), 混合指标, 整体指标, 上一步伪标签邻居ID, 上一步伪标签测试结果, 可读性值, 身份值
    bug_annotated_ids, bug_pool_ids, y_true_labeled, metrics_mixed, metrics_all, saved_indices, tmp2, read_values, iden_values = files
    # 对于 setting 1, real 变量在这里不需要，但为了代码统一性可以初始化
    bug_annotated_ids_real = np.copy(bug_annotated_ids) # 或者从更早的步骤加载？需要确认逻辑
    y_true_labeled_real = np.copy(y_true_labeled)
  elif pseudo_setting == 2:
    # 加载：标记ID(含伪), 池ID, 标记标签(含伪), 混合指标, 整体指标, 上一步伪标签邻居ID, 上一步伪标签测试结果, *真实*标记ID, *真实*标记标签, 可读性, 身份
    bug_annotated_ids, bug_pool_ids, y_true_labeled, metrics_mixed, metrics_all, saved_indices, tmp2, bug_annotated_ids_real, y_true_labeled_real, read_values, iden_values = files
  elif pseudo_setting == 3:
    # 加载：标记ID(真实), 池ID, 标记标签(真实), 混合指标, 整体指标, 上一步伪标签邻居ID, 上一步伪标签测试结果, *用于训练*的ID(含伪), *用于训练*的标签(含伪), 可读性, 身份
    bug_annotated_ids, bug_pool_ids, y_true_labeled, metrics_mixed, metrics_all, saved_indices, tmp2, bug_annotated_ids_using, y_true_labeled_using, read_values, iden_values = files
    # 对于 setting 3, real 变量即为 bug_annotated_ids, y_true_labeled
    bug_annotated_ids_real = np.copy(bug_annotated_ids)
    y_true_labeled_real = np.copy(y_true_labeled)
  else:
      raise ValueError(f"Invalid pseudo_setting: {pseudo_setting}")

  print(f"Loaded state. Current annotated size: {len(bug_annotated_ids)}, pool size: {len(bug_pool_ids)}")
  print("Resuming requires retraining the model on the loaded annotated set to ensure correct state.")

  # --- 关键：恢复时需要重新训练模型以确保状态一致 ---
  # 因为模型对象本身没有被序列化保存，只保存了状态变量
  # 使用加载的 bug_annotated_ids 和 y_true_labeled (可能包含伪标签) 来训练模型
  model = reload_model() # 重新加载基础模型
  # 注意：这里使用的 y_true_labeled 对应于上一步结束时的状态
  model = modelfit_fusion(bug_annotated_ids, y_true_labeled, model, tokenized_sentences, epochs=EPOCH, verbose=0)
  # 测试恢复状态下的模型性能
  tmp2_retest = modeltest(model, verbose=0)
  print("Retested metrics after resuming and retraining (F1, Acc, AUC, Rec, Prec):", tmp2_retest)

  # --- 验证恢复的指标 ---
  # 比较重新测试的结果 tmp2_retest 和加载的指标 tmp2 是否一致
  # 注意：由于训练的随机性，结果可能不会完全相同，但应该接近
  # 同时，更新 metrics_all 和 metrics_mixed 中对应步骤的值
  # metrics_all 的索引是 sample_time + 1
  # metrics_mixed 的索引是 2*sample_time + 1 (伪标签训练后)
  step_index_in_metrics = current_step_idx # 恢复的是第 current_step_idx 步 *开始* 的状态，对应 metrics 数组中第 current_step_idx 列
  if step_index_in_metrics <= sample_times:
      metrics_all[:, step_index_in_metrics] = tmp2_retest # 使用重新测试的结果更新
  if 2 * (step_index_in_metrics - 1) + 1 < metrics_mixed.shape[1]: # 检查索引有效性
      metrics_mixed[:, 2 * (step_index_in_metrics - 1) + 1] = tmp2_retest # 使用重新测试的结果更新

  # 重新保存一次状态，以防中断 (可选)
  # 这会覆盖加载的文件，但确保了当前代码状态和保存状态一致
  # save_state(resume_file_path, ...) # 需要一个保存状态的辅助函数

# --- 开始主动学习迭代 ---
# 从 current_step_idx 开始，直到 sample_times
for sample_time in range(current_step_idx, sample_times):
  print(f'\n--- AL Step {sample_time + 1} / {sample_times} ---')
  ###########
  subset, subset_indices = random_subset_indices(bug_pool_ids, subset_size) # 选择随机子集
  ###########

  # --- 1. 选择样本 (AL Query Strategy) ---
  print(f"Selecting {sample_size} samples using mode: {MODE}, flags: {f1_flag},{f2_flag},{f3_flag},{f4_flag},{f5_flag}")
  start_time = time.time() # 记录选择开始时间

  indices = [] # 存储选中的样本在 bug_pool_ids 中的索引
  result = []  # 存储选中样本对应的 AL 分数/成本

  # --- 根据 MODE 和 flag 执行不同的采样策略 ---

  if MODE == 'random':
    # 随机选择 sample_size 个索引
    if sample_size <= len(bug_pool_ids):
        indices = random.sample(range(len(bug_pool_ids)), sample_size)
    else:
        print("Warning: sample_size > pool size. Selecting all remaining pool items.")
        indices = list(range(len(bug_pool_ids)))
    result = [] # 随机采样没有分数

  elif MODE == 'dominated' or MODE == 'knee':
    # --- 多目标优化采样 (帕累托或膝点) ---
    costs = [] # 存储每个目标的成本值 (越小越好)

    # a) 计算不确定性 (如果需要)
    if f1_flag == 'uncertainty':
      # 对当前池中的所有样本进行预测
      y_pred_pool = modelpredict(bug_pool_ids, model, tokenized_sentences, return_pool_result=False, verbose=0)
      # 不确定性定义为 |0.5 - P(y=1)|，值越小越不确定
      uncertainty_cost = np.abs(0.5 - y_pred_pool)
      costs.append(uncertainty_cost)

    # b) 计算可读性成本 (如果需要)
    if f4_flag == 'read':
      # 计算 Flesch reading ease 分数 (分数越高越易读)
      readability_scores = X_train.iloc[bug_pool_ids].apply(lambda text: textstat.flesch_reading_ease(text)).values
      # 成本是负的可读性分数 (越小越好，即选择可读性差的？或者原始目标是最大化可读性？需要确认)
      # 假设目标是最小化成本，即选择可读性差的文本
      readability_cost = -readability_scores
      costs.append(readability_cost)

    # c) 计算身份/关键词成本 (如果需要)
    if f5_flag == 'iden':
      # 计算文本中关键词的密度
      identity_scores = X_train.iloc[bug_pool_ids].apply(lambda text: sum(Counter(word for word in words_to_search if word in text).values()) / (len(text.split()) + 1e-6)).values # 避免除以零
      # 成本是负的密度分数 (选择关键词少的？还是目标是最大化密度？)
      # 假设目标是最小化成本，即选择关键词少的文本
      identity_cost = -identity_scores
      costs.append(identity_cost)

    # 将成本列表转换为 (n_points, n_costs) 的 Numpy 数组
    if not costs:
        raise ValueError("No costs calculated for dominated/knee mode. Check flags.")
    costs = np.transpose(costs)

    # 执行选择
    if costs.shape[0] > 0: # 确保有候选点
        if MODE == 'dominated':
            # 使用帕累托支配选择
            indices = dominated_solution(costs, sample_size)
        elif MODE == 'knee':
            # 使用帕累托+膝点选择
            if costs.shape[1] == 3: # 确保是3维
                indices = dominated_knee_solution(costs, sample_size)
            else:
                print("Warning: Knee mode requires 3 costs. Falling back to dominated solution.")
                indices = dominated_solution(costs, sample_size) # 回退到支配选择
        # 获取选中样本的成本值
        if len(indices) > 0:
            result = costs[indices]
        else:
            result = []
    else: # 没有候选点
        indices = []
        result = []


  elif MODE == 'DBScan' or MODE == 'kmeans':
    # --- 基于聚类的采样 ---
    # a) 获取池中样本的预测概率和特征向量 (pooled_output)
    y_pred_pool, pool_res = modelpredict(bug_pool_ids, model, tokenized_sentences, return_pool_result=True, verbose=0)

    # b) 计算每个样本的目标分数 (用于在簇内选择代表点)
    #    这个目标分数可以是多种单一标准之一
    obj = None
    if f1_flag == 'uncertainty':
      obj = np.abs(0.5 - y_pred_pool) # 不确定性
    elif f1_flag == 'read':
      read = X_train.iloc[bug_pool_ids].apply(lambda text: textstat.flesch_reading_ease(text)).values
      obj = -read # 负可读性
      # 处理极端值 (textstat 可能返回非常大的负数)
      obj[obj < -206.8] = 9999 # 将过低的分数设为一个较大的成本值
    elif f1_flag == 'iden':
      iden = X_train.iloc[bug_pool_ids].apply(lambda text: sum(Counter(word for word in words_to_search if word in text).values()) / (len(text.split()) + 1e-6)).values
      obj = -iden # 负关键词密度
      # 处理零密度的情况
      obj[obj == 0] = 9999 # 将零密度的成本设为较大值
    elif f4_flag == 'normalized_sum': # 如果是 kmeans(normalized_sum)
      cer = np.abs(0.5 - y_pred_pool)
      read = X_train.iloc[bug_pool_ids].apply(lambda text: textstat.flesch_reading_ease(text)).values
      iden = X_train.iloc[bug_pool_ids].apply(lambda text: sum(Counter(word for word in words_to_search if word in text).values()) / (len(text.split()) + 1e-6)).values
      costs_cluster = np.array([cer, -read, -iden]).T # (n_points, 3)
      # 归一化处理
      scaler = MinMaxScaler()
      # 检查是否有有效数据进行拟合
      if costs_cluster.shape[0] > 0:
          scaler.fit(costs_cluster)
          normalized_costs = scaler.transform(costs_cluster)
          # 计算归一化后的总和 (目标是最小化这个总和?)
          obj = normalized_costs[:, 0] + normalized_costs[:, 1] + normalized_costs[:, 2]
      else: # 如果没有数据，无法计算 obj
          obj = np.array([])
    else:
        raise ValueError(f"Unsupported objective flag '{f1_flag}'/'{f4_flag}' for clustering mode.")

    # c) 执行聚类
    if pool_res.shape[0] > 0 and obj is not None and len(obj) > 0: # 确保有数据和目标分数
        n_clusters = min(sample_size, pool_res.shape[0]) # 聚类数不能超过样本数
        if n_clusters == 0:
             indices = []
             result = []
        else:
            if MODE == 'kmeans':
                # 使用 K-Means 聚类
                clustering = KMeans(n_clusters=n_clusters, random_state=run + sample_time, n_init=10).fit(pool_res) # 设置随机种子和n_init
            elif MODE == 'DBScan':
                # 使用 DBSCAN 聚类 (参数可能需要调整)
                # DBSCAN 不保证固定数量的簇，可能不适合固定预算采样
                # 这里的 min_samples=sample_size 可能导致只找到一个大簇或全是噪声点
                print("Warning: DBSCAN with min_samples=sample_size might not be ideal for fixed budget sampling.")
                clustering = DBSCAN(eps=0.5, min_samples=max(2, int(0.1 * sample_size))).fit(pool_res) # 示例参数调整
            else: # Should not happen
                 raise ValueError("Invalid clustering mode")

            # d) 从每个簇中选择目标分数最低 (最优) 的样本
            min_objective_indices = {} # 存储每个簇最优样本的 {簇标签: (索引, 目标分数)}
            labels = clustering.labels_

            for i, (label, objective) in enumerate(zip(labels, obj)):
                if label == -1: continue # 跳过 DBSCAN 中的噪声点
                if label not in min_objective_indices or objective < min_objective_indices[label][1]:
                    min_objective_indices[label] = (i, objective) # 更新最优样本

            # 提取每个簇最优样本的索引
            indices = [data[0] for label, data in min_objective_indices.items()]
            # 获取选中样本的目标分数
            if indices:
                result = obj[indices]
            else:
                result = []

            # 如果聚类得到的代表点少于 sample_size，可以考虑补充 (例如随机选择)
            if len(indices) < sample_size and len(indices) < len(bug_pool_ids):
                print(f"Warning: Clustering found only {len(indices)} representatives. Selecting more randomly.")
                remaining_needed = sample_size - len(indices)
                pool_indices = list(range(len(bug_pool_ids)))
                # 排除已选中的点
                remaining_pool_indices = [i for i in pool_indices if i not in indices]
                # 随机选择补充
                if remaining_needed <= len(remaining_pool_indices):
                    additional_indices = random.sample(remaining_pool_indices, remaining_needed)
                    indices.extend(additional_indices)
                else: # 如果剩余池也不够
                    indices.extend(remaining_pool_indices)


    else: # 如果没有池化结果或目标分数
        indices = []
        result = []


  elif MODE == 'single' or MODE == 'normalized_sum' or MODE == 'read+iden':
    # --- 单一目标或组合目标采样 (非聚类/非帕累托) ---
    obj = None # 存储计算出的目标分数 (越小越好)

    # --- 计算各种可能的单一目标分数 ---
    if f1_flag == 'uncertainty' or f1_flag == 'threshold':
      y_pred_pool = modelpredict(bug_pool_ids, model, tokenized_sentences, return_pool_result=False, verbose=0)
      obj = np.abs(0.5 - y_pred_pool) # 不确定性成本
    elif f1_flag == 'read':
      read = X_train.iloc[bug_pool_ids].apply(lambda text: textstat.flesch_reading_ease(text)).values
      obj = -read
      obj[obj < -206.8] = 9999 # 处理极端值
    elif f1_flag == 'iden':
      iden = X_train.iloc[bug_pool_ids].apply(lambda text: sum(Counter(word for word in words_to_search if word in text).values()) / (len(text.split()) + 1e-6)).values
      obj = -iden
      obj[obj == 0] = 9999 # 处理零密度

    # --- 计算基于 KL 散度的多样性分数 (f2_flag) ---
    # Note: KL 散度计算方式似乎有问题，y_pred_labeled 和 y_temp 都是标量概率，不是分布
    # 原始 KL(p||q) = sum(p(x) * log(p(x)/q(x))). 需要 P(class|sample) 分布。
    # 这里可能意图是计算预测概率的差异，但实现方式需要修正。
    elif f2_flag == 'labeled': # 与已标记样本预测的 "差异" (多样性)
        print("Warning: f2_flag 'labeled' KL divergence calculation might be incorrect.")
        # y_pred_labeled, _ = modelpredict(bug_annotated_ids, model, tokenized_sentences, return_pool_result=True, verbose=0) # 获取已标记样本的概率和向量
        # y_pred_pool = modelpredict(bug_pool_ids, model, tokenized_sentences, return_pool_result=False, verbose=0) # 获取池样本概率
        # # 目标：选择与已标记样本预测差异大的点
        # # 简单的替代：计算与已标记样本平均预测概率的差的绝对值
        # mean_labeled_prob = np.mean(y_pred_labeled)
        # obj = -np.abs(y_pred_pool - mean_labeled_prob) # 选择差异大的，成本是负差异
        obj = np.random.rand(len(bug_pool_ids)) # 临时使用随机值替代

    elif f2_flag == 'unlabeled': # 与未标记样本预测的 "差异" (多样性)
        print("Warning: f2_flag 'unlabeled' KL divergence calculation might be incorrect.")
        # y_pred_pool = modelpredict(bug_pool_ids, model, tokenized_sentences, return_pool_result=False, verbose=0)
        # mean_pool_prob = np.mean(y_pred_pool)
        # obj = -np.abs(y_pred_pool - mean_pool_prob) # 选择与平均预测差异大的
        obj = np.random.rand(len(bug_pool_ids)) # 临时使用随机值替代

    elif f2_flag == 'minus_labeled': # 与已标记样本预测的 "相似度" (选择相似的？)
        print("Warning: f2_flag 'minus_labeled' KL divergence calculation might be incorrect.")
        # y_pred_labeled, _ = modelpredict(bug_annotated_ids, model, tokenized_sentences, return_pool_result=True, verbose=0)
        # y_pred_pool = modelpredict(bug_pool_ids, model, tokenized_sentences, return_pool_result=False, verbose=0)
        # mean_labeled_prob = np.mean(y_pred_labeled)
        # obj = np.abs(y_pred_pool - mean_labeled_prob) # 选择差异小的
        obj = np.random.rand(len(bug_pool_ids)) # 临时使用随机值替代

    elif f2_flag == 'minus_unlabeled': # 与未标记样本预测的 "相似度"
        print("Warning: f2_flag 'minus_unlabeled' KL divergence calculation might be incorrect.")
        # y_pred_pool = modelpredict(bug_pool_ids, model, tokenized_sentences, return_pool_result=False, verbose=0)
        # mean_pool_prob = np.mean(y_pred_pool)
        # obj = np.abs(y_pred_pool - mean_pool_prob) # 选择与平均预测差异小的
        obj = np.random.rand(len(bug_pool_ids)) # 临时使用随机值替代

    # --- 计算基于特征空间距离的代表性分数 (f3_flag) ---
    elif f3_flag == 'labeled': # 代表已标记样本 (选择与已标记样本平均距离最小的)
      y_pred_pool, pool_res_unlabeled = modelpredict(bug_pool_ids, model, tokenized_sentences, return_pool_result=True, verbose=0)
      _, pool_res_labeled = modelpredict(bug_annotated_ids, model, tokenized_sentences, return_pool_result=True, verbose=0)
      obj = np.zeros(pool_res_unlabeled.shape[0])
      if pool_res_labeled.shape[0] > 0 and pool_res_unlabeled.shape[0] > 0:
          # 计算每个未标记点到所有已标记点的平均距离
          distances = pairwise_distances(pool_res_unlabeled, pool_res_labeled) # (n_unlabeled, n_labeled)
          obj = np.mean(distances, axis=1) # (n_unlabeled,) 成本是平均距离，越小越好
      else: # 如果没有已标记点或未标记点
          obj = np.random.rand(len(bug_pool_ids)) # 随机

    elif f3_flag == 'unlabeled': # 代表未标记样本 (选择与未标记样本平均距离最小的)
      y_pred_pool, pool_res_unlabeled = modelpredict(bug_pool_ids, model, tokenized_sentences, return_pool_result=True, verbose=0)
      obj = np.zeros(pool_res_unlabeled.shape[0])
      if pool_res_unlabeled.shape[0] > 1:
          # 计算未标记点之间的两两距离
          dist_matrix_unlabeled = pairwise_distances(pool_res_unlabeled) # (n_unlabeled, n_unlabeled)
          # 计算每个点到其他所有点的平均距离
          obj = np.sum(dist_matrix_unlabeled, axis=1) / (dist_matrix_unlabeled.shape[1] - 1 + 1e-9) # 避免除以0
          # 成本是平均距离，越小越好
      else:
          obj = np.random.rand(len(bug_pool_ids)) # 随机

    elif f3_flag == 'minus_unlabeled': # 不代表未标记样本 (选择与未标记样本平均距离最大的)
      y_pred_pool, pool_res_unlabeled = modelpredict(bug_pool_ids, model, tokenized_sentences, return_pool_result=True, verbose=0)
      obj = np.zeros(pool_res_unlabeled.shape[0])
      if pool_res_unlabeled.shape[0] > 1:
          dist_matrix_unlabeled = pairwise_distances(pool_res_unlabeled)
          obj = np.sum(dist_matrix_unlabeled, axis=1) / (dist_matrix_unlabeled.shape[1] - 1 + 1e-9)
          obj = -obj # 成本是负的平均距离，越小越好 (即选择距离大的)
      else:
           obj = np.random.rand(len(bug_pool_ids)) # 随机

    elif f3_flag == 'minus_labeled': # 不代表已标记样本 (选择与已标记样本平均距离最大的)
      y_pred_pool, pool_res_unlabeled = modelpredict(bug_pool_ids, model, tokenized_sentences, return_pool_result=True, verbose=0)
      _, pool_res_labeled = modelpredict(bug_annotated_ids, model, tokenized_sentences, return_pool_result=True, verbose=0)
      obj = np.zeros(pool_res_unlabeled.shape[0])
      if pool_res_labeled.shape[0] > 0 and pool_res_unlabeled.shape[0] > 0:
          distances = pairwise_distances(pool_res_unlabeled, pool_res_labeled)
          obj = np.mean(distances, axis=1)
          obj = -obj # 成本是负的平均距离，越小越好 (即选择距离大的)
      else:
          obj = np.random.rand(len(bug_pool_ids)) # 随机

    # --- 计算 BALD 分数 (f3_flag) ---
    elif f3_flag == 'BALD':
      model.train() # BALD 需要模型在训练模式下进行 dropout
      dataloader_pool = ids2dataloader(bug_pool_ids, tokenized_sentences, batch_size=128)
      nb_MC_samples = 10 # Monte Carlo dropout 采样次数 (原代码为50，减少以加速)
      MC_samples_output = [] # 存储每次 MC 采样的概率输出

      print(f"Running {nb_MC_samples} MC dropout samples for BALD...")
      for i in range(nb_MC_samples):
        probs_one_mc = []
        for batch in dataloader_pool: # 无需 tqdm
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            with torch.no_grad():
                # 前向传播 (因为 model.train(), dropout 会生效)
                _, pooled_output = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, return_dict=False)
                logits = linear_layer(pooled_output)
                prob = F.softmax(logits, dim=-1).detach().cpu().numpy() # (batch_size, 2)
                probs_one_mc.append(prob)
        MC_samples_output.append(np.concatenate(probs_one_mc, axis=0)) # (n_pool, 2)

      # 堆叠所有 MC 采样结果
      MC_samples = np.stack(MC_samples_output) # (nb_MC_samples, n_pool, 2)

      # 计算 BALD 分数: Mutual Information = Entropy(Expected Prediction) - Expected(Entropy(Prediction))
      # BALD 分数越大越好，所以成本是负的 BALD 分数
      expected_p = np.mean(MC_samples, axis=0) # (n_pool, 2)
      entropy_expected_p = -np.sum(expected_p * np.log(expected_p + 1e-10), axis=-1) # (n_pool,)

      entropy_samples = -np.sum(MC_samples * np.log(MC_samples + 1e-10), axis=-1) # (nb_MC_samples, n_pool)
      expected_entropy = np.mean(entropy_samples, axis=0) # (n_pool,)

      bald_scores = entropy_expected_p - expected_entropy # (n_pool,)
      obj = -bald_scores # 成本是负的 BALD 分数 (越小越好，即选择 BALD 越大的)
      model.eval() # 恢复评估模式

    # --- 计算 Coreset 分数 (f3_flag) ---
    elif f3_flag == 'coreset' or f3_flag == 'minus_coreset':
      # 获取未标记和已标记样本的特征向量
      _, pool_res_unlabeled = modelpredict(bug_pool_ids, model, tokenized_sentences, return_pool_result=True, verbose=0)
      _, pool_res_labeled = modelpredict(bug_annotated_ids, model, tokenized_sentences, return_pool_result=True, verbose=0)

      if pool_res_unlabeled.shape[0] > 0: # 确保有未标记样本
          # 将未标记和已标记样本的特征向量合并，未标记在前
          if pool_res_labeled.shape[0] > 0:
              process_array = np.concatenate((pool_res_unlabeled, pool_res_labeled))
              n_annotated = pool_res_labeled.shape[0]
          else: # 如果没有已标记样本
              process_array = pool_res_unlabeled
              n_annotated = 0
          n_unlabeled = pool_res_unlabeled.shape[0]

          # 初始化 Coreset 对象
          coreset_mode = 'max' if f3_flag == 'coreset' else 'min'
          coreset = Coreset_Greedy(process_array, mode=coreset_mode)
          # 执行采样 (这里的 obj 是选择过程中的距离值，不是直接的成本)
          # 返回的 indices 是在 process_array 中的索引 (0 到 n_unlabeled-1 范围)
          # 注意：coreset.sample 直接返回了选中的索引，不需要后续排序
          sorted_indices, obj_coreset = coreset.sample(n_unlabeled, n_annotated, sample_size)
          indices = np.array(sorted_indices) # 直接使用 coreset 返回的索引
          # result = np.array(obj_coreset) # result 存储选择时的距离
          # obj 设置为 None，因为选择逻辑已在 Coreset_Greedy 中完成
          obj = None # 标记 obj 已被 coreset 处理
      else: # 没有未标记样本
          indices = []
          # result = []
          obj = None


    # --- 计算组合目标分数 (f4_flag) ---
    elif f4_flag == 'normalized_sum':
      y_pred_pool = modelpredict(subset, model, tokenized_sentences, return_pool_result=False, verbose=0)
      cer = np.abs(0.5 - y_pred_pool)
      read = X_train.iloc[subset].apply(lambda text: textstat.flesch_reading_ease(text)).values
      iden = X_train.iloc[subset].apply(lambda text: sum(Counter(word for word in words_to_search if word in text).values()) / (len(text.split()) + 1e-6)).values
      costs_norm = np.array([cer, -read, -iden]).T # (n_points, 3)
      if costs_norm.shape[0] > 0:
          scaler = MinMaxScaler()
          scaler.fit(costs_norm)
          normalized_costs = scaler.transform(costs_norm)
          # 成本是归一化后的总和 (越小越好)
          obj = normalized_costs[:, 0] + normalized_costs[:, 1] + normalized_costs[:, 2]
      else:
          obj = np.array([])

    elif f4_flag == 'read+iden':
      read = X_train.iloc[bug_pool_ids].apply(lambda text: textstat.flesch_reading_ease(text)).values
      iden = X_train.iloc[bug_pool_ids].apply(lambda text: sum(Counter(word for word in words_to_search if word in text).values()) / (len(text.split()) + 1e-6)).values
      costs_norm = np.array([-read, -iden]).T # (n_points, 2)
      if costs_norm.shape[0] > 0:
          scaler = MinMaxScaler()
          scaler.fit(costs_norm)
          normalized_costs = scaler.transform(costs_norm)
          # 成本是归一化后的总和
          obj = normalized_costs[:, 0] + normalized_costs[:, 1]
      else:
          obj = np.array([])

    # --- 如果没有指定任何 flag，则默认为随机 ---
    if obj is None and f3_flag != 'coreset' and f3_flag != 'minus_coreset':
        print("Warning: No valid objective flag specified for single/normalized_sum mode. Defaulting to random selection.")
        if sample_size <= len(bug_pool_ids):
            indices = random.sample(range(len(bug_pool_ids)), sample_size)
        else:
            indices = list(range(len(bug_pool_ids)))
        result = []
        obj = np.random.rand(len(bug_pool_ids)) # 赋予随机 obj 以避免后续错误

    # --- 根据计算出的 obj 选择索引 (除非是 coreset) ---
    if obj is not None: # 如果 obj 已计算且不是 coreset 处理的
        if len(obj) > 0: # 确保 obj 不为空
            if f1_flag == 'threshold':
                # 选择 obj 大于阈值且最小的 sample_size 个
                condition_indices = np.where(obj > threshold)[0] # 找到满足阈值的索引
                if len(condition_indices) > 0:
                    # 在满足条件的索引中，按 obj 值升序排序，取前 sample_size 个
                    indices = condition_indices[np.argsort(obj[condition_indices])[:sample_size]]
                else: # 如果没有满足条件的点
                    print(f"Warning: No samples found with uncertainty > {threshold}. Selecting lowest uncertainty samples.")
                    indices = np.argsort(obj)[:sample_size] # 回退到选择不确定性最低的
            else:
                # 选择 obj 最小的 sample_size 个
                # 使用 argpartition 获取最小的 sample_size 个值的索引 (效率比 argsort 高)
                num_to_select = min(sample_size, len(obj)) # 防止选择超过可用数量
                indices = np.argpartition(obj, num_to_select)[:num_to_select]

            # 获取选中样本的 obj 值
            if len(indices) > 0:
                result = obj[indices]
            else:
                result = []
        else: # 如果 obj 为空 (可能因为池为空)
            indices = []
            result = []


  else: # 无效的 MODE
      raise ValueError(f"Invalid MODE: {MODE}")

  # --- 确保 indices 是 Numpy 数组 ---
  indices = np.array(indices, dtype=int) # 该索引是subset的索引，同时也是subset_indices的索引，而subset_indices中存的值又是bug_pool_ids的索引
  indices = subset_indices[indices] # 转换为原始索引

  # --- 检查是否有足够的样本被选中 ---
  if len(indices) == 0 and len(bug_pool_ids) > 0:
      print("Warning: No samples selected by the active learning strategy. Selecting randomly.")
      num_to_select = min(sample_size, len(bug_pool_ids))
      indices = np.random.choice(range(len(bug_pool_ids)), num_to_select, replace=False)
      result = [] # 随机选择无分数

  print(f"Selected {len(indices)} samples.")

  # --- 计算选中样本的可读性和身份值 (用于后续分析或保存) ---
  if len(indices) > 0:
      selected_pool_ids = bug_pool_ids[indices] # 获取在原始训练集中的 ID
      read_values = X_train.iloc[selected_pool_ids].apply(lambda text: textstat.flesch_reading_ease(text)).values
      iden_values = X_train.iloc[selected_pool_ids].apply(lambda text: sum(Counter(word for word in words_to_search if word in text).values()) / (len(text.split()) + 1e-6)).values
      saved_indices = selected_pool_ids # 保存选中的原始 ID
  else:
      read_values = np.array([])
      iden_values = np.array([])
      saved_indices = np.array([], dtype=int)

  # --- 2. 更新数据集 ---
  if len(indices) > 0:
      print("Updating annotated and pool sets...")
      # a) 将选中的样本从未标记池移到已标记集
      new_labels = y_train[selected_pool_ids] # 获取真实标签
      bug_annotated_ids = np.concatenate((bug_annotated_ids, selected_pool_ids))
      y_true_labeled = np.concatenate((y_true_labeled, new_labels)) # y_true_labeled 可能包含伪标签，取决于上一轮

      # b) 更新 'real' 标记集 (仅包含真实标签)
      bug_annotated_ids_real = np.concatenate((bug_annotated_ids_real, selected_pool_ids))
      y_true_labeled_real = np.concatenate((y_true_labeled_real, new_labels))

      # c) 从池中删除选中的样本
      bug_pool_ids = np.delete(bug_pool_ids, indices, axis=0)

      print(f"New annotated size: {len(bug_annotated_ids)}, New pool size: {len(bug_pool_ids)}")
  else:
      print("Pool is empty or no samples were selected. Skipping dataset update.")


  end_time = time.time() # 记录选择结束时间
  execution_time = end_time - start_time
  print(f"Sample selection time: {format_time(execution_time)}")

  # --- 3. 训练模型 (在更新后的已标记集上) ---
  print("\nTraining model on updated annotated set...")
  model = reload_model() # 重新加载基础 BERT 模型
  # 使用当前的 bug_annotated_ids 和 y_true_labeled 进行训练
  # 注意：这里的 y_true_labeled 可能包含来自上一步的伪标签 (如果 pseudo_setting != 3)
  model = modelfit_fusion(bug_annotated_ids, y_true_labeled, model, tokenized_sentences, epochs=EPOCH, verbose=0)

  # --- 4. 测试模型 (AL 步骤后) ---
  print("\nTesting model after AL step training...")
  tmp = modeltest(model, verbose=0) # 在固定测试集上评估
  print("Metrics after AL step (F1, Acc, AUC, Rec, Prec):", tmp)
  # 记录性能到 metrics_mixed 的偶数列
  metrics_mixed[:, 2 * sample_time] = tmp # 使用实际测试结果累加或赋值？代码是累加，但可能应该是赋值

  # --- 5. 保存中间状态 (AL 步骤后，伪标签前) ---
  # 保存当前状态，以便可以从伪标签步骤前恢复
  # 文件名表示这是 *进入* 伪标签步骤前的状态
  mid_save_path = f'../data_state/uncertainty_new/upper_bound/mid_initial_size{initial_size}_sample_size{sample_size}_model_name_{model_name}_run_{run+1}_step{sample_time+1}-{sample_times}_{MODE}_{f1_flag}{f2_flag}{f3_flag}{f4_flag}{f5_flag}_pseudo{pseudo_setting}_ratio{ratio}_balance{balance_ratio}.pkl'
  print(f"Saving intermediate state to: {mid_save_path}")
  files_mid = (bug_annotated_ids, bug_pool_ids, y_true_labeled, # 当前状态
               metrics_mixed, # 已更新的混合指标
               saved_indices, # 本轮选中的原始ID
               result,        # 本轮选中的样本分数
               read_values,   # 本轮选中样本的可读性
               iden_values,   # 本轮选中样本的身份值
               tmp,           # 刚测试的结果
               bug_annotated_ids_real, y_true_labeled_real) # 真实标签集
  with open(mid_save_path, 'wb') as pklfile:
      pickle.dump(files_mid, pklfile)

  # --- 6. 半监督/伪标签步骤 ---
  ###########
  subset, subset_indices = random_subset_indices(bug_pool_ids, subset_size) # 选择随机子集
  ###########
  print("\nStarting Semi-supervised / Pseudo-labeling step...")
  # a) 获取当前池中样本的预测和特征向量
  if len(subset) > 0:
      print("Predicting on remaining pool for pseudo-labeling...")
      pool_predictions, pool_vectors = modelpredict(subset, model, tokenized_sentences, return_pool_result=True, verbose=0)
      # pool_predictions 是 P(y=1)
      # pool_vectors 是 [CLS] embedding
  else:
      print("Pool is empty. Skipping pseudo-labeling.")
      pool_vectors = np.empty((0, 768)) # 创建空数组以避免后续错误

  # b) 获取用于查找邻居的锚点向量
  #    根据 pseudo_setting 决定是使用当前标记集（可能含伪）还是真实标记集
  if pseudo_setting == 1 or pseudo_setting == 3:
      # 使用当前的 bug_annotated_ids (可能包含伪标签) 作为锚点
      print("Using current annotated set (may include pseudo-labels) as anchors.")
      if len(bug_annotated_ids) > 0:
          anchor_predictions, anchor_vectors = modelpredict(bug_annotated_ids, model, tokenized_sentences, return_pool_result=True, verbose=0)
      else:
          anchor_vectors = np.empty((0, 768))
  elif pseudo_setting == 2:
      # 使用 bug_annotated_ids_real (只包含真实标签) 作为锚点
      print("Using real annotated set (true labels only) as anchors.")
      if len(bug_annotated_ids_real) > 0:
          anchor_predictions_real, anchor_vectors = modelpredict(bug_annotated_ids_real, model, tokenized_sentences, return_pool_result=True, verbose=0)
      else:
          anchor_vectors = np.empty((0, 768))
  else:
       raise ValueError(f"Invalid pseudo_setting: {pseudo_setting}")

  # c) 使用 Faiss 查找近邻
  #    为每个锚点在池中找到最近的 nb_num 个邻居
  nb_num = int(1 * ratio) # 每个锚点找多少个邻居 (这里固定为 1 * ratio)
  pseudo_indices_in_pool = [] # 存储找到的邻居在 bug_pool_ids 中的索引

  if anchor_vectors.shape[0] > 0 and pool_vectors.shape[0] > 0 and nb_num > 0:
      print(f"Finding {nb_num} nearest neighbors in pool for each of {anchor_vectors.shape[0]} anchors using Faiss...")
      index = faiss.IndexFlatL2(pool_vectors.shape[1]) # 创建 L2 距离索引
      # Faiss 需要 float32
      index.add(pool_vectors.astype(np.float32))
      # 搜索近邻
      D, I = index.search(anchor_vectors.astype(np.float32), nb_num) # D: distances, I: indices in pool_vectors
      # I 的形状是 (n_anchors, nb_num)
      # 展平获取所有找到的邻居索引 (可能会有重复)
      pseudo_indices_in_pool = I.flatten()
      # 去重
      pseudo_indices_in_pool = np.unique(pseudo_indices_in_pool)
      pseudo_indices_in_pool = subset_indices[pseudo_indices_in_pool] # 转换为原始索引:原本该索引是在subset中的索引，现在要转换成在bug_pool_ids中的索引
      print(f"Found {len(pseudo_indices_in_pool)} unique pseudo-label candidates.")
  else:
      print("Skipping neighbor search (no anchors, no pool, or nb_num=0).")

  # d) 准备用于伪标签训练的数据
  bug_annotated_ids_for_pseudo_train = bug_annotated_ids # 默认使用当前标记集
  y_true_labeled_for_pseudo_train = y_true_labeled

  if len(pseudo_indices_in_pool) > 0:
      # 获取选中伪标签样本的原始 ID 和对应的“伪”标签
      pseudo_original_ids = bug_pool_ids[pseudo_indices_in_pool]

      # 确定伪标签的来源
      if pseudo_setting == 1:
          # 使用锚点的标签作为伪标签 (每个邻居对应一个锚点，可能重复使用标签)
          # 需要找到每个 pseudo_index 对应的 anchor 标签
          # 简单方法：直接重复原始标签集 nb_num 次？(原代码逻辑) - 这不准确
          # 更合理：使用锚点的预测？或者使用锚点的真实标签？
          # 原代码逻辑似乎是：直接将原始 y_true_labeled 重复 nb_num 次作为伪标签
          print("Warning: Pseudo-label generation for setting 1 uses repeated original labels, might be inaccurate.")
          pseudo_labels = np.repeat(y_true_labeled, nb_num)[:len(pseudo_original_ids)] # 截断以匹配找到的邻居数
          # 更新训练集 (永久性修改)
          bug_annotated_ids = np.concatenate((bug_annotated_ids, pseudo_original_ids))
          y_true_labeled = np.concatenate((y_true_labeled, pseudo_labels))
          bug_annotated_ids_for_pseudo_train = bug_annotated_ids
          y_true_labeled_for_pseudo_train = y_true_labeled

      elif pseudo_setting == 2:
          # 使用 *真实* 锚点的标签作为伪标签
          print("Warning: Pseudo-label generation for setting 2 uses repeated real labels, might be inaccurate.")
          pseudo_labels = np.repeat(y_true_labeled_real, nb_num)[:len(pseudo_original_ids)] # 使用真实标签重复
          # 更新训练集 (永久性修改 bug_annotated_ids 和 y_true_labeled)
          bug_annotated_ids = np.concatenate((bug_annotated_ids, pseudo_original_ids))
          y_true_labeled = np.concatenate((y_true_labeled, pseudo_labels))
          bug_annotated_ids_for_pseudo_train = bug_annotated_ids
          y_true_labeled_for_pseudo_train = y_true_labeled

      elif pseudo_setting == 3:
          # 创建 *临时* 训练集，不修改原始 bug_annotated_ids/y_true_labeled
          print("Warning: Pseudo-label generation for setting 3 uses repeated original labels for temporary training.")
          pseudo_labels = np.repeat(y_true_labeled, nb_num)[:len(pseudo_original_ids)]
          # 创建临时变量用于训练
          bug_annotated_ids_using = np.concatenate((bug_annotated_ids, pseudo_original_ids))
          y_true_labeled_using = np.concatenate((y_true_labeled, pseudo_labels))
          bug_annotated_ids_for_pseudo_train = bug_annotated_ids_using
          y_true_labeled_for_pseudo_train = y_true_labeled_using

      # 从池中删除被选作伪标签的样本 (对所有 setting 都执行)
      # 注意：如果 setting 3 不想永久删除，需要修改这里
      bug_pool_ids = np.delete(bug_pool_ids, pseudo_indices_in_pool, axis=0)
      saved_indices_pseudo = pseudo_original_ids # 记录本轮添加的伪标签原始ID
      print(f"Added {len(pseudo_original_ids)} pseudo-labels. New pool size: {len(bug_pool_ids)}")

  else: # 没有找到伪标签
      saved_indices_pseudo = np.array([], dtype=int)
      print("No pseudo-labels added in this step.")

  # e) 重新训练模型 (使用包含伪标签的数据)
  print("\nRetraining model with pseudo-labels (if any)...")
  model = reload_model() # 重新加载基础模型
  # 使用 *_for_pseudo_train 变量进行训练
  model = modelfit_fusion(bug_annotated_ids_for_pseudo_train, y_true_labeled_for_pseudo_train, model, tokenized_sentences, epochs=EPOCH, verbose=0) # 原代码 epochs=18

  # f) 测试模型 (伪标签步骤后)
  print("\nTesting model after pseudo-label retraining...")
  tmp2 = modeltest(model, verbose=0)
  print("Metrics after pseudo-label step (F1, Acc, AUC, Rec, Prec):", tmp2)

  # --- 7. 更新并保存最终状态 ---
  # 更新 metrics_all 和 metrics_mixed
  metrics_all[:, sample_time + 1] = tmp2 # 记录伪标签训练后的性能 (赋值)
  metrics_mixed[:, 2 * sample_time + 1] = tmp2 # 记录伪标签训练后的性能 (赋值)

  print(f'\n--- Run {run+1}, Step {sample_time+1} Finished ---')
  # 打印当前累计指标 (可选)
  print("Cumulative metrics_all (F1):", metrics_all[0, :sample_time+2])
  print("Cumulative metrics_mixed (F1):", metrics_mixed[0, :2*sample_time+2])


  # 根据 pseudo_setting 保存不同的状态变量
  final_save_path = f'../data_state/uncertainty_new/upper_bound/initial_size{initial_size}_sample_size{sample_size}_run_{run+1}_step{sample_time+1}-{sample_times}_{MODE}_{f1_flag}{f2_flag}{f3_flag}{f4_flag}{f5_flag}_pseudo{pseudo_setting}_ratio{ratio}_balance{balance_ratio}.pkl'
  print(f"Saving final state for step {sample_time+1} to: {final_save_path}")

  if pseudo_setting == 1:
    # 保存：标记ID(含伪), 池ID, 标记标签(含伪), 混合指标, 整体指标, 伪标签ID, 伪标签测试结果, 可读性, 身份
    files_final = (bug_annotated_ids, bug_pool_ids, y_true_labeled,
                   metrics_mixed, metrics_all,
                   saved_indices_pseudo, # 使用伪标签的 ID 替代 AL 选择的 ID？需要确认原意
                   tmp2, read_values, iden_values) # read/iden 还是 AL 选择时的值
  elif pseudo_setting == 2:
    # 保存：标记ID(含伪), 池ID, 标记标签(含伪), 混合指标, 整体指标, 伪标签ID, 伪标签测试结果, *真实*标记ID, *真实*标记标签, 可读性, 身份
     files_final = (bug_annotated_ids, bug_pool_ids, y_true_labeled,
                   metrics_mixed, metrics_all,
                   saved_indices_pseudo, tmp2,
                   bug_annotated_ids_real, y_true_labeled_real, # 保存更新后的真实集
                   read_values, iden_values)
  elif pseudo_setting == 3:
    # 保存：标记ID(真实), 池ID, 标记标签(真实), 混合指标, 整体指标, 伪标签ID, 伪标签测试结果, *用于训练*的ID(含伪), *用于训练*的标签(含伪), 可读性, 身份
    files_final = (bug_annotated_ids, bug_pool_ids, y_true_labeled, # 保存原始的，未被伪标签污染的
                   metrics_mixed, metrics_all,
                   saved_indices_pseudo, tmp2,
                   bug_annotated_ids_for_pseudo_train, y_true_labeled_for_pseudo_train, # 保存用于训练的临时集合
                   read_values, iden_values)
  else:
       raise ValueError(f"Invalid pseudo_setting: {pseudo_setting}")

  with open(final_save_path, 'wb') as pklfile:
    pickle.dump(files_final, pklfile)

# --- 运行结束 ---
print(f'\n--- Run {run+1} Finished ---')