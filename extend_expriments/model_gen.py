# -*- coding: utf-8 -*-
# 导入必要的库
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import argparse  # 用于解析命令行参数
import pdb       # Python Debugger，用于调试
from tool_funcs import parse_balance_ratio

# --- 命令行参数解析 ---
# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser()
# 添加命令行参数 '--model_name'，用于指定使用的模型名称
parser.add_argument('--model_name')
# 添加命令行参数 '--initial_size'，用于指定初始样本大小
parser.add_argument('--initial_size')

parser.add_argument('--balance_ratio', type=str, required=True, help='The balance ratio of the dataset')# 数据集中类别不平衡的比例
# 解析命令行传入的参数
args = parser.parse_args()

# --- 全局常量和配置 ---
EPOCH = 18  # 训练的总轮数
model_name = args.model_name  # 从命令行参数获取模型名称
sample_size = int(args.initial_size)  # 从命令行参数获取初始样本大小，并转换为整数
balance_ratio = parse_balance_ratio(args.balance_ratio) # 命令行输入元组解析

sample_times = 10  # 采样的次数或运行的次数 (似乎在主循环中被硬编码为 10 次，这里可能未使用或作为参考)
batch_size = 32    # 训练和预测时每个批次的大小 (在 `ids2dataloader` 中有默认值，这里可能未使用或作为参考)
MAX_LEN = 100     # 输入序列的最大长度，超过部分会被截断

# --- 导入其他必要的库 ---
import pandas as pd  # 用于数据处理，特别是 DataFrame
import re            # 正则表达式操作
import nltk          # 自然语言处理工具包 (代码中未使用，可能为遗留导入)
import numpy as np   # 用于数值计算
import string        # 包含常用字符串常量 (代码中未使用，可能为遗留导入)
import json          # 用于处理 JSON 数据 (代码中未使用，可能为遗留导入)
from sklearn.preprocessing import OneHotEncoder # 用于独热编码 (代码中未使用，可能为遗留导入)
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold # 用于数据集划分 (train_test_split 在 modelfit 中使用)
import torch         # PyTorch 深度学习框架
import random        # 用于生成随机数 (在 'else' 分支的初始采样中使用)
from sklearn.metrics import (accuracy_score, recall_score, precision_score, f1_score, auc, roc_curve, confusion_matrix) # 用于模型评估的指标
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler # PyTorch 数据处理工具
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer, get_linear_schedule_with_warmup, AutoTokenizer, RobertaTokenizer # Hugging Face Transformers 库，用于加载预训练模型和分词器
from torch.optim import AdamW # AdamW 优化器
import time          # 用于时间相关操作 (计算耗时)
import datetime      # 用于日期和时间格式化
from tqdm import tqdm # 用于显示进度条
import math          # 数学运算 (代码中未使用，可能为遗留导入)
import pickle        # 用于序列化和反序列化 Python 对象 (加载/保存数据和模型状态)
# import logging     # 用于日志记录 (被注释掉了)
import torch.nn.functional as F # PyTorch 神经网络函数库 (如 softmax)
from abc import abstractmethod # 用于定义抽象基类 (代码中未使用，可能为遗留导入)
# from sklearn.metrics import pairwise_distances # 用于计算距离 (代码中未使用，可能为遗留导入)
from transformers import BertModel, RobertaModel, AutoModel # Hugging Face 模型类
import os            # 用于操作系统交互 (如路径操作)
# import matplotlib.pyplot as plt # 用于绘图 (代码中未使用，可能为遗留导入)
from torch.utils.data import Dataset # PyTorch 数据集基类 (代码中未使用，但 DataLoader 依赖它)
# from datetime import datetime # datetime 已在前面导入
import torch.nn as nn # PyTorch 神经网络模块 (代码中未使用，但 linear_layer 使用了其子类)
import torch.optim as optim # PyTorch 优化器 (代码中未使用，使用了 AdamW)
from torch.optim.lr_scheduler import StepLR # 学习率调度器 (代码中未使用，使用了 get_linear_schedule_with_warmup)
# from sklearn.metrics import pairwise_distances # 重复导入
from sklearn.preprocessing import MinMaxScaler # 数据归一化 (代码中未使用，可能为遗留导入)
from sklearn.cluster import DBSCAN, KMeans # 聚类算法 (代码中未使用，可能为遗留导入)
import textstat      # 计算文本统计特征 (代码中未使用，可能为遗留导入)
from collections import Counter # 计数器 (代码中未使用，可能为遗留导入)
# from transformers import BertModel # 重复导入

# --- 数据加载 ---
test_size = 5000 # 定义测试集的大小 (用于文件名)

# 从 pickle 文件加载预处理好的数据
# 文件名包含测试集大小，表明数据是根据特定测试集划分的
with open(f'./initial_data/data_all_test_size{test_size}_balance{balance_ratio}.pkl', 'rb') as pklfile:
    # 反序列化文件内容，加载到 files 变量
    # 假设文件包含: 测试集特征, 测试集标签, 训练集 bug ID, 训练集特征, 训练集标签
    files = pickle.load(pklfile)
# 解包加载的数据
X_test, y_test, bug_train_ids, X_train, y_train = files

# --- Tokenizer 初始化 ---
# 根据命令行指定的 model_name 选择并加载相应的 Tokenizer
if model_name == 'rta':
    tokenizer = AutoTokenizer.from_pretrained("Colorful/RTA")
elif model_name == 'roberta':
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
elif model_name == 'codebert':
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
else:  # 默认为 bert-base-uncased
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# --- 测试数据准备 ---
# 使用加载的 tokenizer 对测试集文本进行编码
test_tokenized_sentences = tokenizer(list(X_test),  # 输入文本列表
                                     max_length=MAX_LEN,       # 最大序列长度
                                     truncation=True,         # 超过最大长度进行截断
                                     padding=True,            # 填充到最大长度
                                     return_tensors="pt",     # 返回 PyTorch 张量
                                     return_attention_mask=True) # 返回 attention mask

# 将编码后的 input_ids, attention_mask 和测试集标签 y_test 封装成 TensorDataset
# 注意：这里直接将 y_test 转换为 tensor，后续 DataLoader 会自动处理
test_data = TensorDataset(torch.tensor(test_tokenized_sentences['input_ids']),
                          torch.tensor(test_tokenized_sentences['attention_mask']),
                          torch.tensor(y_test))
# 创建测试数据的 SequentialSampler，按顺序采样
test_sampler = SequentialSampler(test_data)
# 创建测试数据的 DataLoader，用于按批次加载数据
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=128) # 使用了固定的 batch_size 128

# --- 辅助函数定义 ---

# 将指定的训练数据索引转换为 DataLoader
def ids2dataloader(ids, batch_size=32):
    """
    根据提供的索引列表 (ids)，从 X_train 和 y_train 中选取数据，
    进行 tokenize 处理，并创建一个 DataLoader。

    Args:
        ids (list or np.array): 需要加载的数据在 X_train/y_train 中的索引。
        batch_size (int, optional): DataLoader 的批次大小. Defaults to 32.

    Returns:
        DataLoader: 包含指定数据的 PyTorch DataLoader.
    """
    # 使用 tokenizer 对选定索引的训练文本进行编码
    tokenized_sentences = tokenizer(list(X_train.iloc[ids]),  # 使用 .iloc 从 Pandas Series 中选择文本
                                    max_length=MAX_LEN,
                                    truncation=True,
                                    padding=True,
                                    return_tensors="pt",
                                    return_attention_mask=True)

    # 创建 TensorDataset
    # 使用 .clone().detach() 可能是为了防止意外的梯度传播或修改原始张量
    data = TensorDataset(tokenized_sentences['input_ids'].clone().detach(),
                         tokenized_sentences['attention_mask'].clone().detach(),
                         torch.tensor(y_train[ids])) # 选择对应索引的标签并转为 Tensor
    # 使用 SequentialSampler 按顺序加载数据
    sampler = SequentialSampler(data)
    # 创建 DataLoader
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    return dataloader

# 格式化时间函数
def format_time(elapsed):
    '''
    将秒数转换为 hh:mm:ss 格式的字符串。
    '''
    # 将秒数四舍五入到最接近的整数
    elapsed_rounded = int(round((elapsed)))
    # 使用 timedelta 进行格式化
    return str(datetime.timedelta(seconds=elapsed_rounded))

# 计算准确率函数 (注意：此函数在当前脚本中未被直接调用)
def flat_accuracy(preds, labels):
    """
    计算预测结果 (preds) 和真实标签 (labels) 之间的准确率。
    假设 preds 是模型的原始输出 (logits 或概率)，需要先找到最大值的索引。
    """
    # 获取预测概率中最大值的索引，作为预测的类别
    pred_flat = np.argmax(preds, axis=1).flatten()
    # 将真实标签展平
    labels_flat = labels.flatten()
    # 计算预测正确的数量并除以总数
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# 单个样本预测函数 (注意：此函数在当前脚本中未被直接调用)
def single_id_predict(id, return_pooled_output=False):
  """
  对训练集中指定索引 (id) 的单个样本进行预测。

  Args:
      id (int): 要预测的样本在 X_train 中的索引。
      return_pooled_output (bool, optional): 是否同时返回模型的 [CLS] token 输出 (pooled output). Defaults to False.

  Returns:
      float or tuple:
          - 如果 return_pooled_output 为 False，返回类别 0 的预测概率。
          - 如果 return_pooled_output 为 True，返回包含 (类别 0 概率, pooled_output) 的元组。
  """
  # 对指定 id 的文本进行 tokenize
  tokens = tokenizer(X_train.iloc[id], max_length=MAX_LEN, truncation=True, padding=True, return_tensors="pt", return_attention_mask=True)
  # 在不计算梯度的模式下进行模型推理
  with torch.no_grad():
    # 获取模型输出，output[1] 通常是 pooled output ([CLS] token 的表示)
    output = model(tokens['input_ids'].to(device), token_type_ids=None, attention_mask=tokens['attention_mask'].to(device))
  # 将 pooled output 输入到线性层得到 logits
  logits = linear_layer(output[1])
  # 使用 softmax 计算概率，并获取类别 0 的概率
  prob = F.softmax(logits, dim=-1)[0][0].detach().cpu().numpy().item()
  if return_pooled_output:
    # 返回概率和 numpy 格式的 pooled output
    return prob, output[1][0].detach().cpu().numpy()
  else:
    # 只返回概率
    return prob

# 模型批量预测函数
def modelpredict(indices, model, batch_size=128, verbose=0, return_pool_result=True):
  """
  使用给定的模型对指定索引 (indices) 的数据进行批量预测。

  Args:
      indices (list or np.array): 需要预测的数据在 X_train 中的索引。
      model (torch.nn.Module): 用于预测的 PyTorch 模型 (通常是基础 Transformer 模型)。
      batch_size (int, optional): 预测时的批次大小. Defaults to 128.
      verbose (int, optional): 控制是否显示 tqdm 进度条 (0 表示不显示). Defaults to 0.
      return_pool_result (bool, optional): 是否返回模型的 pooled output. Defaults to True.

  Returns:
      tuple or np.array:
          - 如果 return_pool_result 为 True，返回 (预测概率数组, pooled output 数组)。
          - 如果 return_pool_result 为 False，仅返回预测概率数组。
          概率数组包含每个样本属于类别 1 的概率。
  """
  # 根据 verbose 参数决定是否禁用 tqdm 进度条
  TQDM_DISABLE = (verbose == 0)
  # 设置模型为评估模式
  model.eval()
  # 使用 ids2dataloader 创建预测数据的 DataLoader
  dataloader = ids2dataloader(indices, batch_size=batch_size) # 使用传入的 batch_size
  # 初始化用于存储预测概率和 pooled output 的数组
  probs = np.array([])
  pooled_outputs = np.empty((0, 768)) # 假设 pooled output 维度为 768

  # 遍历 DataLoader 中的每个批次
  for batch in tqdm(dataloader, disable=TQDM_DISABLE):
    # 将批次数据移动到指定设备 (GPU 或 CPU)
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask, b_labels = batch # 解包批次数据 (标签在这里未使用)

    # 在不计算梯度的模式下执行模型前向传播
    with torch.no_grad():
          # 获取模型输出，只需要 pooled_output
          _, pooled_output = model(b_input_ids,
                                   token_type_ids=None, # BERT 类模型需要，其他可能不需要
                                   attention_mask=b_input_mask,
                                   return_dict=False) # 禁用字典输出格式
    # 将 pooled output 输入线性层得到 logits
    logits = linear_layer(pooled_output)
    # 计算 softmax 概率，并提取类别 1 的概率
    prob = F.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
    # 将当前批次的概率追加到总数组
    probs = np.append(probs, prob)
    # 将当前批次的 pooled output 拼接到总数组
    pooled_outputs = np.concatenate((pooled_outputs, pooled_output.detach().cpu().numpy()), axis=0)

  if return_pool_result:
    # 返回概率和 pooled outputs
    return probs, pooled_outputs
  else:
    # 只返回概率
    return probs

# 模型测试评估函数
def modeltest(model, verbose=0):
    """
    在预定义的测试集 (test_dataloader) 上评估给定模型的性能。

    Args:
        model (torch.nn.Module): 需要评估的 PyTorch 模型 (基础 Transformer 模型)。
        verbose (int, optional): 控制是否显示 tqdm 进度条 (0 表示不显示). Defaults to 0.

    Returns:
        np.array: 包含 [f1_score, accuracy, auc, recall, precision] 的 NumPy 数组。
    """
    # 根据 verbose 参数决定是否禁用 tqdm 进度条
    TQDM_DISABLE = (verbose == 0)
    # 初始化用于存储预测概率的数组
    y_pred = np.array([])
    # 设置模型为评估模式
    model.eval()

    # 遍历测试数据 DataLoader
    for batch in tqdm(test_dataloader, disable=TQDM_DISABLE):
        # 将批次数据移动到设备
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch # 解包 (标签在此函数内计算指标时使用)

        # 不计算梯度
        with torch.no_grad():
            # 模型前向传播，获取 pooled output
            _, pooled_output = model(b_input_ids,
                                     token_type_ids=None,
                                     attention_mask=b_input_mask,
                                     return_dict=False)
        # 通过线性层获取 logits
        logits = linear_layer(pooled_output)
        # 计算 softmax 概率
        prob = F.softmax(logits, dim=-1)
        # 提取类别 1 的概率
        prob = prob[:, 1]
        # 将概率移到 CPU 并转为 NumPy 数组
        prob = prob.detach().cpu().numpy()
        # 追加到预测结果数组
        y_pred = np.append(y_pred, prob)

    # 将概率转换为 0/1 的预测标签 (阈值为 0.5)
    y_pred2 = np.zeros(y_pred.shape)
    y_pred2[y_pred > 0.5] = 1

    # 使用 scikit-learn 计算各项评估指标
    f1 = f1_score(y_test, y_pred2)       # F1 分数
    acc = accuracy_score(y_test, y_pred2)  # 准确率
    rec = recall_score(y_test, y_pred2)    # 召回率
    prec = precision_score(y_test, y_pred2) # 精确率
    # 计算 ROC 曲线和 AUC 值 (使用原始概率 y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
    auc_value = auc(fpr, tpr)

    # 返回包含各项指标的 NumPy 数组
    return np.array([f1, acc, auc_value, rec, prec])

# 模型训练函数
def modelfit(indices, model, epochs=18, verbose=1):
  """
  在给定的训练数据索引 (indices) 上训练 (微调) 模型。

  Args:
      indices (list or np.array): 用于训练的数据在 X_train/y_train 中的索引。
      model (torch.nn.Module): 需要训练的 PyTorch 模型 (基础 Transformer 模型)。
      epochs (int, optional): 训练的总轮数. Defaults to 18.
      verbose (int, optional): 控制是否显示 tqdm 进度条 (1 表示显示). Defaults to 1.

  Returns:
      torch.nn.Module: 训练完成的模型。
  """
  # 根据 verbose 参数决定是否禁用 tqdm 进度条
  TQDM_DISABLE = (verbose == 0)

  # --- 数据集划分与 DataLoader 创建 ---
  # 将传入的索引划分为训练集和验证集 (80% 训练, 20% 验证)
  train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=66) # 固定随机种子保证可复现性
  # 创建训练集和验证集的 DataLoader
  train_dataloader = ids2dataloader(train_indices, batch_size=32) # 使用默认 batch_size 32
  val_dataloader = ids2dataloader(val_indices, batch_size=128)   # 验证集使用较大 batch_size 128

  # --- 优化器与学习率调度器设置 ---
  # 计算总的训练步数
  total_steps = len(train_dataloader) * epochs
  # 创建学习率调度器 (线性预热 + 线性衰减)
  # num_warmup_steps = 0 表示没有预热阶段
  scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

  # --- 训练循环 ---
  # 存储每个 epoch 的平均训练损失
  loss_values = []

  # 按指定的 epoch 数进行循环训练
  for epoch in range(epochs):
      # 使用 tqdm 创建一个外部进度条，显示 epoch 信息
      with tqdm(total=len(train_dataloader), unit="batch", disable=TQDM_DISABLE) as tepoch:
        tepoch.set_description(f"Epoch {epoch+1}")

        # ========================================
        #               训练阶段
        # ========================================
        # 记录 epoch 开始时间
        t0 = time.time()
        # 初始化 epoch 的总损失
        total_loss = 0
        # 设置模型为训练模式
        model.train()

        # 遍历训练 DataLoader 中的每个批次
        for step, batch in enumerate(train_dataloader):
            # 更新内部进度条
            tepoch.update(1)

            # 将批次数据移动到指定设备
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].type(torch.LongTensor).to(device) # 标签需要是 LongTensor 类型

            # 清除之前的梯度
            model.zero_grad()
            linear_layer.zero_grad() # 确保线性层的梯度也清零

            # 模型前向传播
            _, pooled_output = model(input_ids=b_input_ids,
                                     token_type_ids=None,
                                     attention_mask=b_input_mask,
                                     return_dict=False)
            # 通过线性层得到 logits
            logits = linear_layer(pooled_output)

            # 计算损失
            train_loss = loss_func(logits, b_labels)
            # 累加 epoch 总损失
            total_loss += train_loss.item()

            # 反向传播计算梯度
            train_loss.backward()
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(linear_layer.parameters(), 1.0) # 对线性层也进行梯度裁剪

            # 更新模型参数
            optimizer.step()
            # 更新学习率
            scheduler.step()

            # 在进度条后缀中显示当前批次的训练损失
            tepoch.set_postfix({'Train loss': train_loss.item()})

        # 计算当前 epoch 的平均训练损失
        avg_train_loss = total_loss / len(train_dataloader)
        # 将平均损失存入列表
        loss_values.append(avg_train_loss)

        # ========================================
        #               验证阶段
        # ========================================
        # 设置模型为评估模式
        model.eval()

        # 初始化验证集的评估指标变量
        eval_accuracy = 0 # 这里只计算了准确率
        # 初始化用于存储验证集预测和标签的列表
        y_pred = np.array([])
        labels = np.array([])

        # 遍历验证 DataLoader
        for batch in val_dataloader:
            # 移动数据到设备
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            # 不计算梯度
            with torch.no_grad():
                # 模型前向传播
                _, pooled_output = model(b_input_ids,
                                         token_type_ids=None,
                                         attention_mask=b_input_mask,
                                         return_dict=False)
            # 通过线性层得到 logits
            logits = linear_layer(pooled_output)
            # 计算 softmax 概率并提取类别 1 的概率
            prob = F.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
            # 将真实标签移到 CPU 并转为 NumPy
            b_labels = b_labels.to('cpu').numpy()
            # 追加预测概率和真实标签
            y_pred = np.append(y_pred, prob)
            labels = np.append(labels, b_labels)

        # 计算验证集准确率 (基于 0.5 阈值)
        eval_accuracy = accuracy_score(labels.round(), y_pred.round())

        # 更新外部进度条的后缀，显示最终训练损失和验证集准确率
        tepoch.set_postfix({'Train loss (final)': avg_train_loss, 'Val acc': eval_accuracy})
        # 刷新并关闭当前 epoch 的进度条
        tepoch.refresh()
        tepoch.close()

  # 训练结束后返回训练好的模型
  return model

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

  # model.cuda()
  return model

# --- 主执行逻辑 ---

# 设置计算设备 (优先使用 GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义损失函数 (交叉熵损失)
loss_func = torch.nn.CrossEntropyLoss()
# 定义一个线性层，将 Transformer 的输出 (假设 768 维) 映射到 2 个类别 (二分类)
# 并将其移动到指定设备
linear_layer = torch.nn.Linear(768, 2, device=device)

# --- 多次运行实验 ---
# 根据模型名称执行不同的初始化和训练流程
if model_name == 'rta' or model_name == 'roberta' or model_name == 'codebert':
    # 对指定的模型类型，执行 10 次运行
    for run in range(0, 10):
        print(f"--- Starting Run {run} for model {model_name} ---")
        
        # --- 随机选择初始数据 ---
        # 从训练集的所有 bug ID 中随机抽取 'sample_size' 个索引
        print(f"Randomly sampling {sample_size} initial indices...")
        indices = random.sample(range(len(bug_train_ids)), sample_size)

        # 重新加载一个新的预训练模型实例
        model = reload_model()
        # 创建优化器，优化模型和线性层的参数
        # 注意：这里优化器每次循环都会重新创建
        optimizer = AdamW(list(model.parameters()) + list(linear_layer.parameters()), lr=3e-5, eps=1e-8)

        # --- 定义初始标注集和池 ---
        # 根据随机抽取的索引确定初始已标注的 bug ID
        bug_annotated_ids = bug_train_ids[indices]
        # 从所有训练 bug ID 中移除已标注的，得到初始的未标注池
        bug_pool_ids = np.delete(bug_train_ids, indices, axis=0)
        print(f"Initial annotated data size: {len(bug_annotated_ids)}")
        print(f"Initial pool size: {len(bug_pool_ids)}")

        # --- 模型训练 ---
        print("Starting model fitting...")
        # 使用加载的已标注 ID (bug_annotated_ids) 训练模型
        model = modelfit(bug_annotated_ids, model, epochs=EPOCH, verbose=1) # 使用全局 EPOCH 值
        print("Model fitting finished.")

        # --- 模型评估 ---
        print("Starting model testing...")
        # 在测试集上评估训练好的模型
        metrics_update = modeltest(model, verbose=1) # 使用 modeltest 获取评估指标
        print(f"Test Metrics (F1, Acc, AUC, Recall, Precision): {metrics_update}")

        # --- 保存结果 ---
        # 定义模型权重和数据文件的保存路径
        model_save_path = f'./initial_data/{model_name}/model{model_name}_run{run}_sample_size{sample_size}_model_weight_balance{balance_ratio}'
        data_save_path = f'./initial_data/{model_name}/model{model_name}_run{run}_sample_size{sample_size}_initial_data_balance{balance_ratio}.pkl'
        print(f"Saving model weights to: {model_save_path}")
        # 保存模型的状态字典 (包含基础模型和线性层)
        # 注意：需要确保优化器优化的是 model 和 linear_layer 的参数，否则只保存 model.state_dict() 可能不完整
        # 改为保存 model 和 linear_layer 的 state_dict
        torch.save({
            'model_state_dict': model.state_dict(),
            'linear_layer_state_dict': linear_layer.state_dict(),
            # 可以选择性保存优化器状态 'optimizer_state_dict': optimizer.state_dict(),
            }, model_save_path)

        print(f"Saving data and metrics to: {data_save_path}")
        # 将索引、ID 列表和更新后的评估指标打包
        files = indices, bug_annotated_ids, bug_pool_ids, metrics_update
        # 保存到新的 pickle 文件
        with open(data_save_path, 'wb') as pklfile:
            pickle.dump(files, pklfile)
        print(f"--- Run {run} finished. ---")

else: # 处理默认情况 (e.g., 'bert-base-uncased')
    # 执行 10 次运行
    for run in range(0, 10):
        print(f"--- Starting Run {run} for model {model_name} (Default/BERT) ---")
        # --- 随机选择初始数据 ---
        # 从训练集的所有 bug ID 中随机抽取 'sample_size' 个索引
        print(f"Randomly sampling {sample_size} initial indices...")
        indices = random.sample(range(len(bug_train_ids)), sample_size)

        # 重新加载一个新的预训练模型实例
        model = reload_model()
        # 创建优化器，优化模型和线性层的参数
        optimizer = AdamW(list(model.parameters()) + list(linear_layer.parameters()), lr=3e-5, eps=1e-8)

        # --- 定义初始标注集和池 ---
        # 根据随机抽取的索引确定初始已标注的 bug ID
        bug_annotated_ids = bug_train_ids[indices]
        # 从所有训练 bug ID 中移除已标注的，得到初始的未标注池
        bug_pool_ids = np.delete(bug_train_ids, indices, axis=0)
        print(f"Initial annotated data size: {len(bug_annotated_ids)}")
        print(f"Initial pool size: {len(bug_pool_ids)}")

        # --- 模型训练 ---
        print("Starting model fitting...")
        # 使用随机选择的已标注 ID 训练模型
        model = modelfit(bug_annotated_ids, model, epochs=EPOCH, verbose=1)
        print("Model fitting finished.")

        # --- 模型评估 ---
        print("Starting model testing...")
        # 在测试集上评估训练好的模型
        metrics_update = modeltest(model, verbose=1)
        print(f"Test Metrics (F1, Acc, AUC, Recall, Precision): {metrics_update}")

        # --- 保存结果 ---
        # 定义模型权重和数据文件的保存路径 (注意路径与上面分支不同)
        model_save_path = f'./initial_data/run{run}_sample_size{sample_size}_model_weight_balance{balance_ratio}' # 可能需要 model_name
        data_save_path = f'./initial_data/run{run}_sample_size{sample_size}_initial_data_balance{balance_ratio}.pkl' # 可能需要 model_name
        print(f"Saving model weights to: {model_save_path}")
        # 保存模型状态字典
        torch.save({
            'model_state_dict': model.state_dict(),
            'linear_layer_state_dict': linear_layer.state_dict(),
            }, model_save_path)

        print(f"Saving data and metrics to: {data_save_path}")
        # 将本次运行随机生成的索引、ID 列表和评估指标打包
        files = indices, bug_annotated_ids, bug_pool_ids, metrics_update
        # 保存到 pickle 文件
        with open(data_save_path, 'wb') as pklfile:
            pickle.dump(files, pklfile)
        print(f"--- Run {run} finished. ---")

print("All runs completed.")