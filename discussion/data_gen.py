import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 定义训练时每个批次的大小
batch_size = 32
# 定义BERT模型输入序列的最大长度
MAX_LEN = 100

# 导入数据处理和分析库
import pandas as pd
# 导入正则表达式库
import re
# 导入自然语言处理工具包
import nltk

# nltk.set_proxy('https://mirrors.tuna.tsinghua.edu.cn/nltk_data/')
# # 下载NLTK的停用词列表
# nltk.download('stopwords')
# # 下载NLTK的句子分词器模型
# nltk.download('punkt')
# 从NLTK导入单词分词器
from nltk.tokenize import word_tokenize
# 从NLTK导入停用词列表
from nltk.corpus import stopwords

# 导入数值计算库
import numpy as np
# 导入字符串常量库 (用于处理标点符号)
import string
# 导入JSON处理库 (虽然在此代码段中未直接使用)
import json
# 从scikit-learn导入OneHotEncoder用于标签编码
from sklearn.preprocessing import OneHotEncoder
# 从scikit-learn导入数据划分工具 (训练/测试集划分, K折交叉验证)
from sklearn.model_selection import train_test_split,KFold,StratifiedKFold # KFold 和 StratifiedKFold 未在此代码段中使用
# 导入PyTorch深度学习框架
import torch
# 导入随机数生成库
import random
# 从scikit-learn导入模型评估指标
from sklearn.metrics import (accuracy_score,recall_score,precision_score,f1_score, auc, roc_curve, confusion_matrix)
# 从PyTorch导入数据处理工具 (创建数据集和数据加载器)
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


# 从Hugging Face Transformers库导入BERT分词器
from transformers import BertTokenizer

# # 从PyTorch导入AdamW优化器 (此处被注释掉了)
# from torch.optim import AdamW
# 导入时间库
import time
# 导入日期时间库
import datetime
# 导入tqdm库，用于显示进度条
from tqdm import tqdm
# 导入数学库
import math
# 导入pickle库，用于序列化和反序列化Python对象 (保存/加载处理后的数据)
import pickle

from tool_funcs import balance_classes, parse_balance_ratio
import argparse  # 用于解析命令行参数

# --- 命令行参数解析 ---
# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser()

parser.add_argument('--balance_ratio', type=str, required=True, help='The balance ratio of the dataset')# 数据集中类别不平衡的比例

# 解析命令行传入的参数
args = parser.parse_args()
balance_ratio = parse_balance_ratio(args.balance_ratio)

# 加载训练数据集
train_data = pd.read_csv(f"./dataset/nlbse23-issue-classification-train.csv")

# 加载测试数据集
test_data = pd.read_csv(f"./dataset/nlbse23-issue-classification-test.csv")

print("load data success !")

# 再次定义最大序列长度 (与前面定义一致)
MAX_LEN = 100
# 定义最终测试集的大小
test_size = 5000

# 定义数据预处理函数
def data_process(data):
  """
  对输入的DataFrame进行文本预处理和标签编码。

  Args:
    data: 包含 'title', 'body', 和 'labels' 列的 pandas DataFrame。

  Returns:
    tuple: 包含处理后的文本数据 (X_text, pandas Series) 和
           处理后的标签 (y, numpy array)。
  """
  # 将 'title' 和 'body' 列合并为 'text' 列
  data['text'] = data["title"]+ data["body"]
  # 删除 'text' 列中包含NaN值的行
  data.dropna(subset=['text'], inplace=True)
  print("dropna success!")

  # --- 开始文本预处理 ---

  # 移除URL
  def remove_url(text):
    # 编译用于匹配URL的正则表达式
    url = re.compile(r'https?://\S+|www\.\S+')
    # 使用空字符串替换匹配到的URL
    return url.sub(r'', text)
  # 对 'text' 列应用移除URL的函数
  data["text"] = data["text"].apply(lambda x: remove_url(x))
  print("remove url success!")

  # 移除HTML标签
  def remove_html(text):
    # 编译用于匹配HTML标签的正则表达式
    html = re.compile(r'<.*?>')
    # 使用空字符串替换匹配到的HTML标签
    return html.sub(r'', text)
  # 对 'text' 列应用移除HTML标签的函数
  data["text"] = data["text"].apply(lambda x: remove_html(x))
  print("remove html success!")

  # 移除Emoji表情符号
  def remove_emoji(text):
    # 编译用于匹配多种Emoji Unicode范围的正则表达式
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    # 使用空字符串替换匹配到的Emoji
    return emoji_pattern.sub(r'', text)
  # 对 'text' 列应用移除Emoji的函数
  data["text"] = data["text"].apply(lambda x: remove_emoji(x))
  print("remove emoji success!")

  # 移除标点符号
  def remove_punctuation(text):
    # 创建一个转换表，将所有标点符号映射为None (即删除)
    table = str.maketrans('', '', string.punctuation)
    # 应用转换表移除标点
    return text.translate(table)
  # 对 'text' 列应用移除标点符号的函数
  data["text"] = data["text"].apply(lambda x: remove_punctuation(x))
  print("remove punctuation success!")

  # 移除停用词 (Stop Words)
  # 获取NLTK提供的标准英文停用词列表
  NLTK_stop_words_list = stopwords.words('english')
  # 定义自定义的停用词列表 (例如，省略号)
  custom_stop_words_list = ['...']
  # 合并标准和自定义停用词列表
  final_stop_words_list = NLTK_stop_words_list + custom_stop_words_list
  def remove_stopwords(text):
    """自定义函数，用于移除文本中的停用词"""
    # 分割文本为单词，保留不在停用词列表中的单词，然后重新组合成字符串
    return " ".join([word for word in str(text).split() if word not in final_stop_words_list])
  # 对 'text' 列应用移除停用词的函数
  data["text"] = data["text"].apply(lambda text: remove_stopwords(text))
  print("remove stopwords success!")

  # 转换为小写
  # 对 'text' 列应用小写转换函数
  data["text"] = data["text"].apply(lambda text: text.lower())

  # --- 文本预处理结束 ---

  # 提取处理后的文本作为特征 X
  X_text = data['text']

  # --- 标签处理 ---
  # # 获取y (多标签版本 - 此处被注释)
  # # 假设原始数据有多列代表不同标签，值为0, 1, 或 2
  # y = np.array(data[['bug', 'enhancement', 'question', 'ui', 'design', 'database', 'client', 'server', 'document', 'security', 'performance']])
  # # 将值为2的标签也视为1 (例如，存在该标签)
  # y[y==2]=1

  # 获取y (使用One-Hot编码处理单列标签)
  # 创建OneHotEncoder实例
  # handle_unknown='ignore': 遇到测试集中未在训练集中出现过的类别时忽略
  # dtype='int32': 指定编码后的数据类型
  encoder = OneHotEncoder(handle_unknown='ignore',dtype='int32')
  # 对 'labels' 列进行One-Hot编码，并转换为DataFrame
  encoder_df = pd.DataFrame(encoder.fit_transform(data[['labels']]).toarray())
  # 打印编码后的特征名称 (即原始标签类别)
  print("One-Hot Encoded Feature Names:", encoder.get_feature_names_out())
  # 将One-Hot编码后的DataFrame转换为NumPy数组作为标签y
  y = np.array(encoder_df)

  # --- 转换为二分类问题 ---
  # 注意: 这一步假设原始问题是多分类的，并且我们只关心第一个类别 (由OneHotEncoder排序决定，通常是字母顺序)
  # 例如，如果类别是 ['bug', 'feature', 'question']，这会提取 'bug' 类别的标签 (1表示是bug, 0表示不是)
  # 如果原始问题已经是二分类，这一步可能需要调整或移除
  y = y[:,0] # 选择One-Hot编码结果的第一列作为最终的二分类标签

  # 返回处理后的文本和标签
  return X_text, y

# 对训练数据进行预处理

X_train,y_train = data_process(train_data)
# X_train,y_train = balance_classes(X_train,y_train,ratio=(1,4))
X_train,y_train = balance_classes(X_train,y_train,ratio=balance_ratio)
print(f"Successfully constructed class-unbalanced dataset, balance_ratio : {balance_ratio}")

# 对原始测试数据进行预处理
X_test,y_test = data_process(test_data)
# X_test,y_test = balance_classes(X_test,y_test,ratio=(1,4))


# 打印训练集和测试集的大小
print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# 从处理后的测试数据中划分出最终的测试集
# 使用train_test_split，但实际上只取了test_size大小的样本作为最终测试集
# _ 表示丢弃不需要的部分 (这里丢弃了分割后的训练部分和对应的标签)
# random_state=66 确保每次划分结果一致，便于复现
_, X_test, _, y_test = train_test_split(X_test, y_test, test_size=test_size, random_state=66, stratify=y_test if len(np.unique(y_test)) > 1 else None) # 添加 stratify 以保持类别比例 (如果不是单类别)

# 创建一个包含训练数据原始索引的NumPy数组 (其用途可能在后续代码中，例如追踪样本)
bug_train_ids = np.array(range(len(X_train)))

# 将处理好的训练集、最终测试集及其索引打包成一个元组
files = X_test, y_test, bug_train_ids, X_train, y_train
# 指定保存处理后数据的pickle文件路径，文件名包含测试集大小
output_path = f'./initial_data/data_all_test_size{test_size}_balance{balance_ratio}.pkl' # 假设 ./initial_data 目录存在
# 以二进制写模式打开文件
with open(output_path,'wb') as pklfile:
  # 使用pickle将 'files' 元组序列化并保存到文件中
  pickle.dump(files, pklfile)
  print(f"Processed data saved to {output_path}")

# --- BERT Tokenization ---

# 加载预训练的'bert-base-uncased'模型对应的分词器
# do_lower_case=True: 在分词前将文本转换为小写 (虽然前面已经做过，但这是标准做法)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# 对处理后的训练文本 (X_train) 进行分词和编码
# list(X_train): 输入的文本列表
# max_length=MAX_LEN: 指定最大序列长度
# truncation=True: 如果序列超过max_length，则截断
# padding=True: 如果序列短于max_length，则填充到max_length (使用[PAD]标记)
# return_tensors="pt": 返回PyTorch张量
# return_attention_mask=True: 返回注意力掩码 (区分真实标记和填充标记)

print(f"Tokenizing training data with MAX_LEN={MAX_LEN}...")
tokenized_sentences = tokenizer(list(X_train), max_length=MAX_LEN,
                                truncation=True, padding=True,
                                return_tensors="pt", return_attention_mask=True)
print("Tokenization complete.")

# 将分词后的数据 (包含input_ids, token_type_ids, attention_mask) 赋值给 'files' 变量
files = tokenized_sentences
# 指定保存分词后数据的pickle文件路径
tokenized_output_path = f'./initial_data/test_size{test_size}_tokenized_sentences_balance{balance_ratio}.pkl' # 假设 ./initial_data 目录存在
# 以二进制写模式打开文件
with open(tokenized_output_path,'wb') as pklfile:
  # 使用pickle将分词后的数据序列化并保存到文件中
  pickle.dump(files, pklfile)
  print(f"Tokenized training data saved to {tokenized_output_path}")