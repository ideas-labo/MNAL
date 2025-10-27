import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
from math import log

import numpy as np
import torch
import copy
import heapq
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
import json
import jsonlines

from tqdm import tqdm, trange
import torch.nn.functional as F
import multiprocessing
cpu_cont = multiprocessing.cpu_count()
import editdistance
import multiprocessing
from rank_bm25 import BM25Okapi  # 补充BM25库引用

debug_sign = False

# 加载训练数据并构建BM25索引（针对文本内容）
train_data = []
with jsonlines.open('./dataset/MNAL/eval.jsonl') as f:
    for obj in f:
        train_data.append(obj)
# 使用文本内容构建BM25，而非代码token
text_corpus = [' '.join(obj['text_tokens']) for obj in train_data]
bm25_model = BM25Okapi([doc.split() for doc in text_corpus])  # 分词处理
average_idf = sum(map(lambda k: float(bm25_model.idf[k]), bm25_model.idf.keys())) / len(bm25_model.idf.keys())



def process(raw_data): # 该函数用于多线程批量处理样本,也可单线程使用
    """处理数据，通过BM25找到最相似的训练样本"""
    processed = []
    for idx in tqdm(range(len(raw_data))):
        obj = raw_data[idx]
        # 使用文本内容作为查询，而非代码
        query = ' '.join(obj['text_tokens']).split()
        # score = bm25_model.get_scores(query, average_idf)
        score = bm25_model.get_scores(query)
        # 找到最相似的训练样本索引
        rtn = sorted(enumerate(score), key=lambda x: x[1], reverse=True)[0]
        obj['dist_id'] = rtn[0]
        processed.append(obj)
    return processed    


class InputFeatures(object):
    """文本分类任务的特征结构（修复版）"""
    def __init__(self,
                 source_tokens,  # 原始token列表（调试用）
                 input_ids,      # 转换为id的输入序列
                 token_type_ids, # 句子类型标识（区分句子1和句子2）
                 attention_mask, # 注意力掩码（区分有效token和padding）
                 label,          # 标签
                 score=None      # 可选：样本置信度分数（用于伪标签等场景）
    ):
        self.source_tokens = source_tokens  # 原始token列表（用于调试）
        self.input_ids = input_ids          # 模型输入的token ID序列
        self.token_type_ids = token_type_ids  # 句子类型ID（单句文本通常全为0）
        self.attention_mask = attention_mask  # 注意力掩码（标识有效token和padding）
        self.label = label                  # 分类标签
        self.score = score                  # 可选：样本置信度分数（如伪标签的可信度）

def convert_examples_to_features(js, tokenizer, args):
    global debug_sign
    """将文本样本转换为模型输入特征（修复版）"""
    # 1. 处理文本并生成token
    text = ' '.join(str(token) for token in js['text_tokens'])  # 确保所有token为字符串
    text_tokens = tokenizer.tokenize(text)
    
    # 2. 截断到最大长度（预留[CLS]和[SEP]的位置）
    max_content_length = 100 - 2  # 减去2个特殊token
    text_tokens = text_tokens[:max_content_length]  # 截断
    
    # 3. 构建完整输入序列（包含特殊token）
    source_tokens = [tokenizer.cls_token] + text_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    
    # 4. 填充到最大长度
    padding_length = 100 - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length  # 用pad_token_id填充（通常为0）
    
    # 5. 生成token_type_ids（单句文本全为0）
    token_type_ids = [0] * 100  # 长度与source_ids一致
    
    # 6. 生成attention_mask（1=有效token，0=padding）
    attention_mask = [1] * len(source_tokens) + [0] * padding_length  # 有效部分为1，padding为0
    
    # 7. 处理标签和分数
    label = int(js['label']) if 'label' in js else 0  # 兼容无标签场景
    score = float(js['score']) if 'score' in js else 1.0
    
    # 调试信息
    if debug_sign:
        print("Tokens:", source_tokens)
        print("Input IDs:", source_ids)
        print("Token Type IDs:", token_type_ids)
        print("Attention Mask:", attention_mask)
        print("Label:", label)
        print("Score:", score)
        debug_sign = False
    
    # 返回包含所有必要特征的对象（确保InputFeatures能接收这些字段）
    return InputFeatures(
        source_tokens=source_tokens,
        input_ids=source_ids,
        token_type_ids=token_type_ids,
        attention_mask=attention_mask,
        label=label,
        score=score
    )

class TextDataset(Dataset):
    """文本分类任务的数据集类"""
    def __init__(self, tokenizer, args, file_path=None, teacher_model=None):
        self.examples = []
        self.tokenizer = tokenizer

        if file_path is not None:
            # 加载数据文件
            with open(file_path) as f:
                for line in f:
                    js = json.loads(line.strip())
                    self.examples.append(convert_examples_to_features(js, tokenizer, args))
        # 如果有教师模型,则添加伪标签数据
        if teacher_model is not None:
            # 生成伪标签
            pseudo_labeling(teacher_model, args, tokenizer)
            # 加载筛选后的伪标签数据
            with open(os.path.join(args.output_dir, 'selected_pseudo.jsonl')) as f:
                for line in f:
                    js = json.loads(line.strip())
                    self.examples.append(convert_examples_to_features(js, tokenizer, args))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        """返回模型输入：输入ID、标签、置信度分数"""
        example = self.examples[i]
        return (
            torch.tensor(example.input_ids, dtype=torch.long),
            torch.tensor(example.token_type_ids, dtype=torch.long),  # 关键：转换为long
            torch.tensor(example.attention_mask, dtype=torch.long),
            torch.tensor(example.label, dtype=torch.long),
            torch.tensor(example.score, dtype=torch.float)
        )

def set_seed(seed=42):
    """设置随机种子，保证实验可复现"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def pseudo_labeling(model, args, tokenizer):
    """生成伪标签数据（针对无标签文本）"""
    # 如果伪标签文件不存在，则生成
    if not os.path.exists(os.path.join(args.output_dir, 'pseudo.jsonl')):
        print("begin pseudo")
        # 加载无标签数据
        eval_dataset = TextDataset(tokenizer, args, args.unlabel_filename)
        args.eval_batch_size = 32 
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=args.eval_batch_size,
            num_workers=4,
            pin_memory=True
        )
        # 教师模型预测伪标签
        model.eval()
        logits = []
        probs = []
        for batch in eval_dataloader:
            inputs = batch[0]
            token_type_ids = batch[1]
            attention_mask = batch[2]
            label = batch[3] # 无标签数据此处可能为占位符
            score = batch[4]
            with torch.no_grad():
                # 调用模型获取预测结果（文本分类模型输出logits）
                loss, logit = model(inputs, token_type_ids, attention_mask, label, score)
                logits.append(logit.cpu().numpy())
                # 计算置信度（取预测概率的最大值）
                probs.append(F.softmax(logit, dim=-1).max(dim=-1)[0].cpu().numpy())
        # 合并所有预测结果
        logits = np.concatenate(logits, 0)
        probs = np.concatenate(probs, 0)
        preds = np.argmax(logits, axis=-1)  # 伪标签

        # 保存伪标签数据
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        pseudo_data = []
        for idx in range(len(preds)):
            pseudo_data.append({
                'text_tokens': eval_dataset.examples[idx].source_tokens,  # 文本token
                'label': int(preds[idx]),  # 伪标签
                'score': float(probs[idx])  # 置信度
            })
        with jsonlines.open(os.path.join(args.output_dir, 'pseudo.jsonl'), mode='w') as f:
            f.write_all(pseudo_data)
            print("pesudo label success")

    # 预处理伪标签数据（通过BM25找相似样本）
    pseudo_data = []
    with jsonlines.open(os.path.join(args.output_dir, 'pseudo.jsonl')) as f:
        for i in f:
            pseudo_data.append(i)

    processed_results = []
    for idx in tqdm(range(len(pseudo_data))):
        obj = pseudo_data[idx]
        # 直接调用process函数处理单个样本
        processed_obj = process([obj])[0]  # process函数返回列表，取第一个元素
        processed_results.append(processed_obj)
        
    # 保存处理后的伪标签数据
    with jsonlines.open(os.path.join(args.output_dir, 'pseudo_dis.jsonl'), mode='w') as f:
        f.write_all(processed_results)
        print("cal dis success")


    # 筛选高质量伪标签数据（基于置信度和相似度）
    train_data = []
    if args.train_data_file is not None:
        with jsonlines.open(args.train_data_file) as f:
            for obj in f:
                train_data.append(obj)
    pseudo_data = []
    pos_total_loss = []
    neg_total_loss = []
    with jsonlines.open(os.path.join(args.output_dir, 'pseudo_dis.jsonl')) as f:
        for i in f:
            pseudo_data.append(i)
            # 记录不同类别样本的损失（用于筛选阈值）
            if int(i['label']) == 1:
                pos_total_loss.append(-log(i['score']))  # 损失与置信度负相关
            else:
                neg_total_loss.append(-log(i['score']))
    # 计算筛选阈值（保留置信度较高的样本）
    pos_total_loss.sort()
    neg_total_loss.sort()
    pos_threshold = pos_total_loss[int(len(pos_total_loss) * args.threshold)]
    neg_threshold = neg_total_loss[int(len(neg_total_loss) * args.threshold)]
    print(f'正类筛选阈值: {pos_threshold}')
    print(f'负类筛选阈值: {neg_threshold}')

    # 筛选伪标签数据
    pos_selected = []
    neg_selected = []
    for obj in tqdm(pseudo_data):
        dist_id = obj['dist_id']  # 最相似的训练样本索引
        # 计算与相似样本的编辑距离（文本相似度）
        source_norm = editdistance.eval(
            train_data[dist_id]['text_tokens'],
            obj['text_tokens']
        ) / len(obj['text_tokens'])
        s = 0.4  # 文本相似度阈值
        # 根据类别和相似度筛选
        if int(obj['label']) == 1:
            loss_val = -log(obj['score'])
            if source_norm <= s and int(obj['label']) == int(train_data[dist_id]['label']):
                pos_selected.append(obj)  # 相似度高且标签一致
            elif loss_val < pos_threshold:
                pos_selected.append(obj)  # 置信度高
        else:
            loss_val = -log(obj['score'])
            if source_norm <= s and int(obj['label']) == int(train_data[dist_id]['label']):
                neg_selected.append(obj)  # 相似度高且标签一致
            elif loss_val < neg_threshold:
                neg_selected.append(obj)  # 置信度高

    # 平衡正负样本数量（根据文本分类任务调整比例）
    pos_num = len(pos_selected)
    neg_num = len(neg_selected)
    # 调整正负样本比例（可根据任务需求修改）
    selected_pseudo = (
        random.sample(pos_selected, k=min(pos_num, neg_num)) +
        random.sample(neg_selected, k=min(neg_num, pos_num))
    )
    print(f'筛选后正类: {pos_num}, 负类: {neg_num}, 总样本: {len(selected_pseudo)}')

    # 保存筛选后的伪标签数据
    with jsonlines.open(os.path.join(args.output_dir, 'selected_pseudo.jsonl'), mode='w') as f:
        f.write_all(selected_pseudo)