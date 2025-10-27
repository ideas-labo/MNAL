import json
data_path = "./dataset/MNAL/eval.jsonl"

# with open(data_path, 'r', encoding='utf-8') as f:
    # i = 0
    # for line in f:
        # if(i >= 10):
            # break
        # data = json.loads(line.strip())  # 解析单行JSON
        # print(data)  # 打印字典对象
        # i += 1
        # 或按需访问字段：print(data["key"])

from utils_txt_clasfy import TextDataset
from transformers import BertConfig, BertForMaskedLM, BertTokenizer, BertModel
import torch
import torch.nn as nn

class BERTWithClassifier(nn.Module):
    def __init__(self, base_model, hidden_size=768, num_classes=2):
        super().__init__()
        self.base_model = base_model  # 基础Transformer模型（如BERT、RoBERTa等）
        self.classifier = nn.Linear(hidden_size, num_classes)  # 二分类线性层（与原linear_layer一致）
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, return_dict=False):
        # 基础模型前向传播，获取pooled_output
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=return_dict
        )
        # 对于return_dict=False，输出为元组 (last_hidden_state, pooled_output, ...)
        # 对于return_dict=True，输出为包含pooler_output的字典
        if return_dict:
            pooled_output = outputs.pooler_output
        else:
            pooled_output = outputs[1]  # 取元组的第二个元素（pooled_output）
        
        # 通过线性层计算logits（二分类输出）
        logits = self.classifier(pooled_output)
        
        # 保持与原输出格式一致：return_dict=False时返回元组，便于兼容原有代码
        if return_dict:
            return logits  # 或封装为字典，按需调整
        else:
            return (outputs[0], logits)  # (last_hidden_state, logits)，兼容原代码中对元组的处理


# --- 修改：reload_model函数，返回包含线性层的完整模型 ---
def reload_model(model_path):
    """
    根据全局变量 `model_name` 加载预训练模型，并绑定二分类线性层，返回完整模型。
    线性层维度：768 → 2（与原代码保持一致）。
    Returns:
        BERTWithClassifier: 包含基础模型和二分类线性层的完整模型。
    """
    base_model = BertModel.from_pretrained("bert-base-uncased")
    
    # 初始化包含二分类线性层的完整模型
    # 注意：RoBERTa、CodeBERT等基础模型的hidden_size均为768，与原线性层维度一致
    model = BERTWithClassifier(base_model, hidden_size=768, num_classes=2)
    # 加载模型权重
    state_dict = torch.load(model_path)
    # 过滤掉不必要的键
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    # # 加载微调过的权重
    model.load_state_dict(state_dict, strict=False) 
    return model

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
dataset = TextDataset(tokenizer, None, file_path=data_path)
teacher_model_path = "./save_models/teacher_models/run0_sample_size300_model_weight_balance(1, 1)"
model = reload_model(teacher_model_path)
# # 加载模型权重
# state_dict = torch.load(teacher_model_path)
# # 过滤掉不必要的键
# if 'model_state_dict' in state_dict:
    # state_dict = state_dict['model_state_dict']
# # # 加载微调过的权重
# model.load_state_dict(state_dict, strict=False)
print("load model success")

# 从数据集获取两个样本
sample1 = dataset[0]
sample2 = dataset[1]

# 解包每个样本的各个组件
inputs1, token_type_ids1, attention_mask1, label1, score1 = sample1
inputs2, token_type_ids2, attention_mask2, label2, score2 = sample2

# 将两个样本的张量在批次维度(第0维)上堆叠
inputs = torch.stack([inputs1, inputs2])
token_type_ids = torch.stack([token_type_ids1, token_type_ids2])
attention_mask = torch.stack([attention_mask1, attention_mask2])
label = torch.stack([label1, label2]) if isinstance(label1, torch.Tensor) else torch.tensor([label1, label2])
score = torch.stack([score1, score2]) if isinstance(score1, torch.Tensor) else torch.tensor([score1, score2])

# 打印批次形状
print(f"Batch inputs shape: {inputs.shape}")  # 应输出 [2, seq_length]

# 模型推理（批量处理）
# pooled_output = model(inputs, attention_mask = attention_mask)
_, logits = model(input_ids=inputs, attention_mask=attention_mask, return_dict=False)
print(logits)