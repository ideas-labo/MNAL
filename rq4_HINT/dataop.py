import json
import pandas as pd
import re
# 从NLTK导入停用词列表
from nltk.corpus import stopwords
import string
from sklearn.preprocessing import OneHotEncoder
import numpy as np

import random
from transformers import BertTokenizer

MAX_LEN = 100
def create_random_sample(input_file_path, output_file_path, sample_num):
    """
    从CSV文件中随机选择3000行数据，并创建一个新的CSV文件
    
    参数:
    input_file_path (str): 输入CSV文件的路径
    output_file_path (str): 输出CSV文件的路径
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(input_file_path)
        
        # 检查文件行数
        total_rows = len(df)
        if total_rows == 0:
            raise ValueError("CSV文件为空，没有数据可供采样")
        
        # 确定采样行数
        sample_size = min(sample_num, total_rows)
        
        # 随机选择行索引
        random_indices = random.sample(range(total_rows), sample_size)
        
        # 根据随机索引选择行
        sampled_df = df.iloc[random_indices]
        
        # 保存为新的CSV文件
        sampled_df.to_csv(output_file_path, index=False)
        
        print(f"成功创建随机样本文件: {output_file_path}")
        print(f"原始文件行数: {total_rows}, 采样行数: {sample_size}")
        
    except FileNotFoundError:
        print(f"错误: 找不到输入文件 - {input_file_path}")
    except Exception as e:
        print(f"发生未知错误: {e}")


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
  y = data["labels"]
  y = y.apply(lambda x : "1" if x == "bug" else "0")
  y = y.tolist()
  # 返回处理后的文本和标签
  return X_text, y

def save_dict_to_jsonl(data, filename):
    """
    将字典列表保存为 JSON Lines 格式的文件
    
    参数:
    data (list): 包含字典的列表
    filename (str): 输出文件名
    """
    with open(filename, 'w', encoding='utf-8') as f:
        # 如果 data 是单个字典，将其转换为列表
        if isinstance(data, dict):
            data = [data]
        
        # 逐行写入每个字典
        for item in data:
            # 确保中文等非 ASCII 字符正确编码
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')



def text_to_dict(texts, labels):
   
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    tokenized_sentences = tokenizer(list(texts), max_length=MAX_LEN,
                                    truncation=True, padding=True,
                                    return_tensors="pt", return_attention_mask=True)

    # 转换为 numpy 数组
    input_ids_np = tokenized_sentences["input_ids"].numpy()

    res = []
    # 遍历每个样本
    for i in range(input_ids_np.shape[0]):
        # tokens = input_ids_np[i].tolist()
        text_tokens = tokenizer.convert_ids_to_tokens(input_ids_np[i])
        res.append({"text_tokens": text_tokens, "label": labels[i]})
    return res

# 示例用法
if __name__ == "__main__":
    train_data_path = "./dataset/MNAL/nlbse23-issue-classification-train-sample3000.csv"
    eval_data_path = "./dataset/MNAL/nlbse23-issue-classification-train-sample300.csv"
    test_data_path = "./dataset/MNAL/nlbse23-issue-classification-test-sample5000.csv"
    # create_random_sample(test_data_path,"./dataset/MNAL/nlbse23-issue-classification-test-sample5000.csv", 5000)
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)
    eval_data = pd.read_csv(eval_data_path)
    
    X_train, y_train = data_process(train_data)
    X_test, y_test = data_process(test_data)
    X_eval, y_eval = data_process(eval_data)
    print("data process success!")

    train_dict = text_to_dict(X_train, y_train)
    test_dict = text_to_dict(X_test, y_test)
    eval_dict = text_to_dict(X_eval, y_eval)
    print("tokenized success!")

    save_dict_to_jsonl(train_dict, "./dataset/MNAL/train.jsonl")
    save_dict_to_jsonl(test_dict, "./dataset/MNAL/test.jsonl")
    save_dict_to_jsonl(eval_dict, "./dataset/MNAL/eval.jsonl")
    
