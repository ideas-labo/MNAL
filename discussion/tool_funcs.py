import numpy as np
import pandas as pd

# 二分查找最合适的目标样本数
def find_base_num(num0, num1, ratio):
    l = 1
    r = min(num0, num1)
    while l < r:
        mid = int((l + r + 1) / 2)
        if mid * ratio[0] <= num0 and mid * ratio[1] <= num1:
            l = mid
        else:
            r = mid - 1
    return l

def balance_classes(X_text, y_test, ratio=(1, 1)):
    """
    调整样本分布使两类样本满足指定比例（保持原始输入类型）
    
    参数:
        X_text (list/pd.Series/np.array): 文本数据列，形状(n,)
        y_test (np.ndarray/pd.Series): 对应的标签列，形状(n,)
        ratio (tuple): 两类样本的目标比例，如(1,1)或(1,2)
        
    返回:
        tuple: (X_text_balanced, y_balanced)，保持X_text原始类型
    """
    # 记录原始输入类型
    input_is_series = isinstance(X_text, pd.Series)
    input_is_array = isinstance(X_text, np.ndarray)
    input_is_list = isinstance(X_text, list)
    
    # 输入验证
    assert len(ratio) == 2, "比例参数应为二元组(如(1,1))"
    assert len(X_text) == len(y_test), "X_text和y_test长度必须相同"
    
    # 统一转换为DataFrame处理
    df = pd.DataFrame({
        'text': X_text if isinstance(X_text, (pd.Series, np.ndarray, list)) else list(X_text),
        'label': y_test if isinstance(y_test, (pd.Series, np.ndarray)) else np.array(y_test)
    })
    
    # 检查类别数
    class_counts = df['label'].value_counts()
    if len(class_counts) != 2:
        raise ValueError("y_test应包含两个类别")
    
    # 获取两类样本
    df_class0 = df[df['label'] == 0]
    df_class1 = df[df['label'] == 1]
    
    base_num = find_base_num(len(df_class0), len(df_class1), ratio)
    target_0 = base_num * ratio[0]
    target_1 = base_num * ratio[1]

    print(f"目标样本数: 0={target_0}, 1={target_1}")
    
    # 下采样
    df_class0 = df_class0.sample(target_0, random_state=20)
    df_class1 = df_class1.sample(target_1, random_state=20)
    
    # 合并并打乱顺序
    balanced_df = pd.concat([df_class0, df_class1]).sample(frac=1, random_state=20)
    
    # 恢复原始类型
    X_balanced = balanced_df['text']
    y_balanced = balanced_df['label'].values
    
    if input_is_series:
        return pd.Series(X_balanced).reset_index(drop=True), y_balanced
    elif input_is_array:
        return X_balanced.values, y_balanced
    elif input_is_list:
        return X_balanced.tolist(), y_balanced
    else:
        return X_balanced.values, y_balanced  # 默认返回numpy数组


import numpy as np

def random_subset_indices(bug_train_ids, subset_size):
    """
    从 bug_train_ids 数组中随机抽取指定数量的元素，并返回一个新的 NumPy 数组及其索引。
    :param bug_train_ids: NumPy 数组
    :param subset_size: 需要抽取的样本数量
    :return: 包含随机抽取样本的新 NumPy 数组和这些样本在原数组中的索引
    """
    # 使用 np.random.choice 从 bug_train_ids 中随机抽取 subset_size 个元素
    random_subset = np.random.choice(bug_train_ids, size=subset_size, replace=False)
    
    # 获取这些元素在原数组中的索引
    indices = np.searchsorted(bug_train_ids, random_subset)
    
    return random_subset, indices


import os
import argparse

def rename(directory, balance_ratio):
    # 遍历指定文件夹下的所有文件
    for filename in os.listdir(directory):
        # 获取文件的完整路径
        filepath = os.path.join(directory, filename)
        
        # 检查是否为文件
        if os.path.isfile(filepath):
            # 获取文件名和文件后缀
            name, ext = os.path.splitext(filename)
            
            # 构建新的文件名
            new_filename = f"{name}_balance{balance_ratio}{ext}"
            
            # 获取新的文件路径
            new_filepath = os.path.join(directory, new_filename)
            
            # 重命名文件
            os.rename(filepath, new_filepath)
            print(f"Renamed: {filename} -> {new_filename}")

import ast

def parse_balance_ratio(value):
    try:
        # 尝试解析为元组（如果输入是 "(4,1)" 或 "4,1"）
        if '(' in value and ')' in value:
            return ast.literal_eval(value)  # 安全地解析字符串为 Python 对象
        else:
            return tuple(map(int, value.split(',')))
    except (ValueError, SyntaxError):
        raise argparse.ArgumentTypeError(f"Invalid balance_ratio format: {value}")

if __name__ == "__main__":
        # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description='Rename files in a directory by adding a suffix.')
    
    # 添加命令行参数
    # parser.add_argument('--directory', type=str, required=True, help='The directory to process.')
    # parser.add_argument('--balance_ratio', type=str, required=True, help='The balance ratio of the dataset')
    parser.add_argument('--balance_ratio', type=tuple, default=(1, 1), help='The balance ratio of the dataset.')

    # 解析命令行参数
    args = parser.parse_args()
    
    # 获取参数值
    # balance_ratio = parse_balance_ratio(args.balance_ratio)
    balance_ratio = args.balance_ratio
    print(balance_ratio)
    print(type(balance_ratio))
    # # 获取参数值
    # directory = './data_state/rq2'
    # balance_ratio = (1,4)
    
    # # 调用主函数
    # rename(directory, balance_ratio)

