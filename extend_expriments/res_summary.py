import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json

def save_to_json(input_dict, filename='data.json'):
    """
    将字典保存为JSON文件
    
    参数:
    input_dict: 要保存的字典
    filename: 保存的文件名，默认为'data.json'
    
    返回:
    bool: 保存成功返回True，失败返回False
    """
    file_path = f'../data_state/uncertainty_new/json_res/{filename}'
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(input_dict, f, ensure_ascii=False, indent=4)
        print(f"字典已成功保存到 {file_path}")
        return True
    except Exception as e:
        print(f"保存失败: {e}")
        return False

def calculate_stats_and_save(input_dict, filename='stats_result.json'):
    """
    计算字典中每个列表的均值和方差，保存为JSON文件并返回结果字典
    
    参数:
    input_dict: 输入字典，键为字符串，值为数值列表
    filename: 保存的JSON文件名，默认为'stats_result.json'
    
    返回:
    dict: 键为原字典的键，值为(均值, 方差)的元组
    """
    
    # 创建结果字典
    result_dict = {}
    
    # 遍历输入字典中的每个键值对
    for key, value_list in input_dict.items():
        # 将列表转换为numpy数组以便计算
        arr = np.array(value_list)
        
        # 计算均值和方差
        mean_val = float(np.mean(arr))
        var_val = float(np.std(arr))  # 总体方差
        # 如果需要样本方差，使用 np.var(arr, ddof=1)
        
        # 存储结果
        result_dict[key] = (mean_val, var_val)
    
    return result_dict


def load_pkl_file(file_path):
    try:
        # 以二进制读取模式打开文件
        with open(file_path, 'rb') as file:
            # 加载文件内容
            data = pickle.load(file)
            return data
    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 未找到！")
    except pickle.UnpicklingError:
        print(f"错误：无法解析文件 '{file_path}'，可能文件已损坏或不是有效的pickle文件。")
    except Exception as e:
        print(f"错误：发生未知错误 - {e}")
    return None


def rq2_step(runs, balance_ratio, step, initial_size=300, sample_size=300):


    res = {"expriment_name" : f"uncertainty with balance ratio {balance_ratio} result in every step",
        "expriment_result" : []
    }

    for run in range(1, runs+1):
        run_data = {"run" : run,
            "step_msgs" : []
        }
        for step in range(1, 11):
            file_path = f'../data_state/uncertainty_new/rq2/initial_size{initial_size}_sample_size{sample_size}_run_{run}_step{step}-10_normalized_sum_nonononormalized_sumno_pseudo2_ratio1_balance{balance_ratio}.pkl'  # 替换为你的pkl文件路径
            data = load_pkl_file(file_path)
            x = {"step" : step}
            if data is not None:
                x["f1_score"] = data[6][0]
                x["Acc"] = data[6][1]
                x["AUC"] = data[6][2]
                x["Prec"] = data[6][3]
                x["Prec"] = data[6][4]
                x["Read"] = np.mean(data[9])
                x["Identify"] = np.mean(data[10])
                run_data["step_msgs"].append(x)
        res["expriment_result"].append(run_data)
    return res


def upper_bound_step(runs, balance_ratio, step, initial_size=300, sample_size=3000):


    res = {"expriment_name" : f"upperbound of uncertainty with balance ratio {balance_ratio} result in every step",
        "expriment_result" : []
    }

    for run in range(1, runs+1):
        run_data = {"run" : run,
            "step_msgs" : []
        }
        for step in range(1, 11):
            file_path = f'../data_state/uncertainty_new/upper_bound/initial_size{initial_size}_sample_size{sample_size}_run_{run}_step{step}-10_normalized_sum_nonononormalized_sumno_pseudo2_ratio1_balance{balance_ratio}.pkl'  # 替换为你的pkl文件路径
            data = load_pkl_file(file_path)
            x = {"step" : step}
            if data is not None:
                x["f1_score"] = data[6][0]
                x["Acc"] = data[6][1]
                x["AUC"] = data[6][2]
                x["Prec"] = data[6][3]
                x["Prec"] = data[6][4]
                x["Read"] = np.mean(data[9])
                x["Identify"] = np.mean(data[10])
                run_data["step_msgs"].append(x)
        res["expriment_result"].append(run_data)
    return res

if __name__ == '__main__':
    # res = diversity_test(8, '(1, 1)', 10, sample_size=300)
    # print(res)
    # mean_std_res = calculate_stats_and_save(res)
    # final_res = {
        # 'expriment name' : 'diversity_test',
        # 'query size' : '300',
        # 'expriment result' : mean_std_res,
    # }
    # save_to_json(final_res, 'diversity_test_query_size300.json')
    # print(final_res)

    # res = rq2_step(8, '(1, 1)', 10)
    # save_to_json(res, 'diversity_test_query_size300_in_every_step.json')

    # res = rq2_step(8, '(1, 1)', 10)
    # save_to_json(res, 'rq2_query_size300_in_every_step.json')
    # res = rq2_step(8, '(1, 4)', 10)
    # save_to_json(res, 'rq2_query_size300_balance_(1,4)_in_every_step.json')
    # res = rq2_step(8, '(4, 1)', 10)
    # save_to_json(res, 'rq2_query_size300_balance_(4,1)_in_every_step.json')
    # res = rq2_step(8, '(1, 9)', 10)
    # save_to_json(res, 'rq2_query_size300_balance_(1,9)_in_every_step.json')
    # res = rq2_step(8, '(9, 1)', 10)
    # save_to_json(res, 'rq2_query_size300_balance_(9,1)_in_every_step.json')

    res = upper_bound_step(3, '(1, 1)', 10)
    save_to_json(res, 'upperbound_query_size3000_balance_(1,1)_in_every_step.json')