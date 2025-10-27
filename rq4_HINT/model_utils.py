import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score, precision_score
import pandas as pd
from tqdm import tqdm
import numpy as np
import os
from datetime import datetime

def train_model(model, train_dataset, eval_dataset, args):
    """
    训练模型并定期评估
    
    参数:
        model: 要训练的模型
        train_dataset: 训练数据集
        eval_dataset: 评估数据集
        tokenizer: 用于文本处理的tokenizer
        args: 包含训练参数的命名空间
        
    返回:
        训练完成的模型和评估结果DataFrame
    """
    # 创建DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.train_batch_size, shuffle=False)
    
    # 设置优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # 记录评估结果
    results = []
    
    # 训练循环
    for epoch in range(args.epoch):
        print(f"\nEpoch {epoch + 1}/{args.epoch}")
        model.train()
        total_loss = 0
        
        # 训练步骤
        progress_bar = tqdm(train_dataloader, desc="Training")
        for batch in progress_bar:
            # 准备输入数据
            inputs = {
                'input_ids': batch[0],
                'token_type_ids': batch[1],
                'attention_mask': batch[2],
                'labels': batch[3],
                'score': batch[4]
            }
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(**inputs)
            loss = outputs[0] if isinstance(outputs, tuple) else outputs
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Average training loss: {avg_train_loss:.4f}")
        
        # 评估模型
        eval_results = evaluate_model(model, eval_dataloader)
        eval_results['epoch'] = epoch + 1
        eval_results['train_loss'] = avg_train_loss
        results.append(eval_results)
        
        # 打印评估结果
        print(f"\nEvaluation results after epoch {epoch + 1}:")
        for metric, value in eval_results.items():
            if metric not in ['epoch', 'train_loss']:
                print(f"{metric}: {value:.4f}")
    
        # 将结果保存为DataFrame
        results_df = pd.DataFrame(results)
        
        # 保存结果到Excel
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"./expriment_result/model_evaluation_results_{timestamp}_epoch_{epoch}.xlsx"
        results_df.to_excel(output_file, index=False)
        print(f"\nAll evaluation results saved to {output_file}")

        model_save_path = f'./save_models/model_{timestamp}_epoch_{epoch}'
        print(f"Saving model weights to: {model_save_path}")
        # 保存模型状态字典
        torch.save({
            'model_state_dict': model.state_dict(),
            }, model_save_path)

    
    return model, results_df

def evaluate_model(model, eval_dataloader):
    """
    评估模型性能
    
    参数:
        model: 要评估的模型
        eval_dataloader: 评估数据加载器
        device: 使用的设备
        
    返回:
        包含评估指标的字典
    """
    model.eval()
    predictions = []
    true_labels = []
    probabilities = []
    
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            inputs = {
                'input_ids': batch[0],
                'token_type_ids': batch[1],
                'attention_mask': batch[2]
            }

            
            # 获取模型输出
            outputs = model(**inputs)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            
            # 计算概率和预测结果
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(batch[3].cpu().numpy())
            probabilities.extend(probs[:, 1].cpu().numpy())  # 正类的概率
    
    # 计算各项指标
    results = {
        'accuracy': accuracy_score(true_labels, predictions),
        'f1-score': f1_score(true_labels, predictions),
        'AUC': roc_auc_score(true_labels, probabilities),
        'recall': recall_score(true_labels, predictions),
        'precision': precision_score(true_labels, predictions)
    }
    
    return results

# # 使用示例
# if __name__ == "__main__":
    # import argparse
    
    # # 设置训练参数
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--batch_size", type=int, default=32)
    # parser.add_argument("--num_epochs", type=int, default=18)
    # parser.add_argument("--learning_rate", type=float, default=2e-5)
    # parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    # parser.add_argument("--k", type=float, default=0.1, help="Weight for KL divergence loss")
    # args = parser.parse_args()
    
    # # 假设已经有以下对象:
    # # model = Model(encoder, config, tokenizer)
    # # train_dataset = ... (您的训练数据集)
    # # eval_dataset = ... (您的评估数据集)
    # # tokenizer = ... (您的tokenizer)
    
    # # 训练模型
    # trained_model, results = train_model(model, train_dataset, eval_dataset, tokenizer, args)