import argparse
from transformers import( 
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaModel, RobertaForSequenceClassification, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
from utils_txt_clasfy import TextDataset
from transformers import BertModel # BERT 模型基础结构
import torch
import torch.nn as nn
from model import Model
from model_utils import train_model
import os


MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
}

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
def reload_model(model_save_path):
    """
    加载保存的模型权重和线性层参数
    
    参数:
        model: 要加载权重的模型实例
        model_save_path: 模型权重保存路径
        device: 加载设备
        
    返回:
        加载了权重的模型
    """
    # 检查文件是否存在
    if not os.path.exists(model_save_path):
        raise FileNotFoundError(f"Model weights file not found at {model_save_path}")
    
    base_model = BertModel.from_pretrained("bert-base-uncased")
    model = BERTWithClassifier(base_model) 
    # 加载保存的状态字典
    checkpoint = torch.load(model_save_path)
    
    # 加载基础模型权重
    if 'model_state_dict' in checkpoint:
        model.base_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise KeyError("'model_state_dict' not found in checkpoint")
    
    # 加载线性层权重
    if hasattr(model, 'classifier') and 'linear_layer_state_dict' in checkpoint:
        model.classifier.load_state_dict(checkpoint['linear_layer_state_dict'])
    elif hasattr(model, 'linear_layer') and 'linear_layer_state_dict' in checkpoint:
        model.linear_layer.load_state_dict(checkpoint['linear_layer_state_dict'])
    else:
        print("Warning: Linear layer weights not found or model has no linear layer attribute")
    
    print(f"Successfully loaded model weights from {model_save_path}")
    return model


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=False,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
                    
    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")    
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_total_limit', type=int, default=None,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=123456,
                        help="random seed for initialization")
    parser.add_argument('--epoch', type=int, default=10,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    parser.add_argument('--mode', type=str, default='', help="For debugging.")
    parser.add_argument('--threshold', type=float, default=0.8, help="For debugging.")
    parser.add_argument('--edit', type=float, default=0.8, help="For debugging.")
    parser.add_argument('--teacher_path', type=str, default='', help="For debugging.")
    parser.add_argument('--unlabel_filename', type=str, default='', help="For debugging.")
    parser.add_argument('--device', type=str, default='cpu', help="Device")
    
    args = parser.parse_args()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.model_name_or_path)
    config.num_labels=2
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name)

    # 加载初始模型权重
    teacher_model_path = "./save_models/teacher_models/run0_sample_size300_model_weight_balance(1, 1)"
    model = reload_model(teacher_model_path) # 加载基础 BERT
    print("load model success")
    
    model=Model(model,config,tokenizer)
    print("Training/evaluation parameters %s", args)
    train_dataset = TextDataset(tokenizer, args, args.train_data_file)
    test_dataset = TextDataset(tokenizer, args, args.test_data_file)
    print("pl_ours success, train_dataset generated, begin train !")
    train_model(model, train_dataset, test_dataset, args)
    print("model training completed !")

if __name__ == '__main__':
    main()