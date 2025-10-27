# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss

class SCELoss(torch.nn.Module):
    def __init__(self, config, num_labels, alpha=1, beta=1):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.config = config
        self.num_labels = num_labels
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, pred, labels, score=None):
        # CCE
        ce = self.cross_entropy(pred, labels)
        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_labels).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))
        
        if score is None:
            loss = (self.alpha * ce).mean()  + self.beta * rce.mean() 
        else:
            loss = (self.alpha * ce).mean()  + self.beta * rce.mean() 
        return loss

def compute_kl_loss(p, q, pad_mask=None):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
    
    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.mean()
    q_loss = q_loss.mean()

    loss = (p_loss + q_loss)
    return loss

class Model(nn.Module):   
    def __init__(self, encoder, config, tokenizer):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.num_labels = 2
        # 添加分类头：将BERT的768维输出投影到2维（num_labels）
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.loss_fct = SCELoss(self.config, self.num_labels, alpha=1, beta=1)
    
    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, labels=None, score=None, input_aug_ids=None): 
        # 获取BERT输出（形状 [batch_size, seq_len, hidden_size]）
        outputs = self.encoder(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]  # [batch_size, seq_len, hidden_size]
        
        # 取[CLS] token的隐藏状态（形状 [batch_size, hidden_size]）
        cls_output = sequence_output[:, 0, :]
        
        # 通过分类头得到logits（形状 [batch_size, num_labels]）
        logits = self.classifier(cls_output)

        if labels is not None:
            loss = self.loss_fct(logits, labels.view(-1), score.view(-1))
            
            # 处理增强数据（同样取[CLS]并通过分类头）
            if input_aug_ids is not None:
                aug_outputs = self.encoder(input_aug_ids, attention_mask=input_aug_ids.ne(1))
                aug_cls_output = aug_outputs[0][:, 0, :]
                aug_logits = self.classifier(aug_cls_output)
                loss_aug = self.loss_fct(aug_logits, labels.view(-1), score.view(-1))
                kl_loss = compute_kl_loss(logits, aug_logits)
                loss = loss + loss_aug + self.args.k * kl_loss
            
            return loss, logits
        else:
            return logits




'''
class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        # x = x.reshape(-1,x.size(-1)*2)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class Model(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.classifier=RobertaClassificationHead(config)
        self.classifier_pseudo=RobertaClassificationHead(config)
        self.args=args
        if self.args.mode == 'pl_ours':
            self.loss_fct = SCELoss(self.config, self.config.num_labels, alpha=1, beta=1)
            #self.loss_fct = CrossEntropyLoss(reduction='none')
        else:
            self.loss_fct = CrossEntropyLoss(reduction='none')
    
    def forward(self, input_ids=None,labels=None,score=None): 
        outputs = self.encoder(input_ids,attention_mask=input_ids.ne(1))[0]
        logits=self.classifier(outputs)
        logits_pseudo=self.classifier_pseudo(outputs)
        if labels is not None:
            loss = (self.loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))*score).mean()
            loss_pseudo = (self.loss_fct(logits_pseudo.view(-1, self.config.num_labels), labels.view(-1))*(1-score)).mean()
            loss=loss+loss_pseudo
            return loss, logits
        else:
            return logits
'''
        
        
      
        
 
