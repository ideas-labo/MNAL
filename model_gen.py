import argparse
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--model_name')
parser.add_argument('--initial_size')
args = parser.parse_args()

EPOCH = 18
model_name = args.model_name
sample_size = int(args.initial_size)

sample_times = 10
batch_size = 32
MAX_LEN = 100

import argparse
import pandas as pd
import re
import nltk
import numpy as np
import string
import json
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split,KFold,StratifiedKFold
import torch
import random
from sklearn.metrics import (accuracy_score,recall_score,precision_score,f1_score, auc, roc_curve, confusion_matrix)
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer,get_linear_schedule_with_warmup,AutoTokenizer,RobertaTokenizer
from torch.optim import AdamW
import time
import datetime
from tqdm import tqdm
import math
import pickle
# import logging
import torch.nn.functional as F
import math
from abc import abstractmethod
import numpy as np
from sklearn.metrics import pairwise_distances
from transformers import BertModel, RobertaModel, AutoModel
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import argparse
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import textstat
from collections import Counter
from transformers import BertModel

test_size = 5000

with open(f'/gpfs/home/i/ideaslab/work1/data_all_test_size{test_size}.pkl','rb') as pklfile:
  files = pickle.load(pklfile)
X_test, y_test, bug_train_ids, X_train, y_train = files

if model_name == 'rta':
  tokenizer = AutoTokenizer.from_pretrained("Colorful/RTA")
elif model_name == 'roberta':
  tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
elif model_name == 'codebert':
  tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
else: 
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
test_tokenized_sentences = tokenizer(list(X_test), max_length=MAX_LEN,
                              truncation=True, padding=True,
                              return_tensors="pt", return_attention_mask=True)
test_data = TensorDataset(torch.tensor(test_tokenized_sentences['input_ids']), torch.tensor(test_tokenized_sentences['attention_mask']), torch.tensor(y_test))
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=128)

def ids2dataloader(ids,batch_size=32):
  tokenized_sentences = tokenizer(list(X_train.iloc[ids]), max_length=MAX_LEN,
                                truncation=True, padding=True,
                                return_tensors="pt", return_attention_mask=True)
  # data = TensorDataset(torch.tensor(tokenized_sentences['input_ids']), torch.tensor(tokenized_sentences['attention_mask']), torch.tensor(y_train[ids]))

  data = TensorDataset(tokenized_sentences['input_ids'].clone().detach(), tokenized_sentences['attention_mask'].clone().detach(), torch.tensor(y_train[ids]))
  sampler = SequentialSampler(data)
  dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
  return dataloader

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def single_id_predict(id,return_pooled_output=False):
  # tokens = tokenizer(X_train.iloc[id], max_length=MAX_LEN,truncation=True, padding=True,return_tensors="pt", return_attention_mask=True)
  # with torch.no_grad():
  #   outputs = model(tokens['input_ids'].to(device),token_type_ids=None,attention_mask=tokens['attention_mask'].to(device))
  # prob = F.softmax(outputs[0], dim=-1)[0][0].detach().cpu().numpy().item()
  tokens = tokenizer(X_train.iloc[id], max_length=MAX_LEN,truncation=True, padding=True,return_tensors="pt", return_attention_mask=True)
  with torch.no_grad():
    output = model(tokens['input_ids'].to(device),token_type_ids=None,attention_mask=tokens['attention_mask'].to(device))
  logits = linear_layer(output[1])
  prob = F.softmax(logits, dim=-1)[0][0].detach().cpu().numpy().item()
  if return_pooled_output:
    return prob,output[1][0].detach().cpu().numpy()
  else:
    return prob

def modelpredict(indices,model,batch_size=128,verbose=0,return_pool_result=True):
  if verbose == 0:
    TQDM_DISABLE = True
  else:
    TQDM_DISABLE = False
  model.eval()
  dataloader = ids2dataloader(indices,batch_size=128)
  probs = np.array([])
  pooled_outputs = np.empty((0,768))
  # Evaluate data for one epoch
  for batch in tqdm(dataloader,disable=TQDM_DISABLE):
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask, b_labels = batch

    with torch.no_grad():

          _, pooled_output = model(b_input_ids,
                          token_type_ids=None,
                          attention_mask=b_input_mask,return_dict=False)
    logits = linear_layer(pooled_output)
    prob = F.softmax(logits, dim=-1)[:,1].detach().cpu().numpy()
    probs = np.append(probs,prob)
    pooled_outputs = np.concatenate((pooled_outputs, pooled_output.detach().cpu().numpy()), axis=0)

  if return_pool_result:
    return probs,pooled_outputs
  else:
    return probs


def modeltest(model,verbose=0):
    if verbose == 0:
      TQDM_DISABLE = True
    else:
      TQDM_DISABLE = False
    y_pred = np.array([])

    # Evaluate data for one epoch

    for batch in tqdm(test_dataloader,disable=TQDM_DISABLE):

        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():

            _, pooled_output = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,return_dict=False)
        # cls_emb = cls_emb[0][:, 0, :].squeeze(1)
        logits = linear_layer(pooled_output)
        prob = F.softmax(logits, dim=-1)
        prob = prob[:,1]
        prob = prob.detach().cpu().numpy()
        y_pred = np.append(y_pred, prob)
    y_pred2 = np.zeros(y_pred.shape)
    y_pred2[y_pred>0.5] = 1
    f1 = f1_score(y_test, y_pred2)
    acc = accuracy_score(y_test, y_pred2)
    rec = recall_score(y_test, y_pred2)
    prec = precision_score(y_test, y_pred2)
    fpr, tpr, thresholds = roc_curve(y_test,y_pred,pos_label=1)
    auc_value=auc(fpr,tpr)
    # current_precision = precision_score(y_true, y_pred2)
    # print('precision is ',current_precision)
    # current_recall = recall_score(y_true, y_pred2)
    # print('recall is ',current_recall)
    # current_accuracy = accuracy_score(y_true, y_pred2)
    # print('accuracy is ',current_accuracy)
#     confusion_matrix = metrics.multilabel_confusion_matrix(y_test, y_pred2)
#     print(confusion_matrix)

    return np.array([f1,acc,auc_value,rec,prec])

def modelfit(indices, model, epochs = 18, verbose = 1):
  if verbose == 0:
    TQDM_DISABLE = True
  else:
    TQDM_DISABLE = False
  train_indices, val_indices = train_test_split(indices, test_size=0.2,random_state=66)
  train_dataloader = ids2dataloader(train_indices)
  val_dataloader = ids2dataloader(val_indices)
  total_steps = len(train_dataloader) * epochs
  scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps = 0,num_training_steps = total_steps)

  # # Set the seed value all over the place to make this reproducible.
  # seed_val = 42

  # random.seed(seed_val)
  # np.random.seed(seed_val)
  # torch.manual_seed(seed_val)
  # torch.cuda.manual_seed_all(seed_val)

  # Store the average loss after each epoch so we can plot them.
  loss_values = []

  for epoch in range(epochs):

      # ========================================
      #               Training
      # ========================================
      with tqdm(total=len(train_dataloader), unit="batch", disable=TQDM_DISABLE) as tepoch:
        tepoch.set_description(f"Epoch {epoch+1}")
        # Perform one full pass over the training set.
        t0 = time.time()
        total_loss = 0

        model.train()
        # for batch in tepoch:
        for step, batch in enumerate(train_dataloader):
            tepoch.update(1)
            # # Progress update every 40 batches.
            # if step % 40 == 0 and not step == 0:
            #     # Calculate elapsed time in minutes.
            #     elapsed = format_time(time.time() - t0)

            #     # Report progress.
            #     print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].type(torch.LongTensor)
            b_labels = b_labels.to(device)

            model.zero_grad()
            _, pooled_output = model(input_ids=b_input_ids, token_type_ids=None, attention_mask=b_input_mask,return_dict=False)
            # cls_emb = cls_emb[0][:, 0, :].squeeze(1)
            logits = linear_layer(pooled_output)
            # print(logits)

            train_loss = loss_func(logits, b_labels)
            # results = outputs.cuda().data.cpu().argmax(dim=1)

            total_loss += train_loss.item()
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)


            optimizer.step()
            scheduler.step()
            tepoch.set_postfix({'Train loss': train_loss.item()})


        avg_train_loss = total_loss / len(train_dataloader)
        # tepoch.set_postfix(loss=loss.item(),avg_train_loss=avg_train_loss)
        loss_values.append(avg_train_loss)

        # print("")
        # print("  Average training loss: {0:.2f}".format(avg_train_loss))
        # print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))

        # ========================================
        #               Validation
        # ========================================

        model.eval()

        # Tracking variables
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        # Evaluate data for one epoch
        y_pred = np.array([])
        labels = np.array([])
        for batch in val_dataloader:

            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            with torch.no_grad():

                _, pooled_output = model(b_input_ids,
                                token_type_ids=None,
                                attention_mask=b_input_mask,return_dict=False)
            # cls_emb = cls_emb[0][:, 0, :].squeeze(1)
            logits = linear_layer(pooled_output)
            prob = F.softmax(logits, dim=-1)
            prob = prob[:,1]
            prob = prob.detach().cpu().numpy()
            b_labels = b_labels.to('cpu').numpy()
            y_pred = np.append(y_pred, prob)
            labels = np.append(labels, b_labels)

        eval_accuracy = accuracy_score(labels.round(), y_pred.round())
            # nb_eval_steps += 1

        # Report the final accuracy for this validation run.
        # tqdm.write(f"\bUsing tqdm.write: {eval_accuracy/nb_eval_steps}")
        tepoch.set_postfix({'Train loss (final)': train_loss.item(), 'Val acc': eval_accuracy})
        tepoch.refresh()
        tepoch.close()
        # print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
        # print("  Validation took: {:}".format(format_time(time.time() - t0)))
  return model



def reload_model():
  if model_name == 'rta':
    model = AutoModel.from_pretrained("Colorful/RTA")
  elif model_name == "roberta":
    model = RobertaModel.from_pretrained("roberta-base")
  elif model_name == 'codebert':
    model = AutoModel.from_pretrained("microsoft/codebert-base")
  else:
    model = BertModel.from_pretrained("bert-base-uncased")
  # Tell pytorch to run this model on the GPU.
  model.cuda()
  return model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loss_func = torch.nn.CrossEntropyLoss()
linear_layer = torch.nn.Linear(768, 2, device=device)

if model_name == 'rta' or model_name == 'roberta' or model_name == 'codebert':
  for run in range(0,10):
    model = reload_model()
    optimizer = AdamW(model.parameters(),lr = 3e-5,eps = 1e-8)

    with open(f'/gpfs/home/i/ideaslab/work1/initial_data/run{run}_sample_size{sample_size}_initial_data.pkl','rb') as pklfile:
      indices, bug_annotated_ids, bug_pool_ids, metrics_update = pickle.load(pklfile)

    model = modelfit(bug_annotated_ids,model,epochs=18,verbose=0)
    metrics_update=modeltest(model,verbose=0)

    torch.save(model.state_dict(), f'/gpfs/home/i/ideaslab/work1/initial_data_new/model{model_name}_run{run}_sample_size{sample_size}_model_weight')

    files = indices, bug_annotated_ids, bug_pool_ids, metrics_update
    with open(f'/gpfs/home/i/ideaslab/work1/initial_data_new/model{model_name}_run{run}_sample_size{sample_size}_initial_data.pkl','wb') as pklfile:
      pickle.dump(files, pklfile)

else:  
  for run in range(0,10):
    # # initial run
    indices = random.sample(range(len(bug_train_ids)),sample_size)
    model = reload_model()
    optimizer = AdamW(model.parameters(),lr = 3e-5,eps = 1e-8)

    bug_annotated_ids = bug_train_ids[indices]
    bug_pool_ids = np.delete(bug_train_ids, indices, axis=0)

    model = modelfit(bug_annotated_ids,model,epochs=18,verbose=0)
    metrics_update=modeltest(model,verbose=0)

    torch.save(model.state_dict(), f'/gpfs/home/i/ideaslab/work1/initial_data/run{run}_sample_size{sample_size}_model_weight')
    files = indices, bug_annotated_ids, bug_pool_ids, metrics_update
    with open(f'/gpfs/home/i/ideaslab/work1/initial_data/run{run}_sample_size{sample_size}_initial_data.pkl','wb') as pklfile:
      pickle.dump(files, pklfile)
