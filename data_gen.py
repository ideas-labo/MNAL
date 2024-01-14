batch_size = 32
MAX_LEN = 100

import pandas as pd
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
import string
import json
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split,KFold,StratifiedKFold
import torch
import random
from sklearn.metrics import (accuracy_score,recall_score,precision_score,f1_score, auc, roc_curve, confusion_matrix)
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer
# from torch.optim import AdamW
import time
import datetime
from tqdm import tqdm
import math
import pickle

# data = pd.read_csv(f'/content/drive/MyDrive/issue_data.csv')
train_data = pd.read_csv(f"nlbse23-issue-classification-train.csv")
test_data = pd.read_csv(f"nlbse23-issue-classification-test.csv")
MAX_LEN = 100
test_size = 5000

def data_process(data):
  data['text'] = data["title"]+ data["body"]
  data.dropna(subset=['text'], inplace=True)

  # preprocessing

  # removing URLs
  def remove_url(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'', text)
  data["text"] = data["text"].apply(lambda x: remove_url(x))

  # removing html
  def remove_html(text):
    html = re.compile(r'<.*?>')
    return html.sub(r'', text)
  data["text"] = data["text"].apply(lambda x: remove_html(x))

  # removing emoji
  def remove_emoji(text):
    emoji_pattern = re.compile("["
                              u"\U0001F600-\U0001F64F" #emoticons
                              u"\U0001F300-\U0001F5FF" #symbols&pics
                              u"\U0001F680-\U0001F6FF" #transportation pic
                              u"\U0001F1E0-\U0001F1FF" #flags
                              u"\U00002702-\U000027B0"
                              u"\U000024C2-\U0001F251"
                              "]+", flags = re.UNICODE)
    return emoji_pattern.sub(r'', text)
  data["text"] = data["text"].apply(lambda x: remove_emoji(x))

  # removing punctuation
  def remove_punctuation(text):
    table = str.maketrans('', '', string.punctuation)
    return text.translate(table)
  data["text"] = data["text"].apply(lambda x: remove_punctuation(x))

  # Stop Word Removal
  NLTK_stop_words_list = stopwords.words('english')
  custom_stop_words_list = ['...']
  final_stop_words_list = NLTK_stop_words_list + custom_stop_words_list
  def remove_stopwords(text):
    """custom function to remove the stopwords"""
    return " ".join([word for word in str(text).split() if word not in final_stop_words_list])
  data["text"] = data["text"].apply(lambda text: remove_stopwords(text))

  # lowercase conversion
  data["text"] = data["text"].apply(lambda text: text.lower())

  # # get X
  X_text = data['text']

  # # get y
  # y = np.array(data[['bug', 'enhancement', 'question', 'ui', 'design', 'database', 'client', 'server', 'document', 'security', 'performance']])
  # y[y==2]=1

  # get y
  #creating instance of one-hot-encoder
  encoder = OneHotEncoder(handle_unknown='ignore',dtype='int32')
  #perform one-hot encoding on column
  encoder_df = pd.DataFrame(encoder.fit_transform(data[['labels']]).toarray())
  print(encoder.get_feature_names_out())
  y = np.array(encoder_df)

  # binary
  y = y[:,0]
  return X_text, y

X_train,y_train = data_process(train_data)
X_test,y_test = data_process(test_data)
_, X_test, _, y_test = train_test_split(X_test, y_test, test_size=test_size,random_state=66)

bug_train_ids = np.array(range(len(X_train)))

files = X_test, y_test, bug_train_ids, X_train, y_train
with open(f'/initial_data/data_all_test_size{test_size}.pkl','wb') as pklfile:
  pickle.dump(files, pklfile)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
tokenized_sentences = tokenizer(list(X_train), max_length=MAX_LEN,
                              truncation=True, padding=True,
                              return_tensors="pt", return_attention_mask=True)
files = tokenized_sentences
with open(f'/initial_data/test_size{test_size}_tokenized_sentences.pkl','wb') as pklfile:
  pickle.dump(files, pklfile)