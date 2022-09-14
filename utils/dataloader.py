import torch
import random
import pandas as pd
import tqdm
import numpy as np
import pickle
import re
from transformers import BertTokenizer
from transformers import RobertaTokenizer
from torch.utils.data import TensorDataset, DataLoader

def _init_fn(worker_id):
    np.random.seed(2021)

def read_pkl(path):
    with open(path, "rb") as f:
        t = pickle.load(f)
    return t

def df_filter(df_data, category_dict):
    df_data = df_data[df_data['category'].isin(set(category_dict.keys()))]
    return df_data

def word2input(texts, max_len, dataset):
    if dataset == 'ch':
        tokenizer = BertTokenizer.from_pretrained('hfl/chinese-bert-wwm-ext')
    elif dataset == 'en':
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    token_ids = []
    for i, text in enumerate(texts):
        token_ids.append(
            tokenizer.encode(text, max_length=max_len, add_special_tokens=True, padding='max_length',
                             truncation=True))
    token_ids = torch.tensor(token_ids)
    masks = torch.zeros(token_ids.shape)
    mask_token_id = tokenizer.pad_token_id
    for i, tokens in enumerate(token_ids):
        masks[i] = (tokens != mask_token_id)
    return token_ids, masks

def process(x):
    x['content_emotion'] = x['content_emotion'].astype(float)
    return x
    

class bert_data():
    def __init__(self, max_len, batch_size, category_dict, dataset, num_workers=2):
        self.max_len = max_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.category_dict = category_dict
        self.dataset = dataset
    
    def load_data(self, path, shuffle):
        self.data = df_filter(read_pkl(path), self.category_dict)
        content = self.data['content'].to_numpy()
        comments = self.data['comments'].to_numpy()
        content_emotion = torch.tensor(np.vstack(self.data['content_emotion']).astype('float32'))
        comments_emotion = torch.tensor(np.vstack(self.data['comments_emotion']).astype('float32'))
        emotion_gap = torch.tensor(np.vstack(self.data['emotion_gap']).astype('float32'))
        style_feature = torch.tensor(np.vstack(self.data['style_feature']).astype('float32'))
        label = torch.tensor(self.data['label'].astype(int).to_numpy())
        category = torch.tensor(self.data['category'].apply(lambda c: self.category_dict[c]).to_numpy())
        content_token_ids, content_masks = word2input(content, self.max_len, self.dataset)
        comments_token_ids, comments_masks = word2input(content, self.max_len, self.dataset)
        dataset = TensorDataset(content_token_ids,
                                content_masks,
                                comments_token_ids,
                                comments_masks,
                                content_emotion,
                                comments_emotion,
                                emotion_gap,
                                style_feature,
                                label,
                                category
                                )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=shuffle,
            worker_init_fn=_init_fn
        )
        return dataloader