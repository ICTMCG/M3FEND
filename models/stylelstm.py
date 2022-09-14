# from math import dist
import os
import torch
from torch.autograd import Variable
import tqdm
import torch.nn as nn
import numpy as np
from .layers import *
from sklearn.metrics import *
from transformers import BertModel
from transformers import RobertaModel
from utils.utils import data2gpu, Averager, metrics, Recorder
import logging

class StyleLstmModel(torch.nn.Module):
    def __init__(self, emb_dim, mlp_dims, dropout):
        super(StyleLstmModel, self).__init__()
        self.fea_size = emb_dim
        if dataset == 'ch':
            self.bert = BertModel.from_pretrained('hfl/chinese-bert-wwm-ext').requires_grad_(False)
        elif dataset == 'en':
            self.bert = RobertaModel.from_pretrained('roberta-base').requires_grad_(False)
        
        self.rnn = nn.GRU(input_size = emb_dim,
                    hidden_size = self.fea_size,
                    num_layers = 1,
                    batch_first = True,
                    bidirectional = True)
        self.attention = MaskAttention(self.fea_size * 2)
        if dataset == 'ch':
            self.classifier = MLP(self.fea_size * 2 + 48, mlp_dims, dropout)
        elif dataset == 'en':
            self.classifier = MLP(self.fea_size * 2 + 32, mlp_dims, dropout)
    
    def forward(self, **kwargs):
        content = kwargs['content']
        content_masks = kwargs['content_masks']
        style_feature = kwargs['style_feature']
        
        content_feature = self.bert(content, attention_mask = content_masks)[0]
        content_feature, _ = self.rnn(content_feature)
        content_feature, _ = self.attention(content_feature, content_masks)

        shared_feature = torch.cat([content_feature, style_feature], dim=1)

        label_pred = self.classifier(shared_feature)

        return torch.sigmoid(label_pred.squeeze(1))

class Trainer():
    def __init__(self,
                 emb_dim,
                 mlp_dims,
                 use_cuda,
                 lr,
                 dropout,
                 train_loader,
                 val_loader,
                 test_loader,
                 category_dict,
                 weight_decay,
                 save_param_dir,
                 early_stop = 5,
                 epoches = 100
                 ):
        self.lr = lr
        self.weight_decay = weight_decay
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.early_stop = early_stop
        self.epoches = epoches
        self.category_dict = category_dict
        self.use_cuda = use_cuda

        self.emb_dim = emb_dim
        self.mlp_dims = mlp_dims
        self.dropout = dropout
        
        if os.path.exists(save_param_dir):
            self.save_param_dir = save_param_dir
        else:
            self.save_param_dir = save_param_dir
            os.makedirs(save_param_dir)
        

    def train(self, logger = None):
        if(logger):
            logger.info('start training......')
        self.model = StyleLstmModel(self.emb_dim, self.mlp_dims, self.dropout)
        if self.use_cuda:
            self.model = self.model.cuda()
        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        recorder = Recorder(self.early_stop)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 100, gamma = 0.98)
        for epoch in range(self.epoches):
            self.model.train()
            train_data_iter = tqdm.tqdm(self.train_loader)
            avg_loss = Averager()

            for step_n, batch in enumerate(train_data_iter):
                batch_data = data2gpu(batch, self.use_cuda)
                label = batch_data['label']
                category = batch_data['category']
                optimizer.zero_grad()
                label_pred = self.model(**batch_data)
                loss =  loss_fn(label_pred, label.float()) 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if(scheduler is not None):
                    scheduler.step()
                avg_loss.add(loss.item())
                
            print('Training Epoch {}; Loss {}; '.format(epoch + 1, avg_loss.item()))
            status = '[{0}] lr = {1}; batch_loss = {2}; average_loss = {3}'.format(epoch, str(self.lr), loss.item(), avg_loss.item())

            results = self.test(self.val_loader)
            mark = recorder.add(results)
            if mark == 'save':
                torch.save(self.model.state_dict(),
                    os.path.join(self.save_param_dir, 'parameter_dualemotion.pkl'))
                best_metric = results['metric']
            elif mark == 'esc':
                break
            else:
                continue
        self.model.load_state_dict(torch.load(os.path.join(self.save_param_dir, 'parameter_dualemotion.pkl')))
        results = self.test(self.test_loader)
        if(logger):
            logger.info("start testing......")
            logger.info("test score: {}\n\n".format(results))
        print(results)
        return results, os.path.join(self.save_param_dir, 'parameter_dualemotion.pkl')

    def test(self, dataloader):
        pred = []
        label = []
        category = []
        self.model.eval()
        data_iter = tqdm.tqdm(dataloader)
        for step_n, batch in enumerate(data_iter):
            with torch.no_grad():
                batch_data = data2gpu(batch, self.use_cuda)
                batch_label = batch_data['label']
                batch_category = batch_data['category']
                batch_label_pred = self.model(**batch_data)

                label.extend(batch_label.detach().cpu().numpy().tolist())
                pred.extend(batch_label_pred.detach().cpu().numpy().tolist())
                category.extend(batch_category.detach().cpu().numpy().tolist())
        
        return metrics(label, pred, category, self.category_dict)
