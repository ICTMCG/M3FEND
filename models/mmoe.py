from genericpath import exists
import os
import torch
import tqdm
import torch.nn as nn
import numpy as np
from .layers import *
from sklearn.metrics import *
from transformers import BertModel
from transformers import RobertaModel
from utils.utils import data2gpu, Averager, metrics, Recorder
import logging

class MMoEModel(torch.nn.Module):
    def __init__(self, emb_dim, num_layers, mlp_dims, dropout, num_head, dataset):
        super(MMoEModel, self).__init__()
        self.num_expert = 5
        self.num_head = num_head
        if dataset == 'ch':
            self.bert = BertModel.from_pretrained('hfl/chinese-bert-wwm-ext').requires_grad_(False)
        elif dataset == 'en':
            self.bert = RobertaModel.from_pretrained('roberta-base').requires_grad_(False)
        
        expert = []
        for i in range(self.num_expert):
            expert.append(MLP(emb_dim, mlp_dims, dropout, False))
        self.expert = nn.ModuleList(expert)

        head = []
        for i in range(self.num_head):
            head.append(torch.nn.Linear(mlp_dims[-1], 1))
        self.head = nn.ModuleList(head)

        gate = []
        for i in range(self.num_head):
            gate.append(torch.nn.Sequential(torch.nn.Linear(emb_dim, mlp_dims[-1]),
                                                   torch.nn.ReLU(),
                                                   torch.nn.Linear(mlp_dims[-1], self.num_expert),
                                                   torch.nn.Softmax(dim=1)))
        self.gate = nn.ModuleList(gate)

        self.attention = MaskAttention(emb_dim)
    
    def forward(self, **kwargs):
        inputs = kwargs['content']
        masks = kwargs['content_masks']
        category = kwargs['category']
        feature = self.bert(inputs, attention_mask = masks)[0]

        feature, _ = self.attention(feature, masks)
        gate_value = []
        for i in range(feature.size(0)):
            gate_value.append(self.gate[category[i]](feature[i].view(1, -1)))
        gate_value = torch.cat(gate_value)

        rep = 0
        for i in range(self.num_expert):
            rep += (gate_value[:, i].unsqueeze(1) * self.expert[i](feature))

        output = []
        for i in range(feature.size(0)):
            output.append(self.head[category[i]](rep[i].view(1, -1)))
        output = torch.cat(output)
        return torch.sigmoid(output.squeeze(1))


class Trainer():
    def __init__(self,
                 emb_dim,
                 mlp_dims,
                 use_cuda,
                 lr,
                 dropout,
                 num_layers, 
                 train_loader,
                 val_loader,
                 test_loader,
                 category_dict,
                 weight_decay,
                 save_param_dir,
                 dataset,
                 early_stop = 5,
                 epoches = 100
                 ):
        self.lr = lr
        self.weight_decay = weight_decay
        self.use_cuda = use_cuda
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.early_stop = early_stop
        self.epoches = epoches
        self.category_dict = category_dict
        self.dataset = dataset

        self.emb_dim = emb_dim
        self.mlp_dims = mlp_dims
        self.dropout = dropout
        self.num_layers = num_layers
        
        if os.path.exists(save_param_dir):
            self.save_param_dir = save_param_dir
        else:
            self.save_param_dir = save_param_dir
            os.makedirs(save_param_dir)
        

    def train(self, logger = None):
        if(logger):
            logger.info('start training......')
        self.model = MMoEModel(self.emb_dim, self.num_layers, self.mlp_dims, self.dropout, num_head = len(self.category_dict), dataset = self.dataset)
        if self.use_cuda:
            self.model = self.model.cuda()
        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        recorder = Recorder(self.early_stop)
        for epoch in range(self.epoches):
            self.model.train()
            train_data_iter = tqdm.tqdm(self.train_loader)
            avg_loss = Averager()

            for step_n, batch in enumerate(train_data_iter):
                batch_data = data2gpu(batch, self.use_cuda)
                label = batch_data['label']

                optimizer.zero_grad()
                pred = self.model(**batch_data)
                loss = loss_fn(pred, label.float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss.add(loss.item())
            print('Training Epoch {}; Loss {}; '.format(epoch + 1, avg_loss.item()))
            status = '[{0}] lr = {1}; batch_loss = {2}; average_loss = {3}'.format(epoch, str(self.lr), loss.item(), avg_loss.item())

            results = self.test(self.val_loader)
            mark = recorder.add(results)
            if mark == 'save':
                torch.save(self.model.state_dict(),
                    os.path.join(self.save_param_dir, 'parameter_mmoe.pkl'))
                best_metric = results['metric']
            elif mark == 'esc':
                break
            else:
                continue
        self.model.load_state_dict(torch.load(os.path.join(self.save_param_dir, 'parameter_mmoe.pkl')))
        results = self.test(self.test_loader)
        if(logger):
            logger.info("start testing......")
            logger.info("test score: {}\n\n".format(results))
        print(results)
        return results, os.path.join(self.save_param_dir, 'parameter_mmoe.pkl')

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
                batch_pred = self.model(**batch_data)

                label.extend(batch_label.detach().cpu().numpy().tolist())
                pred.extend(batch_pred.detach().cpu().numpy().tolist())
                category.extend(batch_category.detach().cpu().numpy().tolist())
        
        return metrics(label, pred, category, self.category_dict)