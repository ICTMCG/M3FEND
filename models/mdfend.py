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

class MultiDomainFENDModel(torch.nn.Module):
    def __init__(self, emb_dim, mlp_dims, domain_num, dropout, dataset):
        super(MultiDomainFENDModel, self).__init__()
        self.domain_num = domain_num
        self.gamma = 10
        self.num_expert = 5
        self.fea_size =256
        if dataset == 'ch':
            self.bert = BertModel.from_pretrained('hfl/chinese-bert-wwm-ext').requires_grad_(False)
        elif dataset == 'en':
            self.bert = RobertaModel.from_pretrained('roberta-base').requires_grad_(False)
        
        feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64, 10: 64}
        expert = []
        for i in range(self.num_expert):
            expert.append(cnn_extractor(feature_kernel, emb_dim))
        self.expert = nn.ModuleList(expert)

        self.gate = nn.Sequential(nn.Linear(emb_dim, mlp_dims[-1]),
                                      nn.ReLU(),
                                      nn.Linear(mlp_dims[-1], self.num_expert),
                                      nn.Softmax(dim = 1))

        self.attention = MaskAttention(emb_dim)

        self.domain_embedder = nn.Embedding(num_embeddings = self.domain_num, embedding_dim = emb_dim)
        self.specific_extractor = SelfAttentionFeatureExtract(multi_head_num = 1, input_size=emb_dim, output_size=self.fea_size)
        self.classifier = MLP(320, mlp_dims, dropout)
        
    
    def forward(self, **kwargs):
        inputs = kwargs['content']
        masks = kwargs['content_masks']
        category = kwargs['category']
        init_feature = self.bert(inputs, attention_mask = masks).last_hidden_state
        
        feature, _ = self.attention(init_feature, masks)
        idxs = torch.tensor([index for index in category]).view(-1, 1).cuda()
        domain_embedding = self.domain_embedder(idxs).squeeze(1)

        gate_value = self.gate(domain_embedding)

        shared_feature = 0
        for i in range(self.num_expert):
            tmp_feature = self.expert[i](init_feature)
            shared_feature += (tmp_feature * gate_value[:, i].unsqueeze(1))

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
                 dataset,
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
        self.dataset = dataset

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
        self.model = MultiDomainFENDModel(self.emb_dim, self.mlp_dims, len(self.category_dict), self.dropout, self.dataset)
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
                    os.path.join(self.save_param_dir, 'parameter_mdfend.pkl'))
                best_metric = results['metric']
            elif mark == 'esc':
                break
            else:
                continue
        self.model.load_state_dict(torch.load(os.path.join(self.save_param_dir, 'parameter_mdfend.pkl')))
        results = self.test(self.test_loader)
        if(logger):
            logger.info("start testing......")
            logger.info("test score: {}\n\n".format(results))
        print(results)
        return results, os.path.join(self.save_param_dir, 'parameter_mdfend.pkl')

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