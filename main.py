import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='m3fend')#textcnn bigru bert eann eddfn mmoe mose dualemotion stylelstm mdfend
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--max_len', type=int, default=170)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--early_stop', type=int, default=3)
parser.add_argument('--dataset', default='ch')# en
parser.add_argument('--batchsize', type=int, default=64)
parser.add_argument('--seed', type=int, default=2021)
parser.add_argument('--gpu', default='0')
parser.add_argument('--emb_dim', type=int, default=768)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--save_log_dir', default= './logs')
parser.add_argument('--save_param_dir', default= './param_model')
parser.add_argument('--param_log_dir', default = './logs/param')
parser.add_argument('--semantic_num', type=int, default=7)
parser.add_argument('--emotion_num', type=int, default=7)
parser.add_argument('--style_num', type=int, default=2)
parser.add_argument('--lnn_dim', type=int, default=50)
parser.add_argument('--domain_num', type=int, default=3)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

from grid_search import Run
import torch
import numpy as np
import random

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

if args.dataset == 'en':
    root_path = './data/en/'
    category_dict = {
        "gossipcop": 0,
        "politifact": 1,
        "COVID": 2,
    }
elif args.dataset == 'ch':
    root_path = './data/ch/'
    if args.domain_num == 9:
        category_dict = {
            "科技": 0,
            "军事": 1,
            "教育考试": 2,
            "灾难事故": 3,
            "政治": 4,
            "医药健康": 5,
            "财经商业": 6,
            "文体娱乐": 7,
            "社会生活": 8,
        }
    elif args.domain_num == 6:
        category_dict = {
            "教育考试": 0,
            "灾难事故": 1,
            "医药健康": 2,
            "财经商业": 3,
            "文体娱乐": 4,
            "社会生活": 5,
        }
    elif args.domain_num == 3:
        category_dict = {
            "政治": 0,  #852
            "医药健康": 1,  #1000
            "文体娱乐": 2,  #1440
        }

print('lr: {}; model name: {}; batchsize: {}; epoch: {}; gpu: {}; domain_num: {}'.format(args.lr, args.model_name, args.batchsize, args.epoch, args.gpu, args.domain_num))


config = {
        'use_cuda': True,
        'batchsize': args.batchsize,
        'max_len': args.max_len,
        'early_stop': args.early_stop,
        'num_workers': args.num_workers,
        'root_path': root_path,
        'weight_decay': 5e-5,
        'category_dict': category_dict,
        'dataset': args.dataset,
        'model':
            {
            'mlp': {'dims': [384], 'dropout': 0.2}
            },
        'emb_dim': args.emb_dim,
        'lr': args.lr,
        'epoch': args.epoch,
        'model_name': args.model_name,
        'seed': args.seed,
        'semantic_num': args.semantic_num,
        'emotion_num': args.emotion_num,
        'style_num': args.style_num,
        'domain_num': args.domain_num,
        'lnn_dim': args.lnn_dim,#the number of cross-view representations
        'save_log_dir': args.save_log_dir,
        'save_param_dir': args.save_param_dir,
        'param_log_dir': args.param_log_dir
        }



if __name__ == '__main__':
    Run(config = config).main()
