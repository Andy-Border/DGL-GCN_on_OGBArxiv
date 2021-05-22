import os.path as osp
import sys

sys.path.append((osp.abspath(osp.dirname(__file__)).split('src')[0] + 'src'))

import torch.nn.functional as F
from GCN.model import GCN
from GCN.config import GCNConfig
import torch as th
import argparse
from utils.util_funcs import exp_init, time_logger
import dgl
from utils.trainer import FullBatchTrainer
from utils.conf_utils import *
from utils.data_utils import preprocess_data


@time_logger
def train_gcn(args):
    exp_init(args.seed, gpu_id=args.gpu)
    # ! config
    cf = GCNConfig(args)
    cf.device = th.device("cuda:0" if args.gpu >= 0 else "cpu")

    # ! Load Graph
    g, features, n_feat, cf.n_class, labels, train_x, val_x, test_x = preprocess_data(cf.dataset)
    features = features.to(cf.device)
    g = dgl.add_self_loop(g).to(cf.device)
    supervision = SimpleObject({'train_x': train_x, 'val_x': val_x, 'test_x': test_x, 'labels': labels})

    # ! Train Init
    print(f'{cf}\nStart training..')
    model = GCN(g, n_feat, cf.n_hidden, cf.n_class, cf.n_layer, F.relu, cf.dropout)
    model.to(cf.device)
    print(model)
    optimizer = th.optim.Adam(
        model.parameters(), lr=cf.lr, weight_decay=cf.weight_decay)


    # ! Train
    trainer = FullBatchTrainer(model=model, g=g, cf=cf, features=features,
                               sup=supervision, stopper=None, optimizer=optimizer,
                               loss_func=th.nn.CrossEntropyLoss())
    trainer.run()
    trainer.eval_and_save()

    return cf


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training settings")
    dataset = 'pubmed'
    dataset = 'citeseer'
    dataset = 'arxiv'
    dataset = 'cora'
    # ! Settings
    parser.add_argument("-g", "--gpu", default=0, type=int, help="GPU id to use.")
    parser.add_argument("-d", "--dataset", type=str, default=dataset)
    parser.add_argument("--seed", default=0)
    args = parser.parse_args()
    # ! Train
    cf = train_gcn(args)
