#!/usr/bin/env python
# encoding: utf-8
# File Name: data_util.py
# Author: Jiezhong Qiu
# Create Time: 2019/12/30 14:20

from ogb.nodeproppred import DglNodePropPredDataset
import dgl
import dgl.data
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch as th
import torch.nn.functional as F
from dgl import DGLGraph
from sklearn.model_selection import train_test_split

import utils.util_funcs as uf
from utils.proj_settings import *
from tqdm import tqdm
from heapq import heapify, heappushpop, merge as heap_merge, nlargest, nsmallest
import pickle

def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def preprocess_data(dataset):
    # Modified from AAAI21 FA-GCN
    if dataset in ['cora', 'citeseer', 'pubmed']:

        edge = np.loadtxt(f'{DATA_PATH}/{dataset}/{dataset}.edge', dtype=int).tolist()
        features = np.loadtxt(f'{DATA_PATH}/{dataset}/{dataset}.feature')
        labels = np.loadtxt(f'{DATA_PATH}/{dataset}/{dataset}.label', dtype=int)

        train = np.loadtxt(f'{DATA_PATH}/{dataset}/{dataset}.train', dtype=int)
        val = np.loadtxt(f'{DATA_PATH}/{dataset}/{dataset}.val', dtype=int)
        test = np.loadtxt(f'{DATA_PATH}/{dataset}/{dataset}.test', dtype=int)

        nclass = len(set(labels.tolist()))
        print(dataset, nclass)

        U = [e[0] for e in edge]
        V = [e[1] for e in edge]
        g = dgl.graph((U, V))
        g = dgl.to_simple(g)
        g = dgl.remove_self_loop(g)
        g = dgl.to_bidirected(g)

        features = normalize_features(features)
        features = th.FloatTensor(features)
        labels = th.LongTensor(labels)
        train = th.LongTensor(train)
        val = th.LongTensor(val)
        test = th.LongTensor(test)

    elif dataset in ['arxiv']:
        dataset = DglNodePropPredDataset(name='ogbn-arxiv', root=f'{DATA_PATH}ogb_arxiv')
        split_idx = dataset.get_idx_split()
        train, val, test = split_idx["train"], split_idx["valid"], split_idx["test"]
        g, labels = dataset[0]
        features = g.ndata['feat']
        nclass = 40
        # labels = labels.squeeze()
    if dataset in ['citeseer']:
        g = dgl.add_self_loop(g)
    return g, features, features.shape[1], nclass, labels, train, val, test