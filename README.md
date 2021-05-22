# DGL-GCN_on_OGBArxiv



A DGL-GCN implementation with batch norm added and settings copied from OGB-Arxiv.

## Environment

- Python>=3.8
- OGB
- Pytorch==1.8.0
- DGL==0.6.1

## Usage

```
python train.py -darxiv # For arxiv default split
python train.py -dcora # For cora default split
```
Model settings are stored in the src/GCN/config.py


## Model

The model settings are copied from OGB's official implementation (https://github.com/snap-stanford/ogb/tree/master/examples/nodeproppred/arxiv)

```
GCN(
  (layers): ModuleList(
    (0): GraphConv(in=128, out=256, normalization=both, activation=None)
    (1): GraphConv(in=256, out=256, normalization=both, activation=None)
    (2): GraphConv(in=256, out=40, normalization=both, activation=None)
  )
  (bns): ModuleList(
    (0): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
)
```



## Performance on OGB-Arxiv

The performance of last 5 epochs:

```
Epoch 0495 |Time 0.1874 | loss 1.1833 | TrainAcc 0.6556 | ValAcc 0.6097 | TestAcc 0.5356
Epoch 0496 |Time 0.1832 | loss 1.1836 | TrainAcc 0.6543 | ValAcc 0.6014 | TestAcc 0.5182
Epoch 0497 |Time 0.1857 | loss 1.1878 | TrainAcc 0.6530 | ValAcc 0.5972 | TestAcc 0.5080
Epoch 0498 |Time 0.1855 | loss 1.1815 | TrainAcc 0.6541 | ValAcc 0.6045 | TestAcc 0.5124
Epoch 0499 |Time 0.1822 | loss 1.1793 | TrainAcc 0.6536 | ValAcc 0.6008 | TestAcc 0.5100

Train seed0 finished
Results:{'test_acc': '0.5100', 'val_acc': '0.6008'}
Config: {'birth_time': '05_22-10_18_58', 'dataset': 'arxiv', 'dropout': 0.5, 'early_stop': -1, 'epochs': 500, 'exp_name': 'default', 'lr': 0.01, 'model': 'GCN', 'n_hidden': 256, 'n_layer': 2, 'weight_decay': 0.0005}
Finished running train_gcn at 05-22 10:20:32, running time = 1.56min.
```

As we can observe, the performance is significantly worse than the reported results of OGB: https://ogb.stanford.edu/docs/leader_nodeprop/

![image-20210522103520100](assets/image-20210522103520100.png)

## Performance on Cora

However, this implementation seems to be correct, since the performance on Cora dataset is normal:

The performance of last 5 epochs:

```
Epoch 0495 |Time 0.0540 | loss 0.0015 | TrainAcc 1.0000 | ValAcc 0.7120 | TestAcc 0.7590
Epoch 0496 |Time 0.0497 | loss 0.0009 | TrainAcc 1.0000 | ValAcc 0.7180 | TestAcc 0.7600
Epoch 0497 |Time 0.0923 | loss 0.0026 | TrainAcc 1.0000 | ValAcc 0.7200 | TestAcc 0.7640
Epoch 0498 |Time 0.0705 | loss 0.0014 | TrainAcc 1.0000 | ValAcc 0.7100 | TestAcc 0.7610
Epoch 0499 |Time 0.0498 | loss 0.0012 | TrainAcc 1.0000 | ValAcc 0.7120 | TestAcc 0.7630

Train seed0 finished
Results:{'test_acc': '0.7630', 'val_acc': '0.7120'}
Config: {'birth_time': '05_22-10_32_14', 'dataset': 'cora', 'dropout': 0.5, 'early_stop': -1, 'epochs': 500, 'exp_name': 'default', 'lr': 0.01, 'model': 'GCN', 'n_hidden': 256, 'n_layer': 2, 'weight_decay': 0.0005}
Finished running train_gcn at 05-22 10:32:54, running time = 40.15s.
```

The above results are obtained on the exact the same settings of ogbn-arxiv. When the n_layer is set as 1 (i.e. two message passing layers), with n_hidden set as 128, the early stopped accuracy is 0.8120 , which is quite close to the reported results in the original paper (0.8150).
