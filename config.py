import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
early_stop = 5
epoch_num = 100

batch_size = 128
lr = 0.0005

n_layers = 3
dropout = 0.1
n_heads = 4
d_model = 256
pf_dim = 512
max_len = 50

unk = '<unk>'
pad = '<pad>'
bos = '<bos>'
eos = '<eos>'
UNK_IDX = 0
PAD_IDX = 1
BOS_IDX = 2
EOS_IDX = 3
