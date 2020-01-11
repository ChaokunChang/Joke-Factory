import numpy as np
import sys
from copy import deepcopy

import torch.nn as nn
from torch.optim import Adam
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence

class JokeModel(nn.Module):
    def __init__(self,vocab_size,conf,device):
        super().__init__()
        self.num_layers = conf.num_layers
        self.hidden_dim = conf.hidden_dim
        self.device = device
        #word embedding layer
        self.embedding = nn.Embedding(vocab_size,conf.embedding_dim)
        # network structure:2 layer lstm
        self.lstm = nn.LSTM(conf.embedding_dim,self.hidden_dim,num_layers = conf.num_layers)
        # 全连接层，后接sofmax进行classification
        self.linear_out = nn.Linear(self.hidden_dim,vocab_size)

    def forward(self, input, hidden=None):
        seq_len,batch_size = input.size()
        # embeds_size = (seq_len*batch_size*embedding_dim)
        embeds = self.embedding(input)
        # print(input.shape())
        if hidden is None:
            h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device)
            c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device)
        else:
            h_0,c_0 = hidden
        output,hidden = self.lstm(embeds,(h_0,c_0))
        # output_size = (seq_len*batch_size*vocab_size)
        output = self.linear_out(output.view(seq_len*batch_size,-1))
        return output,hidden