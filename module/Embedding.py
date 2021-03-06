import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, input_size, embedding_size, padding_idx=0):
        super(Embedding, self).__init__()
        self.padding_idx = padding_idx
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=padding_idx)
        #nn.init.uniform(self.embedding.weight.data, -0.1, 0.1)


    def forward(self, input_seqs):
        embedded = self.embedding(input_seqs)
        return embedded
