import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


class EncoderRNN(nn.Module):
    """ The standard RNN encoder. """
    def __init__(self, input_size,
                hidden_size, num_layers=1, 
                dropout=0.1):
        super(EncoderRNN, self).__init__()

        hidden_size = hidden_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=True)
        
    def forward(self, input, lengths=None, hidden=None):
        """ See EncoderBase.forward() for description of args and returns."""

        emb = self.dropout(input)

        packed_emb = emb
        if lengths is not None:
            # Lengths data is wrapped inside a Variable.
            packed_emb = pack(emb, lengths)

        outputs, hidden_t = self.rnn(packed_emb, hidden)

        if lengths is not None:
            outputs = unpack(outputs)[0]


        hidden_t = torch.cat([hidden_t[0:hidden_t.size(0):2], hidden_t[1:hidden_t.size(0):2]], 2)

        return outputs, hidden_t
