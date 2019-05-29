import torch
import torch.nn as nn
from constants import *


class Baseline(nn.Module):
  def __init__(self, hidden_size, rnn_cell='lstm', n_layers=1):
    super(Baseline, self).__init__()
    self.hidden_size = hidden_size
    self.n_layers = n_layers
    self.encoder = nn.Embedding(N_DICT+1, hidden_size)  # because of start token, we add +1 on N_DICT

    # TODO: Fill in below
    # Hint: define nn.LSTM / nn.GRU units, with hidden_size and n_layers.
    if rnn_cell == 'lstm':
      self.rnn = nn.LSTM(input_size= hidden_size, hidden_size=hidden_size, num_layers=n_layers)
    elif rnn_cell == 'gru':
      self.rnn = nn.GRU(input_size= hidden_size, hidden_size=hidden_size, num_layers= n_layers)

    # TODO: Fill in below
    # input of decoder should be output of rnn,
    # output of decoder should be number of classes
    self.decoder = nn.Linear(in_features= hidden_size, out_features= N_DICT+1)
    self.log_softmax = nn.LogSoftmax(dim=-1)

  def forward(self, x, hidden, temperature=1.0):
    encoded = self.encoder(x)  # shape of (Batch, N_DICT)
    # To match the RNN input form(step, Batch, Feature), add new axis on first dimension
    encoded = encoded.unsqueeze(0)

    # TODO: Fill in below
    # hint: use self.rnn you made. encoded input and hidden should be fed.
    output, hidden = self.rnn(encoded, hidden)
    output = output.squeeze(0)

    # TODO: Fill in below
    # connect output of rnn to decoder.
    # hint: use self.decoder
    output = self.decoder(output)

    # Optional: apply temperature
    # https://cs.stackexchange.com/questions/79241/what-is-temperature-in-lstm-and-neural-networks-generally
    pred = self.log_softmax(output)

    return pred, hidden

  def init_hidden(self, batch_size, random_init=False):
    if random_init:
      return torch.randn(self.n_layers, batch_size, self.hidden_size), \
             torch.randn(self.n_layers, batch_size, self.hidden_size)
    else:
      return torch.zeros(self.n_layers, batch_size, self.hidden_size),\
             torch.zeros(self.n_layers, batch_size, self.hidden_size)
