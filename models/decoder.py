import torch
import torch.nn as nn
from torch.nn import init
from word_embedder import WordEmbedder

class Decoder(nn.Module):
  def __init__(self,in_channels = 1):
    super(Decoder,self).__init__()
    enc_out_dim = (9,1,512)
    self.iter = 0
    inputLSTMsize = 512
    Embedder = WordEmbedder(vec_size = inputLSTMsize)

    self.word2idx, self.wordEmbeddings = Embedder.generate_embeddings()

    output_size = 10
    self.BiLSTM_stack  = nn.LSTM(inputLSTMsize, output_size, num_layers = 2, bidirectional = True)



    # Attention Mech
    INIT = 1e-2
    self.beta = nn.Parameter(torch.Tensor(enc_out_dim))
    init.uniform_(self.beta, -INIT, INIT)
    self.W_1 = nn.Linear(enc_out_dim, enc_out_dim, bias=False)
    self.W_2 = nn.Linear(dec_rnn_h, enc_out_dim, bias=False)

    self.W_3 = nn.Linear(dec_rnn_h+enc_out_dim, dec_rnn_h, bias=False)
    self.W_out = nn.Linear(dec_rnn_h, out_size, bias=False)

    self.add_pos_feat = add_pos_feat
    self.dropout = nn.Dropout(p=dropout)
    self.uniform = Uniform(0, 1)

  # def upd_Hidden_Cell_state(self,):
  #   pass
  def forward(self,x):


    # Still not clear
    h0 = torch.randn(4, 1, 10)
    c0 = torch.randn(4, 1, 10)

    k = self.BiLSTM_stack(x,(h0,c0))


    pass

if __name__ == '__main__':
  inp = torch.randn(9, 1, 512)
  dec = Decoder()
  f = dec(inp)
