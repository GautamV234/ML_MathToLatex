import torch.nn as nn
from utils.word_embedder import WordEmbedder

class Decoder(nn.Module):
  def __init__(self,in_channels = 1):
    super(Decoder,self).__init__()

    Embedder = WordEmbedder()
    word2idx, wordEmbeddings = Embedder.generate_embeddings()


    self.LSTM_stack  = nn.LSTM()
  def forward(self,x):
    pass