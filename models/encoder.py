import torch
import torch.nn as nn
from positional_encodings import PositionalEncodingPermute2D

class Encoder(nn.Module):
  def __init__(self,in_channels = 1):
    super(Encoder,self).__init__()
    self.conv1 = nn.Conv2d(in_channels = in_channels,out_channels = 64,kernel_size=(3,3),padding=(1,1),stride=(1,1))
    self.conv2 = nn.Conv2d(in_channels = 64,out_channels = 128 ,kernel_size=(3,3),padding=(1,1),stride=(1,1))
    self.conv3 = nn.Conv2d(in_channels = 128,out_channels = 256 ,kernel_size=(3,3),padding=(1,1),stride=(1,1))
    self.conv3_ = nn.Conv2d(in_channels = 256,out_channels = 256 ,kernel_size=(3,3),padding=(1,1),stride=(1,1))
    self.conv4 = nn.Conv2d(in_channels = 256,out_channels = 512 ,kernel_size=(3,3),padding=(1,1),stride=(1,1))
    self.max_pool = nn.MaxPool2d(kernel_size=(2,2),padding=(0,0),stride=(2,2))
    self.max_pool2 = nn.MaxPool2d(kernel_size=(2,1),padding=(0,0),stride=(2,1))
    self.batch_norm2 = nn.BatchNorm2d(num_features=512)
    self.batch_norm1 = nn.BatchNorm2d(num_features=256)

  def forward(self,x):

    x = self.conv1(x)
    x = self.max_pool(x)
    x = self.conv2(x)
    x = self.max_pool(x)
    x = self.conv3(x)
    x = self.batch_norm1(x) 
    x = self.conv3_(x)
    x = self.max_pool(x)
    x = self.conv4(x)
    x = self.batch_norm2(x)

    p_enc_2d = PositionalEncodingPermute2D(x.shape[1])
    penc_x = p_enc_2d(x)
    assert penc_x.size()==x.size()

    combined_x = penc_x+x
    combined_x = combined_x.reshape(x.shape[0],x.shape[2]*x.shape[3],1,x.shape[1])
    return combined_x

if __name__ == '__main__':
    encoder = Encoder(in_channels=1)
    x = torch.randn(64,1,28,28)
    # print(x.shape)
    x=encoder(x)
    print(x.shape)