import torch
import torch.nn as nn

# this class make the double convolution
class DoubleConv(nn.Module):

    def __init__(self,in_channels,out_channels):
        super(DoubleConv,self).__init__()
        self.conv =  nn.Sequential(
            #first convolution
            nn.Conv2d(in_channels,out_channels,3,1,1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            #second convolution
            nn.Conv2d(out_channels,out_channels,3,1,1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True) 
        )
    
    def forward(self,input):
        return self.conv(input)
        