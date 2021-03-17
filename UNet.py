import torch
import torch.nn as nn
import torchvision.transforms.functional as tv
from DoubleConv import DoubleConv

# implementation the architecture U-Net ref to https://arxiv.org/pdf/1505.04597.pdf
class UNet(nn.Module):

    def __init__(self,in_channels = 3,out_channels = 1):
        super(UNet,self).__init__()
        features = [64,128,256,512]
        
        self.downsampling = nn.ModuleList()
        self.upsampling   = nn.ModuleList()
        self.max_pool = nn.MaxPool2d(kernel_size=2,stride=2)

        for feature in features:
            self.downsampling.append(DoubleConv(in_channels,feature))
            in_channels = feature
            
        for feature in reversed(features):
            # feature * 2 beacause we add the skip connection in the entry
            self.upsampling.append(nn.ConvTranspose2d(feature * 2,feature,2,2))
            # same as downsampling, we need Double convolution the entry
            self.upsampling.append(DoubleConv(feature*2,feature))
        # the deep layer in the UNet
        self.bottleneck = DoubleConv(features[-1],features[-1] * 2)

        # last convolution for getting the semantic map
        self.output = nn.Conv2d(features[0],out_channels,kernel_size = 1)

    def forward(self,x):

        skip_informations = []

        # the downsampling forward
        for down_layer in self.downsampling:
            x = down_layer(x)
            # need to save the output layer for the upsampling
            skip_informations.append(x)
            # then we max pool
            x = self.max_pool(x)
        
        skip_informations.reverse()
        x = self.bottleneck(x)

        for index in range(0,len(self.upsampling),2):
            # first we conv transpose for up sampling
            x = self.upsampling[index](x)
            # if skip layer and current layer dosen't have the same shape, we reize the current layer
            if x.shape != skip_informations[index//2]:
                # resize the channel dimension
                tv.resize(x,size = skip_informations[index//2].shape[2:])
            # we concate the upsamling with the skipconnection
            x = torch.cat((skip_informations[index//2],x),dim=1)
            x = self.upsampling[index+1](x)

        return self.output(x)
    

def test():
    x = torch.randn(5,1,160,160)
    model = UNet(in_channels=1,out_channels=1)
    result = model(x)
    print(x.shape)
    print(result.shape)
    

if __name__ == "__main__":
    test()

            







