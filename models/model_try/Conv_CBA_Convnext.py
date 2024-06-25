import torch
import torch.nn as nn
import time
from einops import rearrange, repeat
import numpy as np
from models.Metaformer import convformer_s18_in21ft1k
from models.Convnext import convnext_pico
import torch.nn as nn
import torch.nn.functional as F



def device_f():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

#####   CBA Module  #####

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


#####   BOTTLENECK  #####
    
class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        
        self.conv_block=nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(out_c),
                                     nn.ReLU(),
                                     nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(out_c),
                                     nn.ReLU())
               
    def forward(self, inputs):
        conv_block_out=self.conv_block(inputs)

        return conv_block_out
    
#####   MODEL #####
    
class Conv_CBA_Convnext(nn.Module):
    def __init__(self,n_classes):
        super().__init__()
        
        self.n_classes = n_classes
        #sigmoid_f      = nn.Sigmoid()
        #softmax_f      = nn.Softmax()


        self.bottleneck = conv_block(512, 512)

        size_dec=[512,320,128,64]

        self.convformer = convformer_s18_in21ft1k()
        self.convnextdecoder = convnext_pico()

        self.CBA            = nn.ModuleList([BasicBlock(in_f, out_f) for in_f, out_f in zip(size_dec[::-1],size_dec[::-1])])  

        #a[start:stop:step] # start through not past stop, by step

        self.sep_conv_block = nn.Sequential(nn.Conv2d(64,64,3,padding='same'),nn.BatchNorm2d(64),nn.ReLU())
        self.up             = nn.Upsample(scale_factor=2, mode='nearest')
        self.convlast       = nn.Conv2d(64,1,kernel_size=1, stride=1,padding='same')

    def forward(self, inputs):                      # 1x  3 x 128 x 128

        # ENCODER        
        _,out = self.convformer(inputs)               # [2, 64, 32, 32]) ([2, 128, 16, 16]) [2, 320, 8, 8]) ([2, 512, 4, 4])
                
        # SKİP CONNECTİONS
        skip_connections=[]
        for i in range (3):
            skip_connections.append(self.CBA[i](out[i]))
        skip_connections.reverse()      

        # BOTTLENECK
        b   = self.bottleneck(out[3])                              # 1x 512 x 8x8

        # DECODER
        out = self.convnextdecoder(b,skip_connections) 

        # LAST CONV
        output = self.sep_conv_block(out)                           
        output = self.up(output)
        output = self.sep_conv_block(output)
        output = self.up(output)
        output = self.convlast(output)

        return output


if __name__ == "__main__":

    start=time.time()

    x = torch.randn((2, 3, 256, 256))
    f = Conv_CBA_Convnext(1)
    y = f(x)
    print(x.shape)
    print(y.shape)

    end=time.time()
    
    print(f'spending time :  {end-start}')








