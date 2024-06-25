import torch
import torch.nn as nn
import time
import numpy as np
from models.Metaformer import caformer_s18_in21ft1k



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

def device_f():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        
        self.conv_block=nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(out_c),
                                     nn.ReLU(),
                                     nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(out_c),
                                     nn.ReLU())
        
        self.pool = nn.MaxPool2d((2, 2))
       
    def forward(self, inputs):
        conv_block_out=self.conv_block(inputs)

        return conv_block_out
    

class skip_conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        
        self.conv_block=nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(out_c),
                                     nn.ReLU())
       
    def forward(self, inputs):
        conv_block_out=self.conv_block(inputs)

        return conv_block_out
     
class down(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))
        self.conv_pool=nn.Sequential(nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                                nn.BatchNorm2d(out_c),
                                nn.ReLU())

    def forward(self, inputs):
        conv_block_out=self.conv(inputs)
        encoder_output = self.pool(conv_block_out)
        

        return encoder_output,conv_block_out


class up(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.up   = nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size=3, stride=1,padding=1),  nn.BatchNorm2d(out_c), nn.ReLU())
        self.conv = conv_block(out_c+out_c, out_c)

    def forward(self, inputs, skip):
        conv_t_out = self.up(self.upsample(inputs))
        concat = torch.concat((conv_t_out,skip),dim=1)
        decoder_output = self.conv(concat)
        return decoder_output
    


class CA_CBA_UNET(nn.Module):
    def __init__(self,n_classes,config_res=None,training_mode=None,imnetpretrained=None):
        super().__init__()
        
        self.n_classes = n_classes

        self.config_res =config_res
        self.training_mode=training_mode
        self.bottleneck = conv_block(512, 512)

        size_dec=[512,320,128,64]
        size_dec_resnet=[2048,1024,512,256,128,64]

        self.decoder_blocks = nn.ModuleList([up(in_f, out_f) for in_f, out_f in zip(size_dec,size_dec[1:])])


        if not self.training_mode ==  "ssl_pretrained": 
            self.caformer        = caformer_s18_in21ft1k(config_res,training_mode,imnetpretrained)

        self.output_norms = nn.ModuleList([nn.LayerNorm(i) for i in size_dec[::-1]])

        self.CBA             = nn.ModuleList([BasicBlock(in_f, out_f) for in_f, out_f in zip(size_dec[::-1],size_dec[::-1])])  

        self.sep_conv_block = nn.Sequential(nn.Conv2d(64,64,3,padding='same'),nn.BatchNorm2d(64),nn.ReLU())
        self.up             = nn.Upsample(scale_factor=2, mode='nearest')
        self.convlast       = nn.Conv2d(64,1,kernel_size=1, stride=1,padding='same')

    def forward(self, inputs):                      # 1x  3 x 128 x 128
        
        _,out = self.caformer(inputs)               # [2, 64, 32, 32]) ([2, 128, 16, 16]) [2, 320, 8, 8]) ([2, 512, 4, 4])
                
        skip_connection=[]

        # SKİP CONNECTİONS
        skip_connections=[]
        for i in range (3):
            out_norm = self.output_norms[i](out[i].permute(0,2,3,1))
            skip_connections.append(self.CBA[i](out_norm.permute(0,3,1,2)))
        skip_connections.reverse()      

        b   = self.bottleneck(out[3])                              # 1x 512 x 4x4

        d1  = self.decoder_blocks[0](b, skip_connections[0])          # 1 x 320 x 8x8
        d2  = self.decoder_blocks[1](d1, skip_connections[1])          # 1 x 128 x 16x16
        d3  = self.decoder_blocks[2](d2, skip_connections[2])          # 1 x 64 x 32x32


        output = self.sep_conv_block(d3)                                       # 1   3 x 128x128
        output = self.up(output)
        output = self.sep_conv_block(output)
        output = self.up(output)
        output = self.convlast(output)

        return output


if __name__ == "__main__":

    start=time.time()

    x = torch.randn((2, 3, 256, 256))
    f = CA_CBA_UNET(1)
    y = f(x)
    print(x.shape)
    print(y.shape)

    end=time.time()
    
    print(f'spending time :  {end-start}')








