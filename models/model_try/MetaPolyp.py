import torch
import torch.nn as nn
import time
import pywt
from einops import rearrange, repeat
import numpy as np
from models.Metaformer import caformer_s18_in21ft1k

import separableconv.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, head_num=4):
        super().__init__()

        self.head_num = head_num
        self.scale = (dim // head_num) ** (1 / 2)

        self.qkv_layer = nn.Linear(dim, dim * 3, bias=False)
        self.projection = nn.Linear(dim, dim, bias=False)

    def forward(self, x):     # input dim b c h w

        b,c,h,w = x.shape
        n       = h*w
        x       = x.reshape(b, n, c) 

        qkv = self.qkv_layer(x)

        query, key, value = tuple(rearrange(qkv, 'b t (d k h ) -> k b h t d ', k=3, h=self.head_num))
        energy = torch.einsum("... i d , ... j d -> ... i j", query, key) * self.scale

        attention = torch.softmax(energy, dim=-1)

        x = torch.einsum("... i j , ... j d -> ... i d", attention, value)

        x = rearrange(x, "b h t d -> b t (h d)")
        x = self.projection(x)
        out = rearrange(x, 'b (h w) c-> b c h w', h=h)
        return out   # output dim b c h w


class Convformer_Attention_Block(nn.Module):

    def __init__(self, token_dim, num_heads=10, mlp_ratio=4 ):
        super().__init__() 

        self.token_dim      = token_dim
        self.num_heads      = num_heads
        self.layer_norm1    = nn.LayerNorm(token_dim)
        self.layer_norm2    = nn.LayerNorm(token_dim)
        
        self.sep_conv = nn.SeparableConv2d(token_dim,token_dim,3,1,1)
        self.att      = MultiHeadAttention (token_dim,head_num=4)

        self.mlp      = nn.Sequential(
                            nn.Linear(token_dim, token_dim*mlp_ratio),
                            nn.GELU(),
                            nn.Dropout(0.1),
                            nn.Linear(token_dim*mlp_ratio, token_dim),
                            nn.Dropout(0.1))
        
        self.dropout  = nn.Dropout(0.1)

    def forward(self, x):
        
        x = self.layer_norm1(x.permute(0,2,3,1)).permute(0,3,1,2)    
        out = (x + self.att(self.dropout(self.sep_conv(x))))  # input dim b c h w ,  permute to use mlp layer.  
        out = self.layer_norm2(out.permute(0,2,3,1)).permute(0,3,1,2)                           
        out = out + self.mlp(out.permute(0,2,3,1)).permute(0,3,1,2)            # output dim b c h w , 
        return out


def device_f():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        
        self.conv_block=nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(out_c),
                                     nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
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


class decodeblock(nn.Module):
    def __init__(self, in_c, out_c,scale=None):
        super().__init__()

        self.up   = nn.Upsample(scale_factor=scale, mode='nearest')
        self.conv1 = nn.Sequential(nn.Conv2d(in_c,out_c,kernel_size=3,stride=1, padding="same"),nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(in_c,out_c,kernel_size=1,stride=1,padding="same"),nn.ReLU())


        # self.skip1 = nn.Sequential(nn.Conv2d(out_c,out_c,kernel_size=3,stride=1,padding="same"),nn.ReLU())
        # self.skip2 = nn.Sequential(nn.Conv2d(out_c,out_c,kernel_size=1,stride=1,padding="same"),nn.ReLU())
        self.bn_act= nn.Sequential(nn.BatchNorm2d(out_c),
                                          nn.ReLU())

    def forward(self, inputs):


        x1 = self.conv1(inputs)
        x2 = self.conv2(inputs)
        merge = x1+x2
        up=self.up(merge)

        # s=self.skip1(merge)
        # s=self.skip1(s)
        # merge = merge+s

        out= self.bn_act(up)

        return out
    
class MetaPolyp(nn.Module):
    def __init__(self,n_classes):
        super().__init__()
        
        self.n_classes = n_classes
        #sigmoid_f      = nn.Sigmoid()
        #softmax_f      = nn.Softmax()

        size_en=[3,64,128,320,512]
        self.encoder_blocks = nn.ModuleList([down(in_f, out_f) for in_f, out_f in zip(size_en,size_en[1:])])

        self.bottleneck = conv_block(512, 512)

        size_dec=[512,320,128,64,3,3]

        self.caformer   = caformer_s18_in21ft1k()

        self.decode        = nn.ModuleList([decodeblock(in_f, out_f,scale=2) for in_f, out_f in zip(size_dec,size_dec)])
        #self.Convformer_att = nn.ModuleList([Convformer_Attention_Block(i) for i in (size_dec[:0:-1])])
        

        self.decode_4 = decodeblock(320, 64,scale=4)
        self.conv_relu_1 = nn.Sequential(nn.Conv2d(64,64,kernel_size=3,stride=1, padding="same"), nn.ReLU())

        self.decode_8 = decodeblock(128, 3,scale=4)
        self.conv_relu_2 = nn.Sequential(nn.Conv2d(3,3,kernel_size=3,stride=1, padding="same"), nn.ReLU())

        size_cnv=[512,320,128,64,3,3]
        self.conv_bn_act = nn.ModuleList([nn.Sequential(nn.Conv2d(in_f, out_f,kernel_size=3,stride=1, padding="same"),nn.BatchNorm2d(out_f), nn.ReLU()) for in_f, out_f in zip(size_cnv,size_cnv[1:])])
       
        if self.n_classes > 1:

            self.outputs  = nn.ConvTranspose2d(64, 2, kernel_size=1, padding=0)
            #self.outputs = softmax_f(self.outputs)
        else:
            self.outputs  = nn.Conv2d(3, 1, kernel_size=3,stride=1, padding='same')
            #self.outputs  = sigmoid_f(self.outputs)

    def forward(self, inputs):                      # 1x  3 x 128 x 128
        
        _,out = self.caformer(inputs)               # [2, 64, 32, 32]) ([2, 128, 16, 16]) [2, 320, 8, 8]) ([2, 512, 4, 4])
        
        out.reverse()
        x = self.bottleneck(out[0])                              # 1x 512 x4x4

        x  = self.decode[0](x)
        x  =  self.conv_bn_act[0](x)
        x_1 = self.decode_4(x)

        x  = x + out[1]
        x  = self.decode[1](x)        
        x  =  self.conv_bn_act[1](x)
        x_2= self.decode_8(x)

        x  = x + out[2]
        x  = self.decode[2](x)
        x  =  self.conv_bn_act[2](x)

        x_1 = self.conv_relu_1(x_1)
        x = x+x_1
        x = self.conv_relu_1(x)

        x  = x + out[3]
        x  = self.decode[3](x)
        x  =  self.conv_bn_act[3](x)
        x_2 = self.conv_relu_2(x_2)
        x = x+x_2
        x = self.conv_relu_2(x)



        x=self.conv_bn_act[-1](x)
        x=self.decode[-1](x)
        x=self.conv_bn_act[-1](x)
   
        x=self.outputs(x)
        return x


if __name__ == "__main__":

    start=time.time()

    x = torch.randn((2, 3, 256, 256))
    f = Unet_CAformer(1)
    y = f(x)
    print(x.shape)
    print(y.shape)

    end=time.time()
    
    print(f'spending time :  {end-start}')








