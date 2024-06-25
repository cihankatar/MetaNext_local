import torch
import torch.nn as nn
import time
from einops import rearrange, repeat
import numpy as np
from models.Metaformer import caformer_s18_in21ft1k

import separableconv.nn as nn


def device_f():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

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


class downsample(nn.Module):
    def __init__(self,in_c,out_c):
        super().__init__()

        self.down=nn.Sequential(nn.BatchNorm2d(in_c),
                                nn.Conv2d(in_c, out_c, kernel_size=7, stride=4, padding=2),
                                nn.BatchNorm2d(out_c),
                                nn.ReLU())
    def forward(self,input):
        out  = self.down(input)
        return out
            

class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        
        self.conv_block=nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size=3, padding='same'),
                                     nn.BatchNorm2d(out_c),
                                     nn.ReLU(),
                                     nn.Conv2d(out_c, out_c, kernel_size=3, padding='same'),
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
        self.conv_pool=nn.Sequential(nn.Conv2d(out_c, out_c, kernel_size=3, padding='same'),
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
    

class UNET_Convformer(nn.Module):
    def __init__(self,n_classes):
        super().__init__()
        
        self.n_classes = n_classes
        #sigmoid_f      = nn.Sigmoid()
        #softmax_f      = nn.Softmax()

        size_en=[64,128,320,512]
        self.downsampling = downsample(3,64)
        self.encoder_blocks = nn.ModuleList([down(in_f, out_f) for in_f, out_f in zip(size_en,size_en[1:])])

        self.bottleneck = conv_block(512, 1024)

        size_dec=[1024,512,320,128,64]
        self.decoder_blocks = nn.ModuleList([up(in_f, out_f) for in_f, out_f in zip(size_dec,size_dec[1:])])

        self.caformer   = caformer_s18_in21ft1k()
        
        size_dec.reverse()
        self.Convformer_att = nn.ModuleList([Convformer_Attention_Block(i) for i in (size_dec[1:])])

        self.sep_conv_block1 = nn.Sequential(nn.Conv2d(128,64,3,padding='same'),nn.BatchNorm2d(64),nn.ReLU())
        self.sep_conv_block2 = nn.Sequential(nn.Conv2d(64,64,3,padding='same'),nn.BatchNorm2d(64),nn.ReLU())
        self.up             = nn.Upsample(scale_factor=2, mode='nearest')
        self.convlast       = nn.Conv2d(64,1,kernel_size=1, stride=1,padding='same')

    def forward(self, inputs):                      # 1x  3 x 128 x 128
        inputs = self.downsampling(inputs)
        s1,p1  = self.encoder_blocks[0](inputs)     # 1x  64 x 64x64   ,  64  x 128x128
        s2,p2  = self.encoder_blocks[1](s1)         # 1x 128 x 32x32   ,  128 x 64x64
        s3,p3  = self.encoder_blocks[2](s2)         # 1x 320 x 16x16   ,  320 x 32x32

        b = self.bottleneck(s3)                     # 1x 1024 x 8x8

        s_in=[p1,p2,p3]
        s_out=[]

        length = len(self.Convformer_att)
        for i in range (3):
            s_out.append(self.Convformer_att[i](s_in[i]))

        d1 = self.decoder_blocks[0](b, s_out[2])          # 1x 512 x  16x16
        d2 = self.decoder_blocks[1](d1, s_out[1])         # 1x 320 x  32x32
        d3 = self.decoder_blocks[2](d2, s_out[0])         # 1x 128 x  64x64

        output = self.sep_conv_block1(d3)                                       # 1   3 x 128x128
        output = self.up(output)
        output = self.sep_conv_block2(output)
        output = self.up(output)
        output = self.convlast(output)

        return output


if __name__ == "__main__":

    start=time.time()

    x = torch.randn((2, 3, 256, 256))
    f = UNET_Convformer(1)
    y = f(x)
    print(x.shape)
    print(y.shape)

    end=time.time()
    
    print(f'spending time :  {end-start}')








