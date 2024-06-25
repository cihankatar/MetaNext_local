import torch
import torch.nn as nn
import time
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

    def __init__(self, token_dim, dim=[32,16,8,4],num_heads=10, mlp_ratio=4 ):
        super().__init__() 

        self.token_dim      = token_dim
        self.num_heads      = num_heads
        self.layer_norm     = nn.LayerNorm(token_dim)
        
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
        x   = self.layer_norm(x.permute(0,2,3,1)).permute(0,3,1,2)   
        out = (x + self.att(self.dropout(self.sep_conv(x))))  # input dim b c h w ,  permute to use mlp layer.  
        out   = self.layer_norm(out.permute(0,2,3,1)).permute(0,3,1,2)   
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
                                     nn.ReLU(),
                                     nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(out_c),
                                     nn.ReLU())
        
        self.pool = nn.MaxPool2d((2, 2))
       
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
    


class CA_Former_Unet(nn.Module):
    def __init__(self,n_classes):
        super().__init__()
        
        self.n_classes = n_classes
        #sigmoid_f      = nn.Sigmoid()
        #softmax_f      = nn.Softmax()

        size_en=[3,64,128,320,512]
        self.encoder_blocks = nn.ModuleList([down(in_f, out_f) for in_f, out_f in zip(size_en,size_en[1:])])

        self.bottleneck = conv_block(512, 512)

        size_dec=[512,320,128,64]
        self.decoder_blocks = nn.ModuleList([up(in_f, out_f) for in_f, out_f in zip(size_dec,size_dec[1:])])

        self.caformer   = caformer_s18_in21ft1k()

        self.Convformer_att = nn.ModuleList([Convformer_Attention_Block(i) for i in (size_dec[:0:-1])])

        self.sep_conv_block = nn.Sequential(nn.Conv2d(64,64,3,padding='same'),nn.BatchNorm2d(64),nn.ReLU())
        self.up             = nn.Upsample(scale_factor=2, mode='nearest')
        self.convlast       = nn.Conv2d(64,1,kernel_size=1, stride=1,padding='same')

    def forward(self, inputs):                      # 1x  3 x 128 x 128
        
        _,out = self.caformer(inputs)               # [2, 64, 32, 32]) ([2, 128, 16, 16]) [2, 320, 8, 8]) ([2, 512, 4, 4])
                
        skip_connection=[]

        length = len(self.Convformer_att)
        for i in range (3):
            skip_connection.append(self.Convformer_att[i](out[i]))
        skip_connection.reverse()        

        b   = self.bottleneck(out[3])                              # 1x 512 x 4x4

        d1  = self.decoder_blocks[0](b, skip_connection[0])          # 1 x 320 x 8x8
        d2  = self.decoder_blocks[1](d1, skip_connection[1])          # 1 x 128 x 16x16
        d3  = self.decoder_blocks[2](d2, skip_connection[2])          # 1 x 64 x 32x32


        output = self.sep_conv_block(d3)                                       # 1   3 x 128x128
        output = self.up(output)
        output = self.sep_conv_block(output)
        output = self.up(output)
        output = self.convlast(output)

        return output


if __name__ == "__main__":

    start=time.time()

    x = torch.randn((2, 3, 256, 256))
    f = CA_Former_Unet(1)
    y = f(x)
    print(x.shape)
    print(y.shape)

    end=time.time()
    
    print(f'spending time :  {end-start}')








