import torch
import torch.nn as nn
#import time
#from models.Metaformer import caformer_s18_in21ft1k
from models.enc import encoder_function
from models.dec import decoder_function
import torch.nn as nn
from SSL.simclr import SimCLR


def device_f():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


#####  GLAM Module  #####


class LocalAttention(nn.Module):

    def __init__(self, dim, drop=0.):
        super().__init__()
        

        self.conv = nn.Conv2d(dim,dim,kernel_size=3, padding='same')
        self.dwconv1 = nn.Conv2d(dim, dim, kernel_size=3, padding='same',groups=dim,dilation=2) # depthwise conv
        self.dwconv2 = nn.Conv2d(dim, dim, kernel_size=3, padding='same',groups=dim,dilation=3) # depthwise conv
        self.sigmoid = nn.Sigmoid()
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()

    def forward(self, x):
        
        x = self.conv(x)
        res=x

        x = self.dwconv1(x)
        x = self.dwconv2(x)
        x = self.sigmoid(x)
        x = x*res

        return x
    
class Attention(nn.Module):
    """
    Vanilla self-attention from Transformer: https://arxiv.org/abs/1706.03762.
    Modified from timm.
    """
    def __init__(self, dim, head_dim=8, num_heads=None, qkv_bias=False,
        attn_drop=0., proj_drop=0., proj_bias=False, **kwargs):
        super().__init__()

        self.head_dim = head_dim
        self.scale = head_dim ** -0.5   #embed_dim/head_number = head

        self.num_heads = num_heads if num_heads else dim // head_dim

        if self.num_heads == 0:
            self.num_heads = 1
        
        self.attention_dim = self.num_heads * self.head_dim

        self.qkv        = nn.Linear(dim, self.attention_dim * 3, bias=qkv_bias)
        self.attn_drop  = nn.Dropout(attn_drop)
        self.proj       = nn.Linear(self.attention_dim, dim, bias=proj_bias)
        self.proj_drop  = nn.Dropout(proj_drop)

        
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        N = H * W
        
        qkv = self.qkv(x).reshape( 3, B, H, W, C)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, self.attention_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x.permute(0, 3, 1, 2)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()

        self.att = Attention(inplanes)
        self.la = LocalAttention(inplanes)

    def forward(self, x):

        out =  self.la(x)+self.att(x)

        return out


#####   BOTTLENECK  #####
    
class Bottleneck(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        med_channels    = int(4 * in_c)

        self.dwconv1     = nn.Conv2d(in_c, in_c, kernel_size=3, padding="same", groups=in_c)
        self.pwconv1    = nn.Linear(in_c, med_channels)
        self.pwconv2    = nn.Linear(med_channels, out_c)
        self.norm       = nn.LayerNorm(out_c)    
        self.act        = nn.GELU()

        #self.attention = Attention(dim=512)
               
    def forward(self, inputs):  
        #x = inputs.permute(0, 2, 3, 1)
        #x   =   x + self.norm(self.attention(inputs.permute(0, 2, 3, 1)))
        x   =   self.dwconv1(inputs).permute(0, 2, 3, 1)
        x   =   self.norm(x)     
        x   =   self.pwconv1(x)
        x   =   self.act(x)
        convout = self.pwconv2(x).permute(0, 3, 1, 2)
        out     = convout+inputs

        return out
#####   MODEL #####
    
class model_topological_out(nn.Module):
    def __init__(self,n_classes,config_res=None,training_mode=None,imnetpretrained=None):
        super().__init__()
        
        self.n_classes = n_classes
        self.config_res =config_res
        self.training_mode=training_mode
        self.bottleneck = Bottleneck(512, 512)
        size_dec=[512,256,128,64]

        if not self.training_mode ==  "ssl_pretrained": 
            self.encoder      = encoder_function(config_res,training_mode,imnetpretrained)

        self.decoder           = decoder_function()
        
        #self.GLAM                = nn.ModuleList([BasicBlock(in_f, out_f) for in_f, out_f in zip(size_dec[::-1],size_dec[::-1])])  

        self.conv      = nn.Conv2d(64,64,3,padding='same',groups=64)
        self.pwconv1   = nn.Linear(64,64)
        self.norm      = nn.LayerNorm(64)
        self.act       = nn.GELU()
        self.pwconv2   = nn.Linear(64,1)

        self.convlast           = nn.Conv2d(64,1,kernel_size=1, stride=1,padding='same')

    def forward(self, inputs):                      # 1x  3 x 128 x 128
        
        # ENCODER   
        if self.training_mode ==  "ssl_pretrained": 
            out = inputs
        else:
            _,en_features = self.encoder(inputs)               # [2, 64, 64, 64]) ([2, 128, 32, 32]) [2, 320, 16, 16]) ([2, 512, 8, 8])


        # SKİP CONNECTİONS
        skip_connections=[]
        for i in range (3):
             skip_connections.append(en_features[i])
        skip_connections.reverse()   


        # SKİP CONNECTİONS
        # skip_connections=[]
        # for i in range (3):
        #    skip_connections.append(self.GLAM[i](out[i]))
        # skip_connections.reverse()      


        # BOTTLENECK
        b   = self.bottleneck(en_features[3])                              # 1x 512 x 8x8

        # DECODER
        out = self.decoder(b,skip_connections) 
        #trainable_params             = sum(p.numel() for p in self.convnextdecoder.parameters() if p.requires_grad)

        # LAST CONV

        out = self.conv(out)
        out = self.norm(out.permute(0, 2, 3, 1))
        out = self.act(out)

        out = self.pwconv1(out)
        out = self.norm(out)
        out = self.act(out)

        out = self.pwconv2(out)
        out=out.permute(0, 3, 1, 2)
        
        return en_features,out


if __name__ == "__main__":

    #start=time.time()

    x = torch.randn((2, 3, 256, 256))
    #f = CA_CBA_Convnext(1)
    #y = f(x)
    #print(x.shape)
    #print(y.shape)

    #end=time.time()
    
    #print(f'spending time :  {end-start}')








