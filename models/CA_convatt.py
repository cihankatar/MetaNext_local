import torch
import torch.nn as nn
#import time
from models.Metaformer import caformer_s18_in21ft1k
from models.convatt_decoder import conv_att
import torch.nn as nn
from SSL.simclr import SimCLR

def device_f():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


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
    
class CA_Convatt(nn.Module):
    def __init__(self,n_classes,config_res=None,training_mode=None,imnetpretrained=None):
        super().__init__()
        
        self.n_classes = n_classes
        self.config_res =config_res
        self.training_mode=training_mode
        self.bottleneck = conv_block(512, 512)
        size_dec=[512,320,128,64]
        size_dec_resnet=[2048,1024,512,256,128,64]
        
        if not self.training_mode ==  "ssl_pretrained": 
            self.caformer        = caformer_s18_in21ft1k(config_res,training_mode,imnetpretrained)

        self.output_norms = nn.ModuleList([nn.LayerNorm(i) for i in size_dec[::-1]])

        self.conv_att_dec = conv_att()

        self.sep_conv_block = nn.Sequential(nn.Conv2d(64,64,3,padding='same'),nn.BatchNorm2d(64),nn.ReLU())
        self.up             = nn.Upsample(scale_factor=2, mode='nearest')
        self.convlast       = nn.Conv2d(64,1,kernel_size=1, stride=1,padding='same')

    def forward(self, inputs):                      # 1x  3 x 128 x 128
        
        # ENCODER   
        if self.training_mode ==  "ssl_pretrained": 
            out = inputs
        else:
            _,out = self.caformer(inputs)               # [2, 64, 64, 64]) ([2, 128, 32, 32]) [2, 320, 16, 16]) ([2, 512, 8, 8])

        # SKİP CONNECTİONS
        skip_connections=[]
        for i in range (3):
            out_norm = self.output_norms[i](out[i].permute(0,2,3,1))
            skip_connections.append(out_norm.permute(0,3,1,2))
        skip_connections.reverse()          

        # BOTTLENECK
        b   = self.bottleneck(out[3])                              # 1x 512 x 8x8

        # DECODER
        out = self.conv_att_dec(b,skip_connections) 
        #trainable_params             = sum(p.numel() for p in self.convnextdecoder.parameters() if p.requires_grad)

        # LAST CONV
        output = self.sep_conv_block(out)
        output = self.up(output)
        output = self.convlast(output)

        return output


if __name__ == "__main__":

    #start=time.time()

    x = torch.randn((2, 3, 256, 256))
    #f = CA_CBA_Convnext(1)
    #y = f(x)
    #print(x.shape)
    #print(y.shape)

    #end=time.time()
    
    #print(f'spending time :  {end-start}')








