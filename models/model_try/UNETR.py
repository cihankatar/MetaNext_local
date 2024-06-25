import torch
import torch.nn as nn
import time
from models.ViT import ViT_c
from vit_pytorch import ViT
# from ViT_copy import ViT_M
import matplotlib.pyplot as plt 

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True) )


    def forward(self, x):
        return self.layers(x)
    

class DeconvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.deconv = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        return self.deconv(x)


class UNET_TR(nn.Module):
    def __init__(self,n_classes):
        super().__init__()
        
        self.n_classes = n_classes


        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.VIT1 = ViT_c(images_dim=128,
                         input_channel=3, 
                         token_dim=768, 
                         n_heads=4, 
                         mlp_layer_size=512, 
                         t_blocks=12, 
                         patch_size=8,
                         classification=False).to(device)

        # self.VIT2 = VİT_NN(images_dim=128,
        #                   input_channel=3, 
        #                   token_dim=768, 
        #                   n_heads=4, 
        #                   mlp_layer_size=512, 
        #                   t_blocks=12, 
        #                   patch_size=8,
        #                   classification=False).to(device)

        # self.VIT3 = ViT(image_size = 128,
        #                 patch_size = 8,
        #                 num_classes = 1,
        #                 dim = 768,
        #                 depth = 12,
        #                 heads = 4,
        #                 mlp_dim = 512,
        #                 dropout = 0,
        #                 emb_dropout = 0)

        # self.VIT4 = ViT_M(img_dim=128,
        #       in_channels=3,
        #       patch_dim=8,
        #       embedding_dim=768,
        #       block_num=12,
        #       head_num=4,
        #       mlp_dim=512)

        self.up_deconv=nn.ConvTranspose2d(64,64,kernel_size=2,stride=2,padding=0)
        ## Decoder 1
        
        self.d1 = DeconvBlock(768, 512)
        
        self.s1 = nn.Sequential(
            DeconvBlock(768, 512),
            ConvBlock(512, 512))
        
        self.c1 = nn.Sequential(
            ConvBlock(512+512, 512),
            ConvBlock(512, 512))
        

        ## Decoder 2
        self.d2 = DeconvBlock(512, 256)

        self.s2 = nn.Sequential(
            DeconvBlock(768, 256),
            ConvBlock(256, 256),
            DeconvBlock(256, 256),
            ConvBlock(256, 256))
        
        self.c2 = nn.Sequential(
            ConvBlock(256+256, 256),
            ConvBlock(256, 256))
        

        ## Decoder 3
        self.d3 = DeconvBlock(256, 128)
        self.s3 = nn.Sequential(
            DeconvBlock(768, 128),
            ConvBlock(128, 128),
            DeconvBlock(128, 128),
            ConvBlock(128, 128),
            DeconvBlock(128, 128),
            ConvBlock(128, 128))
        
        self.c3 = nn.Sequential(
            ConvBlock(128+128, 128),
            ConvBlock(128, 128))
        

        ## Decoder 4
        self.d4 = DeconvBlock(128, 64)
        
        self.s4 = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 64))
        
        self.c4 = nn.Sequential(
            ConvBlock(64+64, 64),
            ConvBlock(64, 64))
        

        if self.n_classes > 1:

            self.outputs  = nn.Conv2d(64, 2, kernel_size=1, padding=0)
            #self.outputs = softmax_f(self.outputs)
        else:
            self.outputs  = nn.Conv2d(64, 1, kernel_size=3, stride=2,padding=1)
            #self.outputs  = sigmoid_f(self.outputs)


    def forward(self, inputs):                      # 2x  3 x 128 x 128
        
        enc1,skip_connections = self.VIT1(inputs)   # 2x  768 x 16 x 16

        z3, z6, z9, z12 = skip_connections

        batch = inputs.shape[0]
        z0 = inputs.view(batch, 3, 128, 128)

        shape = (batch, 768, 16, 16)
        z3 = z3.view(shape)
        z6 = z6.view(shape)
        z9 = z9.view(shape)
        z12 = z12.view(shape)

        # plt.figure()
        # plt.subplot(2,5,1)    
        # plt.imshow(z0[1,0].detach().numpy(),cmap='gray')
        # plt.subplot(2,5,2)
        # plt.imshow(z3[1,112].detach().numpy(),cmap='gray')
        # plt.subplot(2,5,3)
        # plt.imshow(z6[1,313].detach().numpy(),cmap='gray')
        # plt.subplot(2,5,4)
        # plt.imshow(z9[1,710].detach().numpy(),cmap='gray')
        # plt.subplot(2,5,5)
        # plt.imshow(z12[1,514].detach().numpy(),cmap='gray')
        
        ## Decoder 1
        x = self.d1(z12)
        s = self.s1(z9)
        x = torch.cat([x, s], dim=1)
        x = self.c1(x)

        # plt.subplot(2,5,6)
        # plt.imshow(x[1,0].detach().numpy(),cmap='gray')
        
        ## Decoder 2
        x = self.d2(x)
        s = self.s2(z6)
        x = torch.cat([x, s], dim=1)
        x = self.c2(x)

        # plt.subplot(2,5,7)
        # plt.imshow(x[1,0].detach().numpy(),cmap='gray')
        
        ## Decoder 3
        x = self.d3(x)
        s = self.s3(z3)
        x = torch.cat([x, s], dim=1)
        x = self.c3(x)

        # plt.subplot(2,5,8)
        # plt.imshow(x[1,0].detach().numpy(),cmap='gray')
        
        ## Decoder 4
        x = self.d4(x)
        s = self.s4(z0)
        s = self.up_deconv(s)
        x = torch.cat([x, s], dim=1)
        x = self.c4(x)

        # plt.subplot(2,5,9)
        # plt.imshow(x[1,0].detach().numpy(),cmap='gray')

        output = self.outputs(x)

        # plt.subplot(2,5,10)
        # plt.imshow(output[1,0].detach().numpy(),cmap='gray')

        return output

if __name__ == "__main__":

    start=time.time()

    x = torch.randn((2, 3, 128, 128))
    f = UNET_TR(1)
    y = f(x)
    print(x.shape)
    print(y.shape)

    end=time.time()
    
    print(f'spending time :  {end-start}')
