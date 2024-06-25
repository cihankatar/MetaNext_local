import torch
import torch.nn as nn
import time
import pywt

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

        self.conv        = conv_block(in_c, out_c)
        
        self.conv_dwt_block=nn.Sequential(nn.Conv2d(out_c*4,out_c , kernel_size=3, padding=1),
                                          nn.BatchNorm2d(out_c),
                                          nn.ReLU())
        
        self.device=device_f()

    def forward(self, inputs):

        skip_con       = self.conv(inputs)
        
        dwt_input      = skip_con.detach().cpu()
        coef           = pywt.dwt2(dwt_input,'db1')
        cA, (cH,cV,cD) = coef
        
        cA = torch.tensor(cA,requires_grad=True).to(self.device)
        cH = torch.tensor(cH,requires_grad=True).to(self.device)
        cV = torch.tensor(cV,requires_grad=True).to(self.device)
        cD = torch.tensor(cD,requires_grad=True).to(self.device)
        
        dwt = torch.concat((cA,cH,cV,cD), dim=1)
        dwt_out = self.conv_dwt_block(dwt)

        return skip_con,dwt_out

        #dwt_out = dwt + skip_down
        #plt.imshow(dwt[1][0].detach().numpy(),cmap='gray')

class up_idwt(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        
        self.input_convdwt= nn.Sequential(nn.Conv2d(in_c, in_c*4, kernel_size=3, stride=1,padding=1),
                                nn.BatchNorm2d(in_c*4),
                                nn.ReLU(inplace=True))
        
        self.conv = nn.Sequential(nn.Conv2d(in_c, int(out_c), kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm2d(int(out_c)),
                                nn.ReLU(inplace=True))
        
        self.s_con= nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size=3, stride=1,padding=1),
                                nn.BatchNorm2d(out_c),
                                nn.ReLU(inplace=True))
        
        self.last_up = nn.Sequential(nn.Conv2d(out_c, out_c, kernel_size=3, stride=1,padding=1),
                                nn.BatchNorm2d(out_c),
                                nn.ReLU(inplace=True))

        self.device  = device_f()

    def forward(self, x, s=None):       #2x 512 x 8 x 8   2x 1024 x 16 x 16
        
        x=self.input_convdwt(x)
        b,c,w,h=x.shape

        cA = x[:,0:int(c/4),:,:]
        cH = x[:,int(c/4):int(c/2),:,:]
        cV = x[:,int(c/2):int(3*c/4),:,:]
        cD = x[:,int(3*c/4):int(c):,:]

        coef = cA.cpu().detach(),(cH.cpu().detach(),cV.cpu().detach(),cD.cpu().detach())

        idwt_out = pywt.idwt2(coef, 'db1')
        idwt_out = torch.tensor(idwt_out,requires_grad=True).to(self.device)

        if s is not None:
            s_out = self.s_con(idwt_out)
            concat = torch.concat((s,s_out),dim=1)
            decoder_output = self.conv(concat)
            
        else:
            decoder_output = self.last_up(idwt_out)

        return decoder_output
    

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
    


class UNET_wave(nn.Module):
    def __init__(self,n_classes):
        super().__init__()
        
        self.n_classes = n_classes
        #sigmoid_f      = nn.Sigmoid()
        #softmax_f      = nn.Softmax()

        size_en=[3,64,128,256,512]
        self.encoder_blocks = nn.ModuleList([down(in_f, out_f) for in_f, out_f in zip(size_en,size_en[1:])])

        self.b = conv_block(512, 1024)

        size_dec=[1024,512,256,128,64]
        self.decoder_blocks = nn.ModuleList([up(in_f, out_f) for in_f, out_f in zip(size_dec,size_dec[1:])])

        
        
        if self.n_classes > 1:

            self.outputs  = nn.Conv2d(64, 2, kernel_size=1, padding=0)
            #self.outputs = softmax_f(self.outputs)
        else:
            self.outputs  = nn.Conv2d(64, 1, kernel_size=1, padding=0)
            #self.outputs  = sigmoid_f(self.outputs)

    def forward(self, inputs):                      # 1x  3 x 128 x 128
        
        s1,p1  = self.encoder_blocks[0](inputs)     # 1x  64 x 64x64   ,  64  x 128x128
        s2,p2  = self.encoder_blocks[1](p1)         # 1x 128 x 32x32   ,  128 x 64x64
        s3,p3  = self.encoder_blocks[2](p2)         # 1x 256 x 16x16   ,  256 x 32x32
        s4,p4  = self.encoder_blocks[3](p3)         # 1x 512 x 8x8     ,  512 x 16x16

        b = self.b(p4)                              # 1x 1024 x 8x8

        d1 = self.decoder_blocks[0](b, s4)          # 1x 512 x  16x16
        d2 = self.decoder_blocks[1](d1, s3)         # 1x 256 x  32x32
        d3 = self.decoder_blocks[2](d2, s2)         # 1x 128 x  64x64
        d4 = self.decoder_blocks[3](d3, s1)         # 1x  64 x 128x128

        outputs = self.outputs(d4)                  # 1   64 x 128x128

        return outputs

if __name__ == "__main__":

    start=time.time()

    x = torch.randn((2, 3, 256, 256))
    f = UNET_wave(1)
    y = f(x)
    print(x.shape)
    print(y.shape)

    end=time.time()
    
    print(f'spending time :  {end-start}')