import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_topological.nn import SignatureLoss
from torch_topological.nn import VietorisRipsComplex
from visualization import *


class Topological_Loss(torch.nn.Module):

    def __init__(self, lam=0.5, dimension=0):
        super().__init__()

        self.lam = lam
        self.loss = SignatureLoss(p=2)
        self.sigmoid_f    = nn.Sigmoid()
        # To do :Make dimensionality configurable
        self.vr = VietorisRipsComplex(dim=dimension)

    def forward(self, model_output,labels):
        predictions = self.sigmoid_f(torch.squeeze(model_output,dim=1))
        masks       = torch.squeeze(labels,dim=1)
        totalloss   = 0

        for i in range(predictions.shape[0]):

            #mask            = torch.tensor(masks[i] > 0.5,dtype=float)
            #prediction      = torch.tensor(predictions[i] > 0.5,dtype=float)
            #pi_mask_        = self.vr(mask)
            #pi_pred_        = self.vr(prediction)
            #topo_loss       = self.loss([prediction, pi_pred_], [mask, pi_mask_])
            
            pi_pred      = self.vr(predictions[i])
            pi_mask      = self.vr(masks[i])
            topo_loss    = self.loss([predictions[i], pi_pred], [masks[i], pi_mask])
            totalloss   +=topo_loss
        
        pr  = torch.flatten(predictions,start_dim=1,end_dim=2)
        ms  = torch.flatten(masks,start_dim=1,end_dim=2)
        pi_pr      = self.vr(pr)
        pi_ms      = self.vr(ms)
        t_l = self.loss([pr, pi_pr], [ms, pi_ms])
            #barcod(masks[i],pi_mask,predictions[i],pi_pred),barcod(mask,pi_mask_,prediction,pi_pred_)
        loss        = self.lam * totalloss/predictions.shape[0]
        return loss


class Dice_CE_Loss():
    def __init__(self):

#        self.batch,self.h,self.w,self.n_class = inputs.shape

        self.sigmoid_f     = nn.Sigmoid()
        self.softmax       = nn.Softmax(dim=-1)
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.bcewithlogic = nn.BCEWithLogitsLoss(reduction="mean")
    
    def Dice_Loss(self,input,target):

        smooth          = 1
        input           = self.sigmoid_f(torch.flatten(input=input))
        target          = torch.flatten(input=target)
        intersection    = (input * target).sum()
        dice_loss       = 1- (2.*intersection + smooth )/(input.sum() + target.sum() + smooth)
        return dice_loss

    def BCE_loss(self,input,target):
        input           = torch.flatten(input=input)
        target          = torch.flatten(input=target)
        sigmoid_f       = nn.Sigmoid()
        sigmoid_input   = sigmoid_f(input)
        #B_Cross_Entropy = F.binary_cross_entropy(sigmoid_input,target)
        entropy_with_logic = self.bcewithlogic(input,target)
        return entropy_with_logic

    def Dice_BCE_Loss(self,input,target):
        return self.Dice_Loss(input,target) + self.BCE_loss(input,target)

    # Manuel cross entropy loss 
    def softmax_manuel(self,input):
        return (torch.exp(input).t() / torch.sum(torch.exp(input),dim=1)).t()

    def CE_loss_manuel(self, input,target):

        last_dim = torch.tensor(input.shape[:-1])
        last_dim = torch.prod(last_dim)
        input    = input.reshape(last_dim,-1)      
        target   = target.view(last_dim,-1)     #    should be converted one hot previously

        return torch.mean(-torch.sum(torch.log(self.softmax_manuel(input)) * (target),dim=1))


    # CE loss 
    def CE_loss(self,input,target):
        cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        last_dim = torch.tensor(input.shape[:-1])
        last_dim = torch.prod(last_dim)
        input    = input.reshape(last_dim,-1)
        target   = target.reshape(last_dim).long         #  it will be converted one hot encode in nn.CrossEnt 

        return cross_entropy(input,target)
