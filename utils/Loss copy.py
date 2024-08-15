import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_topological.nn import SignatureLoss,SummaryStatisticLoss
from torch_topological.nn import VietorisRipsComplex
from visualization import *
from skimage.feature import local_binary_pattern 
import matplotlib.pyplot as plt

#import gudhi as gd

class Topological_Loss(torch.nn.Module):

    def __init__(self, lam=0.00003, dimension=1,point_threshould=5,radius=1,n_points_rate=8,loss_norm=2):
        super().__init__()

        self.lam                = lam
        self.point_threshould   = point_threshould
        self.radius             = radius
        self.n_points_rate      = n_points_rate
        self.dimension          = dimension
        self.loss_norm          = loss_norm

        self.loss               = SignatureLoss(p=self.loss_norm)
        self.sigmoid_f          = nn.Sigmoid()
        self.vr                 = VietorisRipsComplex(dim=self.dimension)
        self.statloss           = SummaryStatisticLoss()
        
    def forward(self, model_output,labels):

        predictions = self.sigmoid_f(torch.squeeze(model_output,dim=1))
        masks       = torch.squeeze(labels,dim=1)
        radius      = self.radius
        n_points    = self.n_points_rate * radius
        METHOD      = 'uniform' 
        totalloss   = 0
        
        for i in range(predictions.shape[0]):
            prediction = predictions[i].cpu().detach().numpy()
            mask       = masks[i].cpu().detach().numpy()

            prediction  = np.array(prediction>np.mean(prediction),dtype=int)
            bin_p       = local_binary_pattern(prediction, n_points, radius, METHOD)
            
            mask        = np.array(mask>np.mean(mask),dtype=int)
            bin_m       = local_binary_pattern(mask, n_points, radius, METHOD)

            points_p = np.array(np.column_stack(np.where(bin_p < self.point_threshould)),float)
            points_m = np.array(np.column_stack(np.where(bin_m < self.point_threshould)),float)
            #print(points_m.shape,points_p.shape)

            if points_p.shape[0]>(points_m.shape[0]*2):
                random_indices = np.random.choice(points_p.shape[0], points_m.shape[0]*2, replace=False)
                points_p = points_p[random_indices]
                points_p = torch.from_numpy(points_p)
                points_m = torch.from_numpy(points_m)

            pi_pred      = self.vr(points_p)
            pi_mask      = self.vr(points_m)
            topo_loss    =  self.statloss(pi_mask,pi_pred)
            totalloss   +=topo_loss
            
        loss        = self.lam * totalloss/predictions.shape[0]
        return loss


'''
            barcod(torch.tensor(bin_m),pi_mask,points_m,torch.tensor(bin_p),pi_pred,points_p,1)
            barcod(masks[i],pi_mask,points_m,predictions[i],pi_pred,points_p,1)
            points_m.shape
            points_p.shape
''' 


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
