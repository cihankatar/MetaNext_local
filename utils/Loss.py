import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_topological.nn import SignatureLoss
from torch_topological.nn import VietorisRipsComplex
from visualization import *
from skimage.feature import local_binary_pattern 
import matplotlib.pyplot as plt
#import gudhi as gd

class Topological_Loss(torch.nn.Module):

    def __init__(self, lam=0.1, dimension=1,point_threshould=5,radius=1,n_points_rate=8,loss_norm=2):
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

    def forward(self,images, model_output,labels):

        predictions = self.sigmoid_f(torch.squeeze(model_output,dim=1))
        masks       = torch.squeeze(labels,dim=1)
        radius      = self.radius
        n_points    = self.n_points_rate * radius
        METHOD      = 'uniform' 
        totalloss   = 0
        for i in range(predictions.shape[0]):

            prediction = torch.tensor(predictions[i]>0.5,dtype=float)
            mask        = torch.tensor(masks[i]>0.5,dtype=float)

            p          = local_binary_pattern(prediction.cpu().detach().numpy(), n_points, radius, METHOD)
            m          = local_binary_pattern(mask.cpu().detach().numpy(), n_points, radius, METHOD)

            points_p = np.array(np.column_stack(np.where(p < self.point_threshould)),float)
            points_m = np.array(np.column_stack(np.where(m < self.point_threshould)),float)

            if points_p.shape[0]>points_m.shape[0]:
                random_indices = np.random.choice(points_p.shape[0], points_m.shape[0], replace=False)
                points_p = points_p[random_indices]
                points_p = torch.from_numpy(points_p)
                points_m = torch.from_numpy(points_m)
            else:
                points_p = torch.from_numpy(points_p)
                points_m = torch.from_numpy(points_m)
                pad_size = (0,0,0,points_m.shape[0]-points_p.shape[0])
                points_p = F.pad(points_p,pad_size, "constant", 0)

            pi_pred      = self.vr(points_p)
            pi_mask      = self.vr(points_m)
            topo_loss    = self.loss([points_p, pi_pred], [points_m, pi_mask])
            totalloss   +=topo_loss

        loss        = self.lam * totalloss/predictions.shape[0]
        return loss


'''
            barcod(torch.tensor(m),pi_mask,torch.tensor(p),pi_pred,1)
            barcod(masks[i],pi_mask,predictions[i],pi_pred,1)
            plt.figure()
            plt.scatter(points_m[:,0],points_m[:,1],s=1)
            plt.figure()
            plt.scatter(points_p[:,0],points_p[:,1],s=1) 

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
