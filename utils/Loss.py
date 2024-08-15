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

    def __init__(self, lam=0.00001, dimension=1,point_threshould=5,radius=1,n_points_rate=8,loss_norm=2):
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
        totalloss = 0
        sobel_predictions = sobel_edge_detection(self.sigmoid_f(model_output))
        sobel_masks       = sobel_edge_detection(labels)
    
        predictions = torch.squeeze(sobel_predictions,dim=1)       
        masks       = torch.squeeze(sobel_masks,dim=1)

        for i in range(predictions.shape[0]):

            threshold = 0.5
            edges_pred = (predictions[i] > threshold)
            edges_mask = (masks[i] > threshold)

            # Extract the coordinates of edge points
            bins_pred = torch.nonzero(edges_pred, as_tuple=False)  # Shape [num_edges, 2]
            bins_mask = torch.nonzero(edges_mask, as_tuple=False)  # Shape [num_edges, 2]

            num_points = 300

            if bins_pred.shape[0]>num_points:
                point_p = bins_pred[torch.randperm(bins_pred.shape[0])[:num_points]]
            else:
                point_p = bins_pred

            if bins_mask.shape[0]>num_points:
                point_m = bins_mask[torch.randperm(bins_mask.shape[0])[:num_points]]
            else:
                point_m = bins_mask

            pi_pred      = self.vr(point_p.float())

            pi_mask      = self.vr(point_m.float())
            topo_loss    =  self.statloss(pi_mask,pi_pred)            
            totalloss   +=topo_loss

        loss        = self.lam * totalloss/predictions.shape[0]

        return loss


'''
            barcod(edges_mask,pi_mask,point_m,edges_pred,pi_pred,point_p,1)
            barcod(torch.tensor(bin_m),pi_mask,points_m,torch.tensor(bin_p),pi_pred,points_p,1)
            barcod(masks[i],pi_mask,points_m,predictions[i],pi_pred,points_p,1)
            points_m.shape
            points_p.shape
''' 


import torch
import matplotlib.pyplot as plt

# Assume `grayscale_image` is already loaded and Sobel edges are detected
# We will use the edge detection tensor `edges` from previous steps

def sobel_edge_detection(image):
    sobel_x = torch.tensor([[-1., 0., 1.], 
                            [-2., 0., 2.], 
                            [-1., 0., 1.]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    sobel_y = torch.tensor([[-1., -2., -1.], 
                            [ 0.,  0.,  0.], 
                            [ 1.,  2.,  1.]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    edge_x = torch.nn.functional.conv2d(image, sobel_x, padding=1)
    edge_y = torch.nn.functional.conv2d(image, sobel_y, padding=1)
    edge_magnitude = torch.sqrt(edge_x**2 + edge_y**2)
    return edge_magnitude



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
