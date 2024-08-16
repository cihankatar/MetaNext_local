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
        self.mask = create_mask(border_width=10) 

        
    def forward(self, model_output,labels):

        totalloss = 0
        sobel_predictions   = sobel_edge_detection(model_output)
        sobel_masks         = sobel_edge_detection(labels)
        predictions         = torch.squeeze(sobel_predictions,dim=1)       
        masks               = torch.squeeze(sobel_masks,dim=1)

        for i in range(predictions.shape[0]):
            
            edges_pred = (predictions[i] > (torch.mean(predictions[i])+(torch.std(predictions[i]))))
            edges_pred = edges_pred*self.mask 
            edges_mask = (masks[i] > (torch.mean(masks[i])+torch.std(masks[i])))
            bins_pred = torch.nonzero(edges_pred, as_tuple=False)  # Shape [num_edges, 2]
            bins_mask = torch.nonzero(edges_mask, as_tuple=False)  # Shape [num_edges, 2]

            if bins_pred.shape[0] < 5:
                print("No predictions to get PH, threshould set to mean")
                edges_pred = (predictions[i] > torch.mean(predictions[i]))
                bins_pred = torch.nonzero(edges_pred, as_tuple=False)  # Shape [num_edges, 2]
            
            if bins_pred.shape[0] < 5:
                print("No masks to get PH, threshould set to mean")
                edges_mask = (masks[i] > torch.mean(masks[i]))
                bins_mask = torch.nonzero(edges_mask, as_tuple=False)  # Shape [num_edges, 2]

            num_points = 100
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
        loss.requires_grad=True
        return loss

def create_mask(border_width=10):
    mask = torch.ones(256,256)
    mask[:border_width, :] = 0
    mask[-border_width:, :] = 0
    mask[:, :border_width] = 0
    mask[:, -border_width:] = 0
    return mask

'''
            barcod(edges_mask,pi_mask,point_m,edges_pred,pi_pred,point_p,1)
            barcod(torch.tensor(bins_m),pi_mask,points_m,torch.tensor(bins_p),pi_pred,points_p,1)
            barcod(masks[i],pi_mask,points_m,predictions[i],pi_pred,points_p,1)

            
            plt.figure()
            plt.subplot(2,4,1)
            plt.title("model_out")
            plt.imshow(model_output[i][0].detach().numpy())
            plt.subplot(2,4,2)
            plt.title("sobel_predictions")
            plt.imshow(sobel_predictions[i][0].detach().numpy())
            plt.subplot(2,4,3)
            plt.title("bins_pred")
            plt.scatter(bins_pred[:,0],bins_pred[:,1],s=1)
            plt.subplot(2,4,4)
            plt.title("selected_points")
            plt.scatter(point_p[:,0],point_p[:,1],s=1)

            plt.figure()
            plt.subplot(2,4,1)
            plt.title("masks")
            plt.imshow(labels[i][0].detach().numpy())
            plt.subplot(2,4,2)
            plt.title("sobel_predictions")
            plt.imshow(sobel_masks[i][0].detach().numpy())
            plt.subplot(2,4,3)
            plt.title("bins_masks")
            plt.scatter(bins_mask[:,0],bins_mask[:,1],s=1)
            plt.subplot(2,4,4)
            plt.title("selected_points")
            plt.scatter(point_m[:,0],point_m[:,1],s=1)

            
''' 

def sobel_edge_detection(image):

    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sobel_x = torch.tensor([[-1., 0., 1.], 
                            [-2., 0., 2.], 
                            [-1., 0., 1.]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    sobel_y = torch.tensor([[-1., -2., -1.], 
                            [ 0.,  0.,  0.], 
                            [ 1.,  2.,  1.]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

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



def circular_lbp(img):
    """
    Compute Circular Local Binary Pattern (LBP) for a given image using PyTorch.
    
    Parameters:
    img (torch.Tensor): Input image as a PyTorch tensor of shape (1, H, W).
    
    Returns:
    torch.Tensor: Circular LBP of the image of shape (1, H, W).
    """
    h, w = img.shape[1], img.shape[2]
    
    # Define the 8 circular neighbors with offsets
    neighbors = [
        (-1, 0), (-1, 1), (0, 1), (1, 1),
        (1, 0), (1, -1), (0, -1), (-1, -1)
    ]
    
    # Prepare an empty tensor for LBP
    lbp_img = torch.zeros_like(img)
    
    # Compute LBP for each neighbor
    for i, (dy, dx) in enumerate(neighbors):
        # Use circular shifts
        neighbor = torch.roll(img, shifts=(dy, dx), dims=(1, 2))
        lbp_img += (neighbor >= img) * (1 << i)
    
    return lbp_img


'''
            lbf_out   = circular_lbp(labels[i]).squeeze()
            p_min = torch.min(lbf_out)
            p_max = torch.max(lbf_out)
            normalized_mask = 1-(lbf_out - p_min) / (p_max - p_min)     
            edges_mask = (normalized_mask > 0.5)
            bins_mask = torch.nonzero(edges_mask, as_tuple=False)  # Shape [num_edges, 2]

            lbf_out   = circular_lbp(model_output[i]).squeeze()
            p_min = torch.min(lbf_out)
            p_max = torch.max(lbf_out)
            normalized_pred = 1-(lbf_out - p_min) / (p_max - p_min)     
            edges_pred = (normalized_pred > 0.5)
            bins_pred = torch.nonzero(edges_pred, as_tuple=False)  # Shape [num_edges, 2]

            num_points = 200
            if bins_pred.shape[0]>num_points:
                point_p = bins_pred[torch.randperm(bins_pred.shape[0])[:num_points]]
            else:
                point_p = bins_pred

            if bins_mask.shape[0]>num_points:
                point_m = bins_mask[torch.randperm(bins_mask.shape[0])[:num_points]]
            else:
                point_m = bins_mask

'''
'''

            if torch.count_nonzero(bins_pred) < 50:
                print("bin_pred is empty. Numer of points calculated based on mean")
                print(edges_mask.shape)
                edges_pred = (predictions[i] > torch.mean(predictions[i]))
                bins_pred = torch.nonzero(edges_pred, as_tuple=False)  

            if torch.count_nonzero(bins_mask) < 50:
                print(bins_mask.unique,edges_mask)
                print("bin_mask is empty. Numer of points calculated based on mean :")
                edges_mask = (masks[i] > torch.mean(masks[i]))
                bins_mask = torch.nonzero(bins_pred, as_tuple=False)  # Shape [num_edges, 2]

            if bins_pred.shape[0]>num_points:
                interval   = int(bins_pred.shape[0]/100)
                selected_indices = bins_pred[::interval]
                point_p = torch.zeros_like(edges_mask)
                point_p[selected_indices[:, 0], selected_indices[:, 1]] = 1
                point_p = torch.nonzero(point_p, as_tuple=False) 

            if bins_mask.shape[0]>num_points:
                interval   = int(bins_mask.shape[0]/100)
                selected_indices = bins_mask[::interval]
                point_m = torch.zeros_like(edges_mask)
                point_m[selected_indices[:, 0], selected_indices[:, 1]] = 1
                point_m = torch.nonzero(point_m, as_tuple=False) 

                '''