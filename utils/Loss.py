import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_topological.nn import WassersteinDistance,CubicalComplex
from torch_topological.nn import VietorisRipsComplex
from visualization import *
#from skimage.feature import local_binary_pattern 
#import gudhi as gd
#from gudhi.wasserstein import wasserstein_distance


#import gudhi as gd

class Topological_Loss(torch.nn.Module):

    def __init__(self, lam=0.01, dimension=1,point_threshould=5,radius=1,n_points_rate=8,loss_norm=2):
        super().__init__()

        self.lam                = lam
        self.point_threshould   = point_threshould
        self.radius             = radius
        self.n_points_rate      = n_points_rate
        self.dimension          = dimension
        self.loss_norm          = loss_norm
        self.sigmoid_f          = nn.Sigmoid()
        self.vr                 = VietorisRipsComplex(dim=self.dimension)
        self.wloss              = WassersteinDistance(p=2)
        self.mask               = create_mask(border_width=5) 
        self.thresholds         = torch.linspace(0, 1, steps=11)  # 10 intervals
        self.cubicalcomplex     = CubicalComplex()

    def forward(self, model_output,labels):

        totalloss = 0
        model_sigmoid_o     = self.sigmoid_f(model_output)
        sobel_predictions   = model_sigmoid_o
        sobel_masks         = labels
        predictions         = torch.squeeze(sobel_predictions,dim=1)       
        masks               = torch.squeeze(sobel_masks,dim=1)
        for i in range(predictions.shape[0]):

            prediction  = (predictions[i] - predictions[i].min()) / (predictions[i].max() - predictions[i].min())
            predictions_q  = torch.round(prediction  * 10) / 10
            masks_q  = torch.round(masks[i]  * 10) / 10


            # mask        = (masks[i] - masks[i].min()) / (masks[i].max() - masks[i].min())
            # sharpness = 10.0  # Increase for sharper thresholding
            # mean_val = torch.mean(predictions[i])
            # std_val = torch.std(predictions[i])
            # threshold = mean_val + std_val
            # edges_pred = torch.sigmoid(sharpness * (predictions[i] - threshold))

            # edge_tensor = edges_pred.unsqueeze(0).unsqueeze(0)
            # # Step 2: Apply a Differentiable Edge Detection (Sobel Filter)
            # sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
            # sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])
            # sobel_x = sobel_x.view(1, 1, 3, 3)  # Reshape for convolution
            # sobel_y = sobel_y.view(1, 1, 3, 3)

            # grad_x = F.conv2d(edge_tensor, sobel_x, padding=1)
            # grad_y = F.conv2d(edge_tensor, sobel_y, padding=1)

            # # Step 3: Calculate Gradient Magnitude (Edge Strength)
            # grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
            # grad_magnitude = grad_magnitude.squeeze()

            # edge_coords = torch.nonzero(edges_pred)

            # # Step 5: Normalize Coordinates and Store as Tensors
            # x_coords = edge_coords[:, 1].float() / prediction.shape[1]
            # y_coords = edge_coords[:, 0].float() / prediction.shape[0]
            # normalized_coords = torch.stack([x_coords, y_coords], dim=1)

            # # Step 6: Compute Gradients with Respect to Coordinates
            # grad_x_at_edges = grad_x.squeeze()[edge_coords[:, 0], edge_coords[:, 1]]
            # grad_y_at_edges = grad_y.squeeze()[edge_coords[:, 0], edge_coords[:, 1]]

            # # Combine coordinates with gradients
            # result = torch.stack([x_coords, y_coords, grad_x_at_edges, grad_y_at_edges], dim=1)



            # predictions[i]=predictions[i]*self.mask 
            # edges_pred = (predictions[i] > torch.mean(predictions[i])+torch.std(predictions[i]))
            # edges_mask = (masks[i] > torch.mean(masks[i])+torch.std(masks[i]))

            # bins_pred = torch.nonzero(edges_pred, as_tuple=False)  # Shape [num_edges, 2]
            # bins_mask = torch.nonzero(edges_mask, as_tuple=False)  # Shape [num_edges, 2]

            # if bins_pred.shape[0] < 5:
            #     print("No predictions to get PH, threshould set to mean")
            #     edges_pred = (predictions[i] > torch.mean(predictions[i]))
            #     bins_pred = torch.nonzero(edges_pred, as_tuple=False)  # Shape [num_edges, 2]
            
            # if bins_pred.shape[0] < 5:
            #     print("No masks to get PH, threshould set to mean")
            #     edges_mask = (masks[i] > torch.mean(masks[i]))
            #     bins_mask = torch.nonzero(edges_mask, as_tuple=False)  # Shape [num_edges, 2]

            # # num_points = 100
            # # if bins_pred.shape[0]>num_points:
            # #     point_p = bins_pred[torch.randperm(bins_pred.shape[0])[:num_points]]
            # # else:
            # #     point_p = bins_pred
            # # if bins_mask.shape[0]>num_points:
            # #     point_m = bins_mask[torch.randperm(bins_mask.shape[0])[:num_points]]
            # # else:
            # #     point_m = bins_mask

            # num_points = 100
            # min_pred = bins_pred.min(dim=0).values
            # max_pred = bins_pred.max(dim=0).values
            # min_mask = bins_mask.min(dim=0).values
            # max_mask = bins_mask.max(dim=0).values
            # bounding_box_pred = torch.prod(max_pred - min_pred)
            # bounding_box_mask = torch.prod(max_mask - min_mask)
            # estimated_grid_pred = bounding_box_pred / num_points
            # estimated_grid_mask = bounding_box_mask / num_points
            # grid_size1 = torch.sqrt(estimated_grid_pred)
            # grid_size2 = torch.sqrt(estimated_grid_mask)

            # if bins_pred.shape[0]>num_points:
            #     grid_indices    = (bins_pred // grid_size1).int()
            #     unique_indices, inverse_indices = torch.unique(grid_indices, dim=0, return_inverse=True)
            #     point_p         = torch.zeros_like(unique_indices, dtype=torch.float32)
            #     counts          = torch.bincount(inverse_indices)
            #     counts          = counts.float()
            #     sums            = torch.zeros_like(point_p)
            #     sums.index_add_(0, inverse_indices, bins_pred.float())
            #     point_p = sums / counts.unsqueeze(1)
            # else:
            #     point_p = bins_pred
            
            # if bins_mask.shape[0]>num_points:

            #     grid_indices    = (bins_mask // grid_size2).int()
            #     unique_indices, inverse_indices = torch.unique(grid_indices, dim=0, return_inverse=True)
            #     point_m         = torch.zeros_like(unique_indices, dtype=torch.float32)
            #     counts          = torch.bincount(inverse_indices)
            #     counts          = counts.float()
            #     sums            = torch.zeros_like(point_m)
            #     sums.index_add_(0, inverse_indices, bins_mask.float())
            #     point_m = sums / counts.unsqueeze(1)

            # else:
            #     point_m = bins_mask
    
            
            pi_pred     = self.cubicalcomplex(predictions_q)
            pi_mask     = self.cubicalcomplex(masks_q)

            topo_loss   = self.wloss(pi_mask,pi_pred)             
            totalloss   +=topo_loss
        loss        = self.lam * totalloss/predictions.shape[0]
        return loss

def create_mask(border_width=5):
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mask = torch.ones(256,256)
    mask[:border_width, :] = 0
    mask[-border_width:, :] = 0
    mask[:, :border_width] = 0
    mask[:, -border_width:] = 0
    return mask.to(device)


'''
    gd.plot_persistence_diagram(diag1)
    
    peristent_diag(masks[i],masks[i],pi_mask[i],predictions[i],predictions[i],pi_pred[i],topo_loss)

    barcod(edges_mask,pi_mask,point_m,edges_pred,pi_pred,point_p,topo_loss)
    barcod(thresholded_mask,pi_mask_c,point_m,thresholded_pred,pi_pred_c,point_p,topo_loss_c)

    figures (model_output,sobel_predictions,bins_pred,point_p,labels,sobel_masks,bins_mask,point_m,i,topo_loss)
    barcod(labels[i][0],pi_mask,point_m,model_output[i][0],pi_pred,point_p,1,topo_loss) 
    barcod(edges_mask,pi_mask,point_m,edges_pred,pi_pred,point_p,topo_loss)

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