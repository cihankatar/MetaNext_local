import numpy as np
import torch
#from torchvision.transforms import v2 
import random
import cv2 



def circular_mix(images, masks, pr, alpha=1.0):
    """
    Apply circular mixing on a batch of images and their corresponding masks.

    Args:
        images (torch.Tensor): Batch of images of shape (batch_size, channels, height, width).
        masks (torch.Tensor): Batch of masks of shape (batch_size, 1, height, width).
        alpha (float): Hyperparameter for the Beta distribution.

    Returns:
        mixed_images (torch.Tensor): Batch of images after applying circular mixing.
        mixed_masks (torch.Tensor): Batch of masks after applying circular mixing.
    """
    if torch.rand(1) < pr:
        batch_size, _, H, W = images.size()
        device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Randomly select the indices of two images
        rand_index = torch.randperm(batch_size)

        # Generate the radius for the circular patch
        radius = np.random.uniform(100, 200)

        # Generate the center of the circle
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        # Create the circular mask
        Y, X = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        dist_from_center = torch.sqrt((X - cx)**2 + (Y - cy)**2)
        circular_mask = dist_from_center <= radius
        circular_mask = circular_mask.float().unsqueeze(0)

        # Smooth the edges of the circular mask
        smooth_mask = torch.clamp(1.0 - (dist_from_center - radius) / (0.1 * radius), 0, 1).to(device)
        smooth_mask = smooth_mask.unsqueeze(0).unsqueeze(0).to(device)

        # Apply the circular mix to the images and masks
        mixed_images = images * smooth_mask + images[rand_index] * (1 - smooth_mask)
        mixed_masks = masks * smooth_mask + masks[rand_index] * (1 - smooth_mask)

        return mixed_images.to(device), mixed_masks.to(device)
    
    else:
        return images, masks


def cutout(img,lbl, pad_size, replace,count=1):
    _, h, w = img.shape
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # cutout_img = img.clone()
    # cutout_lbl = lbl.clone()
    cutout_c_img = img.clone().cpu()
    cutout_c_lbl = lbl.clone().cpu()
    mask = np.ones((h, w), np.uint8)
    # Randomly select the center of the circle
    center_x = random.randint(0, h)
    center_y = random.randint(0, w)
    mask     = cv2.circle(mask, (center_x, center_y), (pad_size*4) // 2, 0, -1)
    
    for _ in range(count):
        # center_h, center_w = torch.randint(high=h, size=(1,)), torch.randint(high=w, size=(1,))
        # low_h, high_h = torch.clamp(center_h-pad_size, 0, h).item(), torch.clamp(center_h+pad_size, 0, h).item()
        # low_w, high_w = torch.clamp(center_w-pad_size, 0, w).item(), torch.clamp(center_w+pad_size, 0, w).item()
        cutout_c_img[:, :] = cutout_c_img[:, :] * mask
        cutout_c_lbl[:, :] = cutout_c_lbl[:, :] * mask
        # cutout_img[:, low_h:high_h, low_w:high_w] = replace
        # cutout_lbl[:, low_h:high_h, low_w:high_w] = replace
    return cutout_c_img,cutout_c_lbl#cutout_img,cutout_lbl


class Cutout(torch.nn.Module):

    def __init__(self, pad_size, replace=0):
        super().__init__()
        self.pad_size = int(pad_size)
        self.replace = replace
        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, images,masks,p):
        B,Cim,H,W = images.shape[0],images.shape[1],images.shape[2],images.shape[3]
        B,Cmask,H,W = masks.shape[0],masks.shape[1],masks.shape[2],masks.shape[3]

        if torch.rand(1) < p:
            # cutout_images = []
            # lbls          = []
            cutout_c_images = []
            lbls_c          = []

            for i in range(images.shape[0]):
                cutout_c_img,cutout_c_lbl = cutout(images[i],masks[i], self.pad_size, self.replace)  #cutout_image,mask, add for squarecutout
                # cutout_images.append(cutout_image)
                # lbls.append(mask)
                cutout_c_images.append(cutout_c_img)
                lbls_c.append(cutout_c_lbl)

            # images = torch.concat(cutout_images,dim=0).reshape(B,Cim,H,W)
            # masks = torch.concat(lbls,dim=0).reshape(B,Cmask,H,W)
            images_c = torch.concat(cutout_c_images,dim=0).reshape(B,Cim,H,W)
            masks_c = torch.concat(lbls_c,dim=0).reshape(B,Cmask,H,W)

            return images_c.to(self.device),masks_c.to(self.device) #,images,masks
        else:
            return images,masks #,images,masks
        


# def rand_bbox(size):
        
#     lam = np.random.beta(1., 1.)
#     W = size[2]
#     H = size[3]
#     cut_rat = np.sqrt(1. - lam)
#     cut_w = np.intc(W * cut_rat)
#     cut_h = np.intc(H * cut_rat)

#     # uniform
#     cx = np.random.randint(W)
#     cy = np.random.randint(H)

#     bbx1 = np.clip(cx - cut_w // 2, 0, W)
#     bby1 = np.clip(cy - cut_h // 2, 0, H)
#     bbx2 = np.clip(cx + cut_w // 2, 0, W)
#     bby2 = np.clip(cy + cut_h // 2, 0, H)
#     return bbx1, bby1, bbx2, bby2

# def cutmix(images,labels,pr):
#     if np.random.rand(1) < pr:
#         rand_index = torch.randperm(images.size()[0])
#         bbx1, bby1, bbx2, bby2 = rand_bbox(images.size())
#         # distx=bbx2-bbx1
#         # ditsy=bby2-bby1

#         images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
#         labels[:, :, bbx1:bbx2, bby1:bby2] = labels[rand_index, :, bbx1:bbx2, bby1:bby2]
    
#         # mask=torch.ones(distx, ditsy)
#         # rows, cols = mask.size()
#         # dashed_pattern_row = torch.arange(cols) % distx/4
#         # dashed_pattern_col = torch.arange(rows) % ditsy/4
#         # mask[0, :] = dashed_pattern_row
#         # mask[-1, :] = dashed_pattern_row
#         # mask[:, 0] = dashed_pattern_col
#         # mask[:, -1] = dashed_pattern_col

#         # images[:, :, bbx1:bbx2, bby1:bby2] = images[:, :, bbx1:bbx2, bby1:bby2]*mask
#         # labels[:, :, bbx1:bbx2, bby1:bby2] = labels[rand_index, :, bbx1:bbx2, bby1:bby2]*mask

#     return images,labels

