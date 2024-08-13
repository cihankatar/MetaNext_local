import numpy as np
import torch
#from torchvision.transforms import v2 
import random
import cv2 

def rand_bbox(size):
        
    lam = np.random.beta(1., 1.)
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.intc(W * cut_rat)
    cut_h = np.intc(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def cutmix(images,labels,pr):
    if np.random.rand(1) < pr:
        rand_index = torch.randperm(images.size()[0])
        bbx1, bby1, bbx2, bby2 = rand_bbox(images.size())
        # distx=bbx2-bbx1
        # ditsy=bby2-bby1

        images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
        labels[:, :, bbx1:bbx2, bby1:bby2] = labels[rand_index, :, bbx1:bbx2, bby1:bby2]
    
        # mask=torch.ones(distx, ditsy)
        # rows, cols = mask.size()
        # dashed_pattern_row = torch.arange(cols) % distx/4
        # dashed_pattern_col = torch.arange(rows) % ditsy/4
        # mask[0, :] = dashed_pattern_row
        # mask[-1, :] = dashed_pattern_row
        # mask[:, 0] = dashed_pattern_col
        # mask[:, -1] = dashed_pattern_col

        # images[:, :, bbx1:bbx2, bby1:bby2] = images[:, :, bbx1:bbx2, bby1:bby2]*mask
        # labels[:, :, bbx1:bbx2, bby1:bby2] = labels[rand_index, :, bbx1:bbx2, bby1:bby2]*mask

    return images,labels


# import torch
# import numpy as np

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

# def create_blend_mask(bbx1, bby1, bbx2, bby2, size, sigma=10):
#     # Create a blank mask with zeros
#     mask = torch.zeros(size[2], size[3], dtype=torch.float32)

#     # Create coordinate grids
#     x = torch.arange(size[2], dtype=torch.float32)
#     y = torch.arange(size[3], dtype=torch.float32)
#     X, Y = torch.meshgrid(x, y)

#     # Calculate distance from the center of the bounding box
#     dist_x = (X - (bbx1 + bbx2) / 2).abs()
#     dist_y = (Y - (bby1 + bby2) / 2).abs()

#     # Combine the distances to form a "rectangle" with soft edges
#     dist = torch.maximum(dist_x - (bbx2 - bbx1) / 2, dist_y - (bby2 - bby1) / 2)
#     dist = dist.clamp(min=0)  # Distances inside the box should be 0

#     # Create a mask with a smooth transition using a Gaussian-like function
#     mask = torch.exp(-0.5 * (dist / sigma)**2)

#     # Expand to match the number of channels
#     mask = mask.unsqueeze(0).expand(size[1], -1, -1)
#     return mask

# def cutmix(images, labels, pr, sigma=10):
#     if np.random.rand(1) < pr:
#         rand_index = torch.randperm(images.size(0))
#         bbx1, bby1, bbx2, bby2 = rand_bbox(images.size())

#         # Create the blending mask with soft transitions
#         blend_mask = create_blend_mask(bbx1, bby1, bbx2, bby2, images.size(), sigma).to(images.device)

#         # Apply the mask to the images and labels
#         images = images * (1 - blend_mask) + images[rand_index] * blend_mask
#         labels = labels * (1 - blend_mask[0]) + labels[rand_index] * blend_mask[0]

#     return images, labels


def cutout(img,lbl, pad_size, replace,count=1):
    _, h, w = img.shape
    # cutout_img = img.clone()
    # cutout_lbl = lbl.clone()
    cutout_c_img = img.clone()
    cutout_c_lbl = lbl.clone()

    mask = np.ones((h, w), np.uint8)
    # Randomly select the center of the circle
    center_x = random.randint(0, h)
    center_y = random.randint(0, w)
    mask     = cv2.circle(mask, (center_x, center_y), (pad_size*2) // 2, 0, -1)
    
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

            return images_c,masks_c #,images,masks
        else:
            return images,masks #,images,masks
        
