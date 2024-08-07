import numpy as np
import torch
from torchvision.transforms import v2 


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
        images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
        labels[:, :, bbx1:bbx2, bby1:bby2] = labels[rand_index, :, bbx1:bbx2, bby1:bby2]
    return images,labels
    

def cutout(img,lbl, pad_size, replace,count=1):
    _, h, w = img.shape
    cutout_img = img.clone()
    cutout_lbl = lbl.clone()

    for _ in range(count):
        center_h, center_w = torch.randint(high=h, size=(1,)), torch.randint(high=w, size=(1,))
        low_h, high_h = torch.clamp(center_h-pad_size, 0, h).item(), torch.clamp(center_h+pad_size, 0, h).item()
        low_w, high_w = torch.clamp(center_w-pad_size, 0, w).item(), torch.clamp(center_w+pad_size, 0, w).item()

        cutout_img[:, low_h:high_h, low_w:high_w] = replace
        cutout_lbl[:, low_h:high_h, low_w:high_w] = replace

    return cutout_img,cutout_lbl


class Cutout(torch.nn.Module):

    def __init__(self, p, pad_size, replace=0):
        super().__init__()
        self.p = p
        self.pad_size = int(pad_size)
        self.replace = replace

    def forward(self, images,masks):
        B,Cim,H,W = images.shape[0],images.shape[1],images.shape[2],images.shape[3]
        B,Cmask,H,W = masks.shape[0],masks.shape[1],masks.shape[2],masks.shape[3]

        if torch.rand(1) < self.p:
            cutout_images = []
            lbls          = []
            for i in range(images.shape[0]):
                cutout_image,mask = cutout(images[i],masks[i], self.pad_size, self.replace)
                cutout_images.append(cutout_image)
                lbls.append(mask)
            images = torch.concat(cutout_images,dim=0).reshape(B,Cim,H,W)
            masks = torch.concat(lbls,dim=0).reshape(B,Cmask,H,W)
            return images,masks
        else:
            return images,masks
        
