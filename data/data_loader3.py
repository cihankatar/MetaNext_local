from torch.utils.data import DataLoader

from data.Custom_Dataset import dataset
from utils.Test_Train_Split import ssl_data_split
from glob import glob
from torchvision.transforms import v2 
import os
import numpy as np
import torch


def data_transform(mode,task,train,image_size):

    if mode == 'ssl':
        if 'simclr' in task:
            if train:

                transformations = v2.Compose([
                    v2.RandomResizedCrop([image_size,image_size],antialias=True),
                    #v2.RandomResizedCrop([image_size],antialias=True),
                    v2.RandomHorizontalFlip(p=0.5),
                    v2.ColorJitter(0.4, 0.4, 0.4, 0.1),
                    v2.RandomGrayscale(p=0.2),
                    #v2.ToTensor(),
                    #v2.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
                ])

            else:
                transformations = v2.Compose([
                    v2.ToTensor(),
                    v2.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
                ])
                
        transformations = SimCLRDataTransform(transformations)

        return transformations

    elif mode == 'supervised' or 'ssl_pretrained':    

        if train:

            transformations = v2.Compose([  v2.Resize([image_size,image_size],antialias=True),                                           
                                        v2.RandomHorizontalFlip(p=0.5),
                                        v2.RandomVerticalFlip(p=0.5),
                                        v2.RandomRotation(degrees=(0, 90)),
                                        #v2.RandomAdjustSharpness(sharpness_factor=10, p=0.),
                                        #v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
                                        #v2.RandomPerspective(distortion_scale=0.5, p=0.5),
                                        #v2.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.75, 0.75)),
                                        #v2.RandomPhotometricDistort(p=0.3),
                                        #v2.Normalize(mean=(0.400, 0.485, 0.456, 0.406), std=(0,222, 0.229, 0.224, 0.225))                              
                                    ])
        else:
            transformations = v2.Compose([  v2.Resize([image_size,image_size],antialias=True),
                            #v2.RandomHorizontalFlip(p=0.5),
                            #v2.RandomVerticalFlip(p=0.5),
                            #v2.RandomRotation(degrees=(0, 90)),
                            #v2.Normalize(mean=(0.400, 0.485, 0.456, 0.406), std=(0,222, 0.229, 0.224, 0.225)),                                
                            ])
            
        return transformations

def loader(mode,sslmode,train,batch_size,num_workers,image_size,cutout_pr,cutout_box,aug,shuffle,split_ratio,data):
    
    if data=='isic_1':
        foldernamepath="isic_2018_3/"
        imageext="/*.jpg"
        maskext="/*.png"
    elif data == 'kvasir_1':
        foldernamepath="kvasir_1/"
        imageext="/*.jpg"
        maskext="/*.jpg"
    elif data == 'ham_1':
        foldernamepath="HAM10000_1/"
        imageext="/*.jpg"
        maskext="/*.png"

    train_im_path   = os.environ["ML_DATA_ROOT"]+foldernamepath+"train/images"   
    train_mask_path = os.environ["ML_DATA_ROOT"]+foldernamepath+"train/masks"
    
    if train:
        test_im_path    = os.environ["ML_DATA_ROOT"]+foldernamepath+"val/images"
        test_mask_path  = os.environ["ML_DATA_ROOT"]+foldernamepath+"val/masks"
    else :
        test_im_path    = os.environ["ML_DATA_ROOT"]+foldernamepath+"test/images"
        test_mask_path  = os.environ["ML_DATA_ROOT"]+foldernamepath+"test/masks"

# if not os.path.exists(train_im_path):
    # split_main(
    
    if split_ratio is not None:
        if train:
            im_train_list,mask_train_list = ssl_data_split(split_ratio)
            train_im_path=[]
            train_mask_path=[]
            for im,mask in zip(im_train_list,mask_train_list):
                im_path = os.path.join(os.environ["ML_DATA_ROOT"]+"isic_2018/train/images", im)
                mask_path = os.path.join(os.environ["ML_DATA_ROOT"]+"isic_2018/train/masks", mask)
                train_im_path.append(im_path)
                train_mask_path.append(mask_path)
        print(f"training with {len(train_im_path)} images\n")



        test_im_path    = sorted(glob(test_im_path+imageext))
        test_mask_path  = sorted(glob(test_mask_path+maskext))
    
    else:
        train_im_path   = sorted(glob(train_im_path+imageext))
        train_mask_path = sorted(glob(train_mask_path+maskext))
        test_im_path    = sorted(glob(test_im_path+imageext))
        test_mask_path  = sorted(glob(test_mask_path+maskext))

    if aug:   
        transformations = data_transform(mode,sslmode,train,image_size)
    else:
        train=False
        transformations = data_transform(mode,sslmode,train,image_size)

    if torch.cuda.is_available():
        if train:
            data_train  = dataset(train_im_path,train_mask_path,cutout_pr,cutout_box, aug, transformations,mode)
        else:
            data_test   = dataset(test_im_path, test_mask_path,cutout_pr,cutout_box, aug, transformations,mode)

    elif train:  #train for debug in local
        data_train  = dataset(train_im_path,train_mask_path,cutout_pr,cutout_box, aug, transformations,mode)

    else:
        data_test   = dataset(test_im_path, test_mask_path,cutout_pr,cutout_box, aug, transformations,mode)

    if train:
        train_loader = DataLoader(
            dataset     = data_train,
            batch_size  = batch_size,
            shuffle     = shuffle,
            num_workers = num_workers,
            )
        return train_loader
    
    else :
        test_loader = DataLoader(
            dataset     = data_test,
            batch_size  = batch_size,
            shuffle     = shuffle,
            num_workers = num_workers,
        )
    
    return test_loader



class SimCLRDataTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
 
        xi = self.transform(sample)
        xj = self.transform(sample)
        return xi, xj

#loader()