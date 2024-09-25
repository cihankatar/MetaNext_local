import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import random_split
import cv2

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def split(all_list, all_path, data_paths):
    for list, path, data_path in zip(all_list, all_path, data_paths):
        os.chdir(path)
        for idx in range(len(list)):
            img_dir = os.path.join(data_path,list[idx])
            if 'jpg' in img_dir:
                img =Image.open(img_dir)
                img = img.resize((256, 256))
                img=np.array(img,dtype=float)
                img = img[:, :, [2, 1, 0]]
                cv2.imwrite(list[idx], img)     

            if 'png' in img_dir:
                img = Image.open(img_dir)
                img = img.convert("RGB") 
                img = img.resize((256, 256))
                img = np.array(img,dtype=float)
                img = img[:, :, [2, 1, 0]]
                cv2.imwrite(list[idx], img)     


def data_path(main_path):
    im_path         = main_path + "/images"
    mask_path       = main_path + "/masks"
    data_path       = [im_path, mask_path,im_path, mask_path,im_path, mask_path]
    return data_path



def dir_list(data_paths):
    
    images_dir_list = sorted(os.listdir(data_paths[0]))
    mask_dir_list   = sorted(os.listdir(data_paths[1]))

    for im,msk in zip(images_dir_list,mask_dir_list):
    
        # Remove the value from list_2 if the "." is 
        # not present in value
        if ".jpg" not in im:
            images_dir_list.remove(im)
        if ".png" not in msk:
            mask_dir_list.remove(msk)

    split_ratio     = [8000,1000,1015]
#   split_ratio     = [int(len(images_dir_list)*0.7),int(len(images_dir_list)*0.2),int(len(images_dir_list)*0.1)]
    train_idx,test_idx,val_idx     = random_split(images_dir_list, split_ratio, generator=torch.Generator().manual_seed(42))
    train_masks,test_masks,val_masks = random_split(mask_dir_list, split_ratio, generator=torch.Generator().manual_seed(42))

    im_train_list    = [images_dir_list[i] for i in train_idx.indices]
    im_test_list     = [images_dir_list[i] for i in test_idx.indices]
    im_val_list      = [images_dir_list[i] for i in val_idx.indices]
    mask_val_list    = [mask_dir_list[i] for i in val_masks.indices]
    masks_train_list = [mask_dir_list[i] for i in train_masks.indices]
    masks_test_list  = [mask_dir_list[i] for i in test_masks.indices]
    all_list         = [im_train_list,masks_train_list,im_test_list,masks_test_list,im_val_list,mask_val_list]
    return all_list

def split_main():
    dataset="HAM10000"

    main_path   = os.environ["ML_DATA_ROOT"]+dataset
    
    im_train_path    = main_path + "/train/images"
    masks_train_path = main_path + "/train/masks"
    im_val_path      = main_path + "/val/images"
    masks_val_path   = main_path + "/val/masks"
    im_test_path     = main_path + "/test/images"
    masks_test_path  = main_path + "/test/masks"
    all_path         = [im_train_path,masks_train_path,im_test_path,masks_test_path,im_val_path,masks_val_path]


    for path in all_path:
        if not os.path.exists(path):
            create_dir(path)

    data_paths  = data_path(main_path)
    all_list    = dir_list(data_paths)

    split(all_list, all_path, data_paths)
    os.chdir(main_path)


def ssl_data_split(ssl_size):
    train_im_path   = os.environ["ML_DATA_ROOT"]+"isic_2018/train/images"
    mask_im_path   = os.environ["ML_DATA_ROOT"]+"isic_2018/train/masks"
    
    images_train_dir_list = sorted(os.listdir(train_im_path))
    masks_train_dir_list = sorted(os.listdir(mask_im_path))
    
    for im,msk in zip(images_train_dir_list,masks_train_dir_list):
    
        # Remove the value from list_2 if the "." is 
        # not present in value
        if ".jpg" not in im:
            images_train_dir_list.remove(im)
        if ".png" not in msk:
            masks_train_dir_list.remove(msk)

    train_ssl_size  = round(ssl_size*len(images_train_dir_list))
    train_rest      = round(len(images_train_dir_list)-train_ssl_size)
    # val_ssl_size  = round(0.1*len(images_train_dir_list)) 
    # test_ssl_size = round(0.2*len(images_train_dir_list))


    ssl_training_size = [train_ssl_size,train_rest]
    train_idx,_     = random_split(images_train_dir_list, ssl_training_size, generator=torch.Generator().manual_seed(42))
    mask_idx,_      = random_split(masks_train_dir_list, ssl_training_size, generator=torch.Generator().manual_seed(42))

    im_train_list     = [images_train_dir_list[i] for i in train_idx.indices]
    mask_train_list    = [masks_train_dir_list[i] for i in mask_idx.indices]

    with open("utils/splt_idx.txt", "w") as write_file:
        for idx,(image,mask) in enumerate(zip(images_train_dir_list,masks_train_dir_list)):
            write_file.write(str(idx)+ ": " +image + " "+mask+ "\n")
            
    with open("utils/splt_idx.txt", "r") as read_line:
        images_list = read_line.readlines()

    return im_train_list,mask_train_list
