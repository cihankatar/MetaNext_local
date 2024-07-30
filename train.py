
import torch
import os
from torch.optim import Adam
import argparse
import matplotlib.pyplot as plt

import yaml
import wandb
from wandb_init  import wandb_init, parser_init
from visualization  import *

import time as timer

from tqdm import tqdm, trange
from utils.one_hot_encode import one_hot,label_encode

from data.data_loader import loader
from utils.Loss import Dice_CE_Loss
from augmentation.Augmentation import cutmix
"""
from models.CA_CBA_CA import CA_CBA_CA
from models.CA_CA import CA_CA
from models.CA_CBA_mnext import CA_CBA_MNEXT_B
from models.CA_CBA_Convnext import CA_CBA_Convnext
from models.CA_Convnext import CA_Convnext
from models.CA_CBA_Unet import CA_CBA_UNET
from models.CA_Unet import CA_UNET
from models.Unet import UNET
"""

from models.CA_CBA_Proposed import CA_CBA_Proposed
from models.Unet import UNET
from models.CA_Proposed import CA_Proposed
from models.Model import model_base
from SSL.simclr import SimCLR
from models.Metaformer import caformer_s18_in21ft1k
from models.resnet import resnet_v1


def load_config(config_name):
    with open(config_name) as file:
        config = yaml.safe_load(file)
    return config

def using_device():
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "") 
    return device

def main():

    data='kvasir_1'
    training_mode="supervised"
    train=True

    if data=='isic_1':
        foldernamepath="isic_1/"
    elif data == 'kvasir_1':
        foldernamepath="kvasir_1/"

    WANDB_DIR      = os.environ["WANDB_DIR"]
    WANDB_API_KEY  = os.environ["WANDB_API_KEY"]

    if torch.cuda.is_available():
        ML_DATA_OUTPUT = os.environ["ML_DATA_OUTPUT"]+foldernamepath
    else:
        ML_DATA_OUTPUT = os.environ["ML_DATA_OUTPUT_LOCAL"]+foldernamepath


    args,res,config_res   = parser_init("segmetnation task","training",training_mode,train)
    res             = ', '.join(res)
    config_res      = ', '.join(config_res)

    config         = wandb_init(WANDB_API_KEY,WANDB_DIR,args,config_res,data)
    #config        = load_config("config.yaml")

    device          = using_device()
    train_loader    = loader(
                            args.mode,
                            args.sslmode_modelname,
                            args.train,
                            args.bsize,
                            args.workers,
                            args.imsize,
                            args.cutoutpr,
                            args.cutoutbox,
                            args.aug,
                            args.shuffle,
                            args.sratio,
                            data )
    
    args.aug = False #cutout aug - false for val set
    val_loader    = loader(
                            args.mode,
                            args.sslmode_modelname,
                            args.train,
                            args.bsize,
                            args.workers,
                            args.imsize,
                            args.cutoutpr,
                            args.cutoutbox,
                            args.aug,
                            args.shuffle,
                            args.sratio,
                            data )
    

    if args.mode == "ssl_pretrained" or args.mode == "supervised":
        model                       = model_base(config['n_classes'],config_res,args.mode,args.imnetpr).to(device)
        checkpoint_path             = ML_DATA_OUTPUT+str(model.__class__.__name__)+"["+str(res)+"]"
        
        if args.mode == "ssl_pretrained":
            if args.sslmode_modelname=="simclr_encoder":    
                pretrained_encoder          = caformer_s18_in21ft1k(config_res,args.mode,args.imnetpr).to(device)
            elif args.sslmode_modelname == "simclr_resnet":
                pretrained_encoder          = resnet_v1((3,256,256),50,1,config_res,args.mode,args.imnetpr).to(device)

            finetune=False


    elif args.mode == "ssl":
            model           = SimCLR(args.mode,args.imnetpr,config['n_classes'],args.sslmode_modelname).to(device)
            modelname       = model.modelname
            checkpoint_path = ML_DATA_OUTPUT+str(modelname)+"["+str(res)+"]"

            if args.sslmode_modelname=="simclr_caformer":
                model_to_save   = model.encoder

            elif args.sslmode_modelname == "simclr_resnet":
                model_to_save   = model.resnet

    trainable_params             = sum(	p.numel() for p in model.parameters() if p.requires_grad)
    wandb.config.update({"Model Parameters": trainable_params})


    print(f"model path:",res)
    print(f"pretrained nodel path :",config_res)              
    print('train loader transform',train_loader.dataset.tr)
    print('val loader transform',val_loader.dataset.tr)

    print(f"Model  : {model.__class__.__name__+'['+str(res)+']'}, trainable params: {trainable_params}")
    print(f"training with {len(train_loader)*args.bsize} images")
    


    best_valid_loss             = float("inf")
    optimizer                   = Adam(model.parameters(), lr=config['learningrate'])
    loss_function               = Dice_CE_Loss()
    scheduler                   = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['epochs'], eta_min=config['learningrate']/10, last_epoch=-1)
    

    for epoch in trange(config['epochs'], desc="Training"):

        epoch_loss = 0.0
        if args.mode == "ssl_pretrained":
            if finetune:
                pretrained_encoder.train()
            else:
                pretrained_encoder.eval()

        model.train()

        for  batches in tqdm(train_loader, desc=f" Epoch {epoch + 1} in training", leave=False):
            #start=timer.time()
            images,labels   = batches

            if isinstance(images, (list,tuple)):
                images=[im.to(device) for im in images] 
                labels=[lab.to(device) for lab in labels] 
            else:
                images,labels=images.to(device),labels.to(device)


            if config["augmentation"]:
                images,labels   = cutmix(images,labels,config["cutmix_pr"])


            if args.mode == "ssl":
                train_loss,_        = model(images)
                epoch_loss          += train_loss.item() 

            elif args.mode == "ssl_pretrained" or args.mode =="supervised":

                if args.mode == "ssl_pretrained":
                    _,features      = pretrained_encoder.get_features(images)
                    features        = [f.detach() for f in features]                   
                    model_output    = model(features)
                else:
                    model_output    = model(images)

                if config['n_classes'] == 1:  
                    model_output    = model_output
                    train_loss      = loss_function.Dice_BCE_Loss(model_output, labels)
                    epoch_loss      += train_loss.item() 

                else:
                    model_output    = torch.transpose(model_output,1,3) 
                    targets_f       = label_encode(labels) 
                    train_loss      = loss_function.CE_loss(model_output, targets_f)
                    epoch_loss     += train_loss.item() 

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            scheduler.step()
            #end=timer.time()
            #print(end-start)

        e_loss = {"epoch_loss" : epoch_loss / len(train_loader)}

        wandb.log(e_loss)

        print(f"Epoch {epoch + 1}/{config['epochs']}, Epoch loss for Model : {model.__class__.__name__} : {e_loss['epoch_loss']:.4f}")


        valid_loss = 0.0
        model.eval()

        with torch.no_grad():
            for batches in tqdm(val_loader, desc=f" Epoch {epoch + 1} in validating", leave=False):

                images,labels   = batches 

                if isinstance(images, (list,tuple)):
                    images = [im.to(device) for im in images] 
                    labels = [lab.to(device) for lab in labels] 
                else:
                    images,labels=images.to(device),labels.to(device)

                if args.mode == "ssl":
                    val_loss,_      = model(images)
                    valid_loss     += val_loss.item()

                elif args.mode == "ssl_pretrained" :
                    _,features = pretrained_encoder(images)
                    model_output    = model(features)
                    loss            = loss_function.Dice_BCE_Loss(model_output, labels)
                    valid_loss     += loss.item() 
                else:
                    model_output    = model(images)
                    loss            = loss_function.Dice_BCE_Loss(model_output, labels)
                    valid_loss     += loss.item() 
                     
            valid_epoch_loss = {"validation_loss": valid_loss/len(val_loader)}
            wandb.log(valid_epoch_loss)

        if valid_epoch_loss['validation_loss'] < best_valid_loss:

            print(f"previous val loss: {best_valid_loss}, new val loss: {valid_epoch_loss['validation_loss']:.4f} Saving checkpoint: {checkpoint_path}")
            best_valid_loss = valid_epoch_loss['validation_loss']

            if args.mode == "ssl_pretrained" or args.mode == "supervised":
                torch.save(model.state_dict(), checkpoint_path)   # saving supervised or ssl_pretrained supervised params with configs

                if args.mode == "ssl_pretrained" and finetune:
                    torch.save(pretrained_encoder.state_dict(), checkpoint_path)
                    
            else:
                torch.save(model_to_save.state_dict(), checkpoint_path)   # saving supervised or ssl_pretrained supervised params with configs


        print(f"\n Training {model.__class__.__name__}, training Loss: {e_loss['epoch_loss']:.4f}, val. Loss: {valid_epoch_loss['validation_loss']:.4f}")
        
    wandb.finish()

if __name__ == "__main__":
   main()