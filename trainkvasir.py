
import argparse
import os
import time as timer
import matplotlib.pyplot as plt
import torch
import wandb
import yaml
from torch.optim import Adam
from tqdm import tqdm, trange

from augmentation.Augmentation import Cutout,cutmix
from data.data_loader import loader
from utils.Loss import Dice_CE_Loss, Topological_Loss
from utils.one_hot_encode import label_encode, one_hot
from visualization import *
from wandb_init import parser_init, wandb_init

from models.Metaformer import caformer_s18_in21ft1k
from models.Model import model_tripleloss
from models.resnet import resnet_v1
from SSL.simclr import SimCLR

"""
from models.CA_CBA_CA import CA_CBA_CA
from models.CA_CA import CA_CA
from models.CA_CBA_mnext import CA_CBA_MNEXT_B
from models.CA_CBA_Convnext import CA_CBA_Convnext
from models.CA_Convnext import CA_Convnext
from models.CA_CBA_Unet import CA_CBA_UNET
from models.CA_Unet import CA_UNET
from models.Unet import UNET
from models.CA_CBA_Proposed import CA_CBA_Proposed
from models.CA_Proposed import CA_Proposed
from models.Unet import UNET

"""
#from ptflops import get_model_complexity_info
#import re


def load_config(config_name):
    with open(config_name) as file:
        config = yaml.safe_load(file)
    return config

def using_device():
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "") 
    return device

def main():

    #### Initial Configs ###
    data            ='kvasir_1'
    training_mode   ="supervised"
    train           = True
    addtopoloss     = True
    
    augmentation_regularization = False
    aug_threshould = 0
    
    device         = using_device()
    WANDB_DIR      = os.environ["WANDB_DIR"]
    WANDB_API_KEY  = os.environ["WANDB_API_KEY"]

    if data=='isic_1':
        foldernamepath="isic_1/"
    elif data == 'kvasir_1':
        foldernamepath="kvasir_1/"

    if torch.cuda.is_available():
        ML_DATA_OUTPUT = os.environ["ML_DATA_OUTPUT"]+foldernamepath
    else:
        ML_DATA_OUTPUT = os.environ["ML_DATA_OUTPUT_LOCAL"]+foldernamepath

    args,res,config_res   = parser_init("segmetnation task","training",training_mode,train)
    res                   = ', '.join(res)
    config_res            = ', '.join(config_res)
    config                = wandb_init(WANDB_API_KEY,WANDB_DIR,args,config_res,data)
    #config               = load_config("config.yaml")
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
    args.aug = True
    ##### Model Building based on arguments  ####

    if args.mode == "ssl_pretrained" or args.mode == "supervised":
        model           = model_tripleloss(config['n_classes'],config_res,args.mode,args.imnetpr).to(device)
        checkpoint_path = ML_DATA_OUTPUT+str(model.__class__.__name__)+"["+str(res)+"]"
        
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


    #print(f"model path:",res)
    #print(f"pretrained nodel path :",config_res)     
    #macs, params = get_model_complexity_info(model, (3, 256, 256), as_strings=True,
    #print_per_layer_stat=True, verbose=True)
    # Extract the numerical value
    #flops = eval(re.findall(r'([\d.]+)', macs)[0])*2
    # Extract the unit
    #flops_unit = re.findall(r'([A-Za-z]+)', macs)[0][0]
    #print('Computational complexity: {:<8}'.format(macs))
    #print('Computational complexity: {} {}Flops'.format(flops, flops_unit))
    #print('Number of parameters: {:<8}'.format(params))
    

    best_valid_loss             = float("inf")
    optimizer                   = Adam(model.parameters(), lr=config['learningrate'])
    loss_function               = Dice_CE_Loss()
    TopoLoss                    = Topological_Loss(lam=0.5).to(device)

    scheduler                   = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['epochs'], eta_min=config['learningrate']/10, last_epoch=-1)
    initialcutoutpr             = args.cutoutpr 
    initialcutmixpr             = args.cutmixpr
    
    trainable_params             = sum(	p.numel() for p in model.parameters() if p.requires_grad)
    wandb.config.update({"Model Parameters": trainable_params})
    print('Train loader transform',train_loader.dataset.tr)
    print('Val loader transform',val_loader.dataset.tr)
    print(f"Model  : {model.__class__.__name__+'['+str(res)+']'}, trainable params: {trainable_params}")
    print(f"Training with {len(train_loader)*args.bsize} images~~ , Saving checkpoint: {checkpoint_path}")
    print(f"Topology Loss Config: regularization, {augmentation_regularization}, lambda - {TopoLoss.lam}, Addtopoloss:{addtopoloss}")
    
    ##########  TRAINING ##########
#    args.shuffle = False

    for epoch in trange(config['epochs'], desc="Training"):

        epoch_loss = 0.0
        epoch_DiceBCEloss=0.0
        epoch_topo_loss=0.0

        if args.mode == "ssl_pretrained":
            if finetune:
                pretrained_encoder.train()
            else:
                pretrained_encoder.eval()

        if augmentation_regularization:
            if epoch >= aug_threshould/2 and epoch <= aug_threshould:
                args.cutoutpr = initialcutoutpr - epoch/(aug_threshould*4)
                args.cutmixpr = initialcutmixpr - epoch/(aug_threshould*4)
                print('augmentation prabability is reducing -- ',args.cutoutpr)

            if epoch > aug_threshould:
                args.aug=False
    
        model.train()

        for  batches in train_loader:
            #start=timer.time()
            images,labels   = batches


            if isinstance(images, (list,tuple)):
                images=[im.to(device) for im in images] 
                labels=[lab.to(device) for lab in labels] 
            else:
                images,labels=images.to(device),labels.to(device)
            
            images_copy = images.clone()
            labels_copy = labels.clone()
            
            if args.aug:
                images,labels   = cutmix(images, labels,args.cutmixpr)
                images,labels   = Cutout(images,labels,args.cutoutpr,args.cutoutbox)  
            
                for i in range(labels.shape[0]):
                    if torch.count_nonzero(labels[i]) < 100:
                        print( "skipping aug")
                        images = images_copy
                        labels = labels_copy
                        break
            
            if args.mode == "ssl_pretrained" or args.mode =="supervised":

                if args.mode == "ssl_pretrained":
                    _,features      = pretrained_encoder.get_features(images)
                    features        = [f.detach() for f in features]                   
                    model_output    = model(features)
                    DiceBCE_loss    = loss_function.Dice_BCE_Loss(model_output, labels)
                    epoch_loss      += DiceBCE_loss.item() 

                elif args.mode =="supervised":
                    _,model_output  = model(images)
                    DiceBCE_loss    = loss_function.Dice_BCE_Loss(model_output, labels)
                    if addtopoloss:
                        topo_loss           = TopoLoss(model_output,labels)
                        Dice_BCE_Topo_loss  = DiceBCE_loss + topo_loss
                        epoch_loss          += Dice_BCE_Topo_loss.item()
                        epoch_topo_loss     += topo_loss.item()
                        epoch_DiceBCEloss   += DiceBCE_loss.item()
                    else:
                        epoch_loss              += DiceBCE_loss.item() 

            elif args.mode == "ssl":
                train_loss,_        = model(images)
                epoch_loss          += train_loss.item() 

            optimizer.zero_grad()
            if addtopoloss:
                Dice_BCE_Topo_loss.backward()
            else:
                DiceBCE_loss.backward()
            optimizer.step()
            scheduler.step()
            #end=timer.time()
            #print(end-start)

        e_loss              = {"epoch_loss" : epoch_loss / len(train_loader)}
        wandb.log(e_loss)

        if addtopoloss:
            epoch_topo_loss     = {"Training_Topo_L" : epoch_topo_loss / len(train_loader)}
            epoch_DiceBCEloss   = {"Training_DiceBCE_L" : epoch_DiceBCEloss / len(train_loader)}
            wandb.log(epoch_topo_loss)
            wandb.log(epoch_DiceBCEloss)
        
        if addtopoloss:
            print(f"-->Epoch {epoch + 1}/{config['epochs']}, Training Losses : BCE+DiceL+TopoL: {e_loss['epoch_loss']:.4f}, DiceBCEloss: {epoch_DiceBCEloss['Training_DiceBCE_L']:.4f}, TopoL: {epoch_topo_loss['Training_Topo_L']:.4f} ")
        else:
            print(f"-->Epoch {epoch + 1}/{config['epochs']}, Training Losses : BCE+DiceL : {e_loss['epoch_loss']:.4f}")

    ##########  VALIDATION ##########

        valid_loss = 0.0
        valid_topo_loss = 0.0
        valid_dicebce_loss = 0.0
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
                    valid_loss      += val_loss.item()

                elif args.mode == "ssl_pretrained" :
                    _,features      = pretrained_encoder(images)
                    model_output    = model(features)
                    DiceBCE_l       = loss_function.Dice_BCE_Loss(model_output, labels)
                    valid_loss      += DiceBCE_l.item() 


                elif args.mode =="supervised":
                    _,model_output       = model(images)
                    DiceBCE_l            = loss_function.Dice_BCE_Loss(model_output, labels)

                    if addtopoloss:
                        topo_loss               = TopoLoss(model_output,labels)
                        Dice_BCE_Topo_loss      = DiceBCE_l+topo_loss
                        valid_loss             += Dice_BCE_Topo_loss.item() 
                        valid_topo_loss        += topo_loss.item() 
                        valid_dicebce_loss     += DiceBCE_l.item() 

                    else:
                        valid_loss     += DiceBCE_l.item() 

            valid_epoch_loss = {"validation_loss": valid_loss/len(val_loader)}
            wandb.log(valid_epoch_loss)

            if addtopoloss:
                valid_topo_loss = {"Valid_Topo_L": valid_topo_loss/len(val_loader)}
                valid_dicebce_loss = {"Valid_DiceBCE_L": valid_dicebce_loss/len(val_loader)}
                wandb.log(valid_topo_loss)
                wandb.log(valid_dicebce_loss)

            if augmentation_regularization:
                if epoch==aug_threshould+1:
                   best_valid_loss = best_valid_loss+valid_topo_loss
                   print(f" Epoch {epoch + 1}/{config['epochs']}, set best_valid_loss (BCE+DiceL+TopoL/{len(val_loader)}): {best_valid_loss:.4f}")


        if addtopoloss:
            print(f" --> Validation Losses: BCE+DiceL+TopoL: {valid_epoch_loss['validation_loss']:.4f}, Valid_DiceBCE_loss:{valid_dicebce_loss['Valid_DiceBCE_L']:.4f}, Valid_TopoLoss: {valid_topo_loss['Valid_Topo_L']:.4f} ")
        else:
            print(f" --> Validation Losses: BCE+DiceL: {valid_epoch_loss['validation_loss']:.4f} ")
      
        if valid_epoch_loss['validation_loss'] < best_valid_loss:
            print(f" Best val. lost updated: {valid_epoch_loss['validation_loss']:.4f},  New param. saved to checkpoint_path \n")
            best_valid_loss = valid_epoch_loss['validation_loss']

            if args.mode == "ssl_pretrained" or args.mode == "supervised":
                torch.save(model.state_dict(), checkpoint_path)   # saving supervised or ssl_pretrained supervised params with configs

                if args.mode == "ssl_pretrained" and finetune:
                    torch.save(pretrained_encoder.state_dict(), checkpoint_path)
                    
            else:
                torch.save(model_to_save.state_dict(), checkpoint_path)   # saving supervised or ssl_pretrained supervised params with configs
  
        print(f" ---------------------  --------------------------- ")

    wandb.finish()

if __name__ == "__main__":
    main()