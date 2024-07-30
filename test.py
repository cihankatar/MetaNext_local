import torch
import wandb
import os 

from operator import add
from tqdm import tqdm, trange
import wandb

from wandb_init  import *
from visualization  import test_image
from utils.metrics import *

from data.data_loader import loader

"""
from models.CA_CBA_CA import CA_CBA_CA
from models.CA_CA import CA_CA
from models.CA_CBA_mnext import CA_CBA_MNEXT_B
from models.CA_CBA_Convnext import CA_CBA_Convnext
from models.CA_Convnext import CA_Convnext
from models.CA_CBA_Unet import CA_CBA_UNET
from models.CA_Unet import CA_UNET
from models.Unet import UNET
from models.CA_Proposed import CA_Proposed

"""
from models.Unet import UNET
from models.CA_CBA_Proposed import CA_CBA_Proposed
from models.CA_Proposed import CA_Proposed

from models.Model import model_base
from SSL.simclr import SimCLR
from models.Metaformer import caformer_s18_in21ft1k
from models.resnet import resnet_v1

from models.Metaformer import caformer_s18_in21ft1k
from models.resnet import resnet_v1


def using_device():
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "") 
    return device

if __name__ == "__main__":

    device      = using_device()

    data='kvasir_1'
    training_mode="supervised"
    train=False

    if data =='isic_1':
        foldernamepath ="isic_1/"
    elif data == 'kvasir_1':
        foldernamepath ="kvasir_1/"

    WANDB_DIR           = os.environ["WANDB_DIR"]
    WANDB_API_KEY       = os.environ["WANDB_API_KEY"]
    ML_DATA_OUTPUT      = os.environ["ML_DATA_OUTPUT"]+foldernamepath

    args,res,config_res = parser_init("segmetnation_task","testing",training_mode,train)
    res                 = ', '.join(res)
    config_res          = ', '.join(config_res)
    config              = wandb_init(WANDB_API_KEY,WANDB_DIR,args,config_res,data)

    

    if args.mode == "ssl_pretrained" or args.mode == "supervised":
        model                       = CA_Proposed(config['n_classes'],config_res,args.mode,args.imnetpr).to(device)
        checkpoint_path             = ML_DATA_OUTPUT+str(model.__class__.__name__)+"["+str(res)+"]"
        
        if args.mode == "ssl_pretrained":
            if args.sslmode_modelname=="simclr_caformer":    
                pretrained_encoder          = caformer_s18_in21ft1k(config_res,args.mode,args.imnetpr).to(device)
            elif args.sslmode_modelname == "simclr_resnet":
                pretrained_encoder          = resnet_v1((3,256,256),50,1,config_res,args.mode,args.imnetpr).to(device)
    
    checkpoint_path             = ML_DATA_OUTPUT+str(model.__class__.__name__)+"["+str(res)+"]"
    trainable_params            = sum(	p.numel() for p in model.parameters() if p.requires_grad)

    args.aug       = False
    test_loader    = loader(
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
    
    print(f"model path:",res)
    print(f"pretrained nodel path :",config_res)
    print('test_loader loader transform',test_loader.dataset.tr)

    print(f"Testing for Model  : {model.__class__.__name__}, model params: {trainable_params}")
    print(f"training with {len(test_loader)*args.bsize} images")

    
    try:
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(checkpoint_path))
        else: 
            model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
    except:
        raise Exception("******* No Checkpoint Path  *********")

    metrics_score = [ 0.0, 0.0, 0.0, 0.0, 0.0]
    model.eval()

    for batch in tqdm(test_loader, desc=f"testing ", leave=False):
        images,labels   = batch                

        with torch.no_grad():

            if args.mode == "ssl_pretrained":
                _,features      = pretrained_encoder.get_features(images)
                model_output    = model(features)
            else:
                model_output    = model(images)

            prediction  = torch.sigmoid(model_output)

            if args.noclasses>1:
                prediction    = torch.argmax(prediction,dim=2)    #for multiclass_segmentation

            else:
                score = calculate_metrics(labels, prediction)
                metrics_score = list(map(add, metrics_score, score))

            acc = {    "jaccard"      : metrics_score[0]/len(test_loader),
                        "f1"          : metrics_score[1]/len(test_loader),
                        "recall"      : metrics_score[2]/len(test_loader),
                        "precision"   : metrics_score[3]/len(test_loader),
                        "acc"         : metrics_score[4]/len(test_loader)
                        }
        print(f" Jaccard (IoU): {acc['jaccard']:1.4f} - F1(Dice): {acc['f1']:1.4f} - Recall: {acc['recall']:1.4f} - Precision: {acc['precision']:1.4f} - Acc: {acc['acc']:1.4f} ")

        #  PLOTTING  #

    args.shuffle = False

    test_loader    = loader(args.mode,args.sslmode_modelname,args.train, args.bsize,args.workers,args.imsize,args.cutoutpr,args.cutoutbox,args.aug,args.shuffle,args.sratio,data)
    
    columns = ["model_name","image", "pred", "target","Jaccard(IOU)"]
    image_table = wandb.Table(columns=columns) 

    for batch in test_loader:
        images,labels   = batch

        if args.mode == "ssl_pretrained":
            _,features = pretrained_encoder(images)
            model_output    = model(features)
        else:
            model_output    = model(images)
            
        print("plotting one set of figure")
        prediction      = torch.sigmoid(model_output)
        im,pred,lab     = log_image_table(images, prediction, labels, (len(test_loader.dataset)%args.bsize)-1)
        image_table.add_data(str(model.__class__.__name__)+'_'+str(res),wandb.Image(im),wandb.Image(pred),wandb.Image(lab),acc["jaccard"])
        break

    wandb.log({"predictions_table":image_table})
    wandb.log(acc)
    wandb.finish()    
