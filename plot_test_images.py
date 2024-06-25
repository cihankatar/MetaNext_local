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

from models.CA_CBA_CA import CA_CBA_CA
from models.CA_CA import CA_CA
from models.CA_CBA_mnext import CA_CBA_MNEXT_B
from models.CA_CBA_convatt import CA_CBA_ConvAtt
from models.CA_CBA_convatt_1 import CA_CBA_ConvAtt_1
from models.CA_convatt import CA_Convatt


from models.CA_CBA_Convnext_pyrmd import CA_CBA_Convnext_pyrmd
from models.CA_CBA_Convnext_f import CA_CBA_Convnext_f
from models.CA_CBA_Convnext import CA_CBA_Convnext
from models.CA_Convnext import CA_Convnext
from models.CA_CBA_Unet import CA_CBA_UNET
from models.CA_Unet import CA_UNET
from models.Unet import UNET

from models.Metaformer import caformer_s18_in21ft1k
from models.resnet import resnet_v1
from data.data_loader import data_transform
from data.Custom_Dataset import dataset
from PIL import Image
from glob import glob


def get_images(train_path,mask_path,index,transforms):        
        print(train_path[int(index)])
        image = Image.open(train_path[int(index)])
        image = np.array(image,dtype=float)
        image = image.astype(np.float32)
        image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(image)

        mask = Image.open(mask_path[int(index)]).convert('L')            
        mask = np.array(mask,dtype=float)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)
        mask = mask.unsqueeze(0)

        image=image/255
        mask=mask/255

        s = np.random.randint(0,2**16)   
        np.random.seed(s)
        torch.manual_seed(s)
        image = transforms(image)
        np.random.seed(s)
        torch.manual_seed(s)
        mask = transforms(mask)

        return image , mask

def using_device():
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "") 
    return device

if __name__ == "__main__":

    
    data='isic_1'
    training_mode="supervised"
    train=False

    if data =='isic_1':
        foldernamepath ="isic_1/"
    elif data == 'kvasir_1':
        foldernamepath ="kvasir_1/"

    WANDB_DIR           = os.environ["WANDB_DIR"]
    WANDB_API_KEY       = os.environ["WANDB_API_KEY"]
    ML_DATA_OUTPUT      = os.environ["ML_DATA_OUTPUT"]+foldernamepath


    args,res,config_res = parser_init("isic_segmetnation","testing",training_mode="supervised",train=False)
    res=', '.join(res)
    config_res=', '.join(config_res)
  
    config           = wandb_init(WANDB_API_KEY,WANDB_DIR,args,plt_test=True)

    if args.mode == "ssl_pretrained" or args.mode == "supervised":
        model                       = CA_CBA_MNEXT_B(config['n_classes'],config_res,args.mode,args.imnetpr).to(device)
        checkpoint_path             = ML_DATA_OUTPUT+str(model.__class__.__name__)+"["+str(res)+"]"
        
        if args.mode == "ssl_pretrained":
            if args.sslmode_modelname=="simclr_caformer":    
                pretrained_encoder          = caformer_s18_in21ft1k(config_res,args.mode,args.imnetpr).to(device)
            elif args.sslmode_modelname == "simclr_resnet":
                pretrained_encoder          = resnet_v1((3,256,256),50,1,config_res,args.mode,args.imnetpr).to(device)
    
    checkpoint_path             = ML_DATA_OUTPUT+str(model.__class__.__name__)+"["+str(res)+"]"
    trainable_params             = sum(	p.numel() for p in model.parameters() if p.requires_grad)

    args.aug = False

    model.eval()
    print(f"plotting  : {model.__class__.__name__}, model params: {trainable_params}")
    
    try:
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(checkpoint_path))
        else: 
            model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
    except:
        raise Exception("******* No Checkpoint Path  *********")

        #    plotting part  #
    args.shuffle = False
    args.testvis = False
    _,test_loader    = loader(args.mode,args.sslmode_modelname,args.train, args.bsize,args.workers,args.imsize,args.cutoutpr,args.cutoutbox,args.aug,args.shuffle,args.sratio)
    
    step=0

    columns = ["model_name","image", "pred", "target"]
    image_table = wandb.Table(columns=columns)

    index = 4
    train_im_path   = os.environ["ML_DATA_ROOT"]+"isic_2018/train/images"   
    train_mask_path = os.environ["ML_DATA_ROOT"]+"isic_2018/train/masks"
    test_im_path    = os.environ["ML_DATA_ROOT"]+"isic_2018/test/images"
    test_mask_path  = os.environ["ML_DATA_ROOT"]+"isic_2018/test/masks"

    train_im_path   = sorted(glob(train_im_path+"/*.jpg"))
    train_mask_path = sorted(glob(train_mask_path+"/*.png"))
    test_im_path    = sorted(glob(test_im_path+"/*.jpg"))
    test_mask_path  = sorted(glob(test_mask_path+"/*.png"))
    
    transformations = data_transform(args.mode,args.sslmode_modelname,args.train,args.imsize)
    images,labels   = get_images(test_im_path, test_mask_path,index, transformations)
    images=torch.unsqueeze(images, 0)
    labels=torch.unsqueeze(labels, 0)

    if args.testvis:
        for step in len(train_im_path):
            if step%20==1:
                if args.mode == "ssl_pretrained":
                    _,features = pretrained_encoder(images)
                    model_output    = model(features)
                else:
                    model_output    = model(images)
                prediction      = torch.sigmoid(model_output)

                im,lab,pred=log_image_table(images, prediction, labels, args.bsize)
                image_table.add_data(str(model.__class__.__name__)+'_'+str(config),wandb.Image(im),wandb.Image(pred),wandb.Image(lab))  # multiple output for one test

    else:
        if args.mode == "ssl_pretrained":
            _,features = pretrained_encoder(images)
            model_output    = model(features)
        else:
            model_output    = model(images)
        print("plotting one set of figures")
        prediction      = torch.sigmoid(model_output)
        im,pred,lab     = log_image_table(images, prediction, labels, (len(test_loader.dataset)%args.bsize)-1)
        image_table.add_data(str(model.__class__.__name__)+'_'+str(config),wandb.Image(im),wandb.Image(pred),wandb.Image(lab))
        

    wandb.log({"predictions_table":image_table})
    wandb.finish()    
