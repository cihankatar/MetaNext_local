import wandb
import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch

def parse_bool(s):
    return True if str(s)=="True" else False

def config_func(training_mode):
    if training_mode == "ssl":
        configs={
        "mode"              :"ssl",
        "sslmode_modelname" :"simclr_caformer",
        "imnetpr"           :True,
        "bsize"             :50,
        "epochs"            :150,
        "imsize"            :256,
        "lrate"             :0.001,
        "aug"               :False,
        "shuffle"           :True,
        "sratio"            :None,
        "workers"           :2,
        "cutoutpr"          :None,
        "cutoutbox"         :None,
        "cutmixpr"          :None,
        "noclasses"         :1,
        }

    elif training_mode == "supervised":

        configs={
        "mode"              :"supervised",
        "sslmode_modelname" :None,
        "imnetpr"           :True,
        "bsize"             :8,
        "epochs"            :150,
        "imsize"            :256,
        "lrate"             :0.0001,
        "aug"               :True,
        "shuffle"           :True,
        "sratio"            :None,
        "workers"           :2,
        "cutoutpr"          :0.5,
        "cutoutbox"         :25,
        "cutmixpr"          :0.5,
        "noclasses"         :1,
        }


    elif training_mode == "ssl_pretrained":
        ssl_config_to_load = config_func("ssl")
        configs={
        "mode"              :"ssl_pretrained",
        "sslmode_modelname" :"simclr_caformer",
        "imnetpr"           :True,
        "bsize"             :8,
        "epochs"            :150,
        "imsize"            :256,
        "lrate"             :0.0001,
        "aug"               :True,
        "shuffle"           :True,
        "sratio"            :0.01,
        "workers"           :2,
        "cutoutpr"          :0.5,
        "cutoutbox"         :25,
        "cutmixpr"          :0.5,
        "ssl_config"        :ssl_config_to_load,
        "noclasses"         :1
        }

    else:
        raise Exception('Invalid training type:', training_mode)

    return configs
    
def parser_init(name,desc,
                training_mode=None,
                train=None):
    
    parser = argparse.ArgumentParser(
        prog=name,
        description=desc )
    
    configs=config_func(training_mode)

    parser.add_argument("-m", "--mode",             type=str,default=configs["mode"])   
    parser.add_argument("-y", "--sslmode_modelname",type=str,default=configs["sslmode_modelname"])
    parser.add_argument("-p", "--imnetpr",          type=parse_bool,default=configs["imnetpr"])
    parser.add_argument("-b", "--bsize",            type=int,default=configs["bsize"])   
    parser.add_argument("-e", "--epochs",           type=int,default=configs["epochs"])
    parser.add_argument("-i", "--imsize",           type=int,default=configs["imsize"])   
    parser.add_argument("-l", "--lrate",            type=float,default=configs["lrate"])  
    parser.add_argument("-a", "--aug",              type=parse_bool,default=configs["aug"])
    parser.add_argument("-s", "--shuffle",          type=parse_bool,default=configs["shuffle"])
    parser.add_argument("-r", "--sratio",           type=float,default=configs["sratio"])
    parser.add_argument("-w", "--workers",          type=int,default=configs["workers"])
    parser.add_argument("-o", "--cutoutpr",         type=float,default=configs["cutoutpr"])
    parser.add_argument("-c", "--cutoutbox",        type=float,default=configs["cutoutbox"])
    parser.add_argument("-x", "--cutmixpr",         type=float,default=configs["cutmixpr"])
    parser.add_argument("-n", "--noclasses",        type=int,default=configs["noclasses"])
    parser.add_argument("-t", "--train",            type=parse_bool,default=train)

    args=parser.parse_args()
    args_key=[]
    args_value=[]
    res=[]
    config_key=[]
    config_value=[]
    config_res=[]

    
    for key,value in parser.parse_args()._get_kwargs():
        if args.mode == "ssl":
            if key=='sratio'or key=='train' or key=='noclasses' or key=='cutoutpr'or key=='cutmixpr'or key=='cutoutbox' or key=='workers':
                continue
        elif args.mode == "supervised":
            if key=='sratio'or key=='sslmode_modelname'or key=='train'or key=='noclasses' or key=='workers':
                continue
        elif args.mode == "ssl_pretrained":
            if key=='train'or key=='noclasses' or key=='workers' or key=="sslmode_modelname":
                continue

        args_key.append(key)
        args_value.append(str(value))

    for i in range(len(args_key)):
        res.append(args_key[i]+"="+args_value[i])


    if args.mode == "ssl_pretrained":
        for key,value in configs["ssl_config"].items():
            if key=="imnetpr":
                value=args.imnetpr

            if key=="sslmode_modelname":
                value=args.sslmode_modelname

            if key=="bsize":
                value=args.bsize
                args.bsize = 8      # returning supervised config to save 
                
            if key=="epochs":
                value=args.epochs
                args.epochs = 150   # returning supervised config to save 

            if key=='sratio'or key=='train' or key=='noclasses' or key=='cutoutpr'or key=='cutmixpr'or key=='cutoutbox' or key=='workers':
                continue

            config_key.append(key)
            config_value.append(str(value))

        for i in range(len(config_key)):
            config_res.append(config_key[i]+"="+config_value[i])      


    return args,res,config_res

def wandb_init (WANDB_API_KEY,WANDB_DIR,args,config_log,data):

    ssl_config      = config_log
    training_mode   = args.mode
    ssl_mode_modelname = args.sslmode_modelname
    train           = args.train
    imnetpr         = args.imnetpr
    batch_size      = args.bsize 
    epochs          = args.epochs
    image_size      = args.imsize
    augmentation    = args.aug
    learningrate    = args.lrate
    n_classes       = args.noclasses
    split_ratio     = args.sratio
    shuffle         = args.shuffle
    cutout_pr       = args.cutoutpr
    box_size        = args.cutoutbox
    cutmixpr        = args.cutmixpr
    workers         = args.workers

    print(f'Taining Configs:\ntrain:{train}\ntraining_mode:{training_mode}\nssl_mode_modelname:{ssl_mode_modelname}\nimagenetpretrained:{imnetpr} \nbatch_size:{batch_size}, \nepochs:{epochs}, \nimagesize:{image_size}, \naugmentation:{augmentation}, \nl_r:{learningrate}, \nn_classes:{n_classes}, \nshuffle:{shuffle}, \ncutout_pr:{cutout_pr}, \ncutout_box_size:{box_size},\ncutmixpr:{cutmixpr}, \nworkers:{workers},\nsplit_ratio:{split_ratio}')

    wandb.login(key=WANDB_API_KEY)
    if train: 
            
        if torch.cuda.is_available():
            if training_mode =="ssl":
                project_name = data+"_ssl"

            elif training_mode =="ssl_pretrained":
                project_name = data+"_ssl_pretrained_segmentation_task"

            else:
                project_name = data+"_segmentation_task_supervised"

        else:
            if training_mode =="ssl":
                project_name = data+"temp_training_ssl_local"
            elif training_mode =="ssl_pretrained":
                project_name = data+"_temp_ssl_pretrained_segmentation_task_local"
            else:
                project_name = "temp_segmentation_task_local_"+data
                
    else:
        if training_mode =="ssl":
            project_name = data+"_test_ssl"

        else:
            project_name = data+"_segmentation_supervised_test"

                
    wandb.init(
        project=project_name,
        dir=WANDB_DIR,
        config={
            "training_mode"  : training_mode,
            "ssl_mode_modelnm": ssl_mode_modelname,
            "train"          : train,
            "imnetpretrained": imnetpr,
            "epochs"         : epochs,
            "batch_size"     : batch_size,
            "learningrate"   : learningrate,
            "n_classes"      : n_classes,
            "split_ratio"    : split_ratio,
            "num_workers"    : workers,
            "image_size"     : image_size,
            "cutmix_pr"      : cutmixpr,
            "cutout_pram"    : [cutout_pr,box_size],
            "augmentation"   : augmentation,
            "shuffle"        : shuffle,
            "ssl_config"     : ssl_config
            })

    config = wandb.config

    return config

def log_image_table(im_test, prediction, labels, batch_size):
    #rand_idx=np.random.randint(batch_size)
    rand_idx=2
    prediction    = prediction[rand_idx]        ## (1, 512, 512)
    prediction    = np.squeeze(prediction)     ## (512, 512)
    prediction    = prediction > 0.5
    prediction    = np.array(prediction, dtype=np.uint8)
    prediction    = np.transpose(prediction)
    im_test       = np.array(im_test[rand_idx]*255,dtype=int)
    im_test       = np.transpose(im_test, (2,1,0))
    im_test       = np.squeeze(im_test)     ## (512, 512)
    label_test    = np.array(labels[rand_idx]*255,dtype=int)
    label_test    = np.transpose(label_test)
    label_test    = np.squeeze(label_test)     ## (512, 512)

    return im_test,prediction,label_test
