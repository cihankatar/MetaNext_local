import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Metaformer import caformer_s18_in21ft1k
from models.enc import encoder_function
from utils.batch_norm import SyncBatchNorm, BatchNorm1d,BatchNorm2d
from utils.layer_norm import LayerNorm
from models.resnet import resnet_v1

# Some code adapted from https://github.com/sthalles/SimCLR

class SimCLR(nn.Module):
    metrics = ['Loss']
    metrics_fmt = [':.4e']
    
    def __init__(self,training_mode,imnetpretrained, n_classes, modelnm,config_res=None,dataset=None,dist=None):
        super().__init__()
        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.temperature = 0.5
        self.projection_dim = 128

        if modelnm == "simclr_caformer":
            self.encoder = encoder_function(config_res,training_mode,imnetpretrained)
            self.modelname = self.encoder.__class__.__name__
            self.model = SyncBatchNorm.convert_sync_batchnorm(self.encoder)
        
        elif modelnm == "simclr_resnet":
            self.resnet     = resnet_v1((3,256,256), 50 ,1,config_res,training_mode,imnetpretrained)
            self.modelname = self.resnet.__class__.__name__
            self.model = SyncBatchNorm.convert_sync_batchnorm(self.resnet)

        self.latent_dim = 2048 if modelnm=="simclr_resnet" else 512
        self.proj = nn.Sequential(
            nn.Linear(self.latent_dim, self.projection_dim, bias=False),
            BatchNorm1d(self.projection_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.projection_dim, self.projection_dim, bias=False),
            BatchNorm1d(self.projection_dim, center=False)
        )

        self.dataset = dataset
        self.n_classes = n_classes
        self.dist = dist

    def construct_classifier(self):
        return nn.Sequential(nn.Linear(self.latent_dim, self.n_classes))

    def forward(self, images):
        n = images[0].shape[0]
        xi, xj = images
        hi, hj = self.encode(xi), self.encode(xj) # (N, latent_dim)
        if self.modelname=="ResNet":
            zi, zj = self.proj(hi[0]), self.proj(hj[0]) # (N, projection_dim)
            zi, zj = F.normalize(zi), F.normalize(zj)
        elif self.modelname=="Encoder":
            hi_out,hj_out  =   torch.mean(hi[1][3], dim=[2, 3]).squeeze(),torch.mean(hj[1][3], dim=[2, 3]).squeeze()
            zi, zj = self.proj(hi_out), self.proj(hj_out) # (N, projection_dim)
            zi, zj = F.normalize(zi), F.normalize(zj)

        # Each training example has 2N - 2 negative samples
        # 2N total samples, but exclude the current and positive sample

        if self.dist is None:
            zis = [zi]
            zjs = [zj]
        else:
            zis = [torch.zeros_like(zi) for _ in range(self.dist.get_world_size())]
            zjs = [torch.zeros_like(zj) for _ in range(self.dist.get_world_size())]

            self.dist.all_gather(zis, zi)
            self.dist.all_gather(zjs, zj)

        z1 = torch.cat((zi, zj), dim=0) # (2N, projection_dim)
        z2 = torch.cat(zis + zjs, dim=0) # (2N * n_gpus, projection_dim)

        sim_matrix = torch.mm(z1, z2.t()) # (2N, 2N * n_gpus)
        sim_matrix = sim_matrix / self.temperature
        # Mask out same-sample terms
        n_gpus = 1 if self.dist is None else self.dist.get_world_size()
        rank = 0 if self.dist is None else self.dist.get_rank()
        sim_matrix[torch.arange(n), torch.arange(rank*n, (rank+1)*n)]  = -float('inf')
        sim_matrix[torch.arange(n, 2*n), torch.arange((n_gpus+rank)*n, (n_gpus+rank+1)*n)] = -float('inf')

        targets = torch.cat((torch.arange((n_gpus+rank)*n, (n_gpus+rank+1)*n),
                             torch.arange(rank*n, (rank+1)*n)), dim=0).to(self.device) 
        targets = targets.long().to(self.device) 

        loss = F.cross_entropy(sim_matrix, targets, reduction='sum')
        loss = loss / n

        return loss, hi

    def encode(self, images):
        return self.model.get_features(images) 

    def get_features(self, images):
        return self.resnet.get_features(images)


"""
def loss_man(a,b,tau=0.5):
        a_norm = torch.norm(a,dim=1).reshape(-1,1)
        a_cap = torch.div(a,a_norm)
        b_norm = torch.norm(b,dim=1).reshape(-1,1)
        b_cap = torch.div(b,b_norm)
        a_cap_b_cap = torch.cat([a_cap,b_cap],dim=0)
        a_cap_b_cap_transpose = torch.t(a_cap_b_cap)
        b_cap_a_cap = torch.cat([b_cap,a_cap],dim=0)
        sim = torch.mm(a_cap_b_cap,a_cap_b_cap_transpose)
        sim_by_tau = torch.div(sim,tau)
        exp_sim_by_tau = torch.exp(sim_by_tau)
        sum_of_rows = torch.sum(exp_sim_by_tau, dim=1)
        exp_sim_by_tau_diag = torch.diag(exp_sim_by_tau)
        numerators = torch.exp(torch.div(torch.nn.CosineSimilarity()(a_cap_b_cap,b_cap_a_cap),tau))
        denominators = sum_of_rows - exp_sim_by_tau_diag
        num_by_den = torch.div(numerators,denominators)
        neglog_num_by_den = -torch.log(num_by_den)
        return torch.mean(neglog_num_by_den)
        """