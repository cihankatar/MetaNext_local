import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat
import matplotlib.pyplot as plt 




def patch_to_image(images,out_dim):

    images = torch.unsqueeze(images, dim=1)
    
    #img_patches = images.reshape(2,1,128,128)
    
    img_patches = rearrange(images, 'b c (patch_x x) y -> b c (x) (patch_x y)',
                                    patch_x=1)
    return img_patches



def patching(images,patch_size):
    
    img_patches = rearrange(images, 'b c (patch_x h) (patch_y w) -> b (h w) (patch_x patch_y c)',
                                    patch_x=patch_size, patch_y=patch_size)

    return img_patches
class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, head_num):
        super().__init__()

        self.head_num = head_num
        self.dk = (embedding_dim // head_num) ** (1 / 2)

        self.qkv_layer = nn.Linear(embedding_dim, embedding_dim * 3, bias=False)
        self.out_attention = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, x, mask=None):
        qkv = self.qkv_layer(x)

        query, key, value = tuple(rearrange(qkv, 'b t (d k h ) -> k b h t d ', k=3, h=self.head_num))
        energy = torch.einsum("... i d , ... j d -> ... i j", query, key) * self.dk

        if mask is not None:
            energy = energy.masked_fill(mask, -np.inf)

        attention = torch.softmax(energy, dim=-1)

        x = torch.einsum("... i j , ... j d -> ... i d", attention, value)

        x = rearrange(x, "b h t d -> b t (h d)")
        x = self.out_attention(x)

        return x


class MLP(nn.Module):
    def __init__(self, embedding_dim, mlp_dim):
        super().__init__()

        self.mlp_layers = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_dim, embedding_dim),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        x = self.mlp_layers(x)

        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim, head_num, mlp_dim):
        super().__init__()

        self.multi_head_attention = MultiHeadAttention(embedding_dim, head_num)
        self.mlp = MLP(embedding_dim, mlp_dim)

        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        _x = self.multi_head_attention(x)
        _x = self.dropout(_x)
        x = x + _x
        x = self.layer_norm1(x)
        
        _x = self.mlp(x)
        x = x + _x
        x = self.layer_norm2(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embedding_dim, head_num, mlp_dim, block_num=12):
        super().__init__()

        self.layer_blocks = nn.ModuleList(
            [TransformerEncoderBlock(embedding_dim, head_num, mlp_dim) for _ in range(block_num)])

    def forward(self, x):
       
        skip_connection_index = [3, 6, 9, 12]
        skip_connections = []

        for idx,layer_block in enumerate(self.layer_blocks):
            x = layer_block(x)
            if (idx+1) in skip_connection_index:
                    skip_connections.append(x[:,1:,:])
        x=x[:, 1:, :]    
        return x,skip_connections


class ViT_M(nn.Module):
    def __init__(self, img_dim, in_channels, embedding_dim, head_num, mlp_dim,
                 block_num, patch_dim, classification=False, num_classes=1):
        super().__init__()
        self.img_dim=img_dim
        self.patch_dim = patch_dim
        self.classification = classification
        self.num_tokens = (img_dim // patch_dim) ** 2
        self.token_dim = in_channels * (patch_dim ** 2)
        self.embedding_dim = embedding_dim

        self.projection = nn.Linear(self.token_dim, embedding_dim)
        self.embedding = nn.Parameter(torch.rand(self.num_tokens + 1, embedding_dim))

        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

        self.dropout = nn.Dropout(0.1)

        self.transformer = TransformerEncoder(embedding_dim, head_num, mlp_dim, block_num)

          
        if self.classification:
          self.linear_classifier = nn.Sequential(nn.Linear(self.token_dim, num_classes), nn.Softmax(dim=-1))

        else:  
            self.linear_output = nn.Sequential(
            nn.Linear(self.embedding_dim, self.patch_dim*self.patch_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.patch_dim*self.patch_dim, self.patch_dim*self.patch_dim),
            nn.Dropout(0.1))


    def forward(self, x):
        img_patches = rearrange(x,
                                'b c (patch_x x) (patch_y y) -> b (x y) (patch_x patch_y c)',
                                patch_x=self.patch_dim, patch_y=self.patch_dim)

        batch_size, tokens, _ = img_patches.shape

        project = self.projection(img_patches).to(self.device)
        token = repeat(self.cls_token, 'b ... -> (b batch_size) ...',
                       batch_size=batch_size)

        patches = torch.cat([token, project], dim=1)
        patches += self.embedding[:tokens + 1, :]

        out = self.dropout(patches)
        out,skip_connections = self.transformer(out)

        if self.classification:
            out=self.linear_classifier(out[:, 0, :]) 
        else: 
            out = self.linear_output(out)
            out = patch_to_image(out,self.img_dim) 

        return out

if __name__ == '__main__':
    vit = ViT_M(img_dim=256,
              in_channels=3,
              patch_dim=16,
              embedding_dim=768,
              block_num=6,
              head_num=4,
              mlp_dim=1024)
    
    print(sum(p.numel() for p in vit.parameters()))
    print(vit(torch.rand(2, 3, 256, 256))[0].shape)