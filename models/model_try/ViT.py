##IMPORT 
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
import matplotlib.pyplot as plt 
#from transformers import ViTImageProcessor, ViTForImageClassification

##PATCHFY FUNCTION & POSITIONAL_EMB

def patching(images,patch_size):
    
    img_patches = rearrange(images, 'b c (patch_x h) (patch_y w) -> b (h w) (patch_x patch_y c)',
                                    patch_x=patch_size, patch_y=patch_size)

    return img_patches

def get_positional_embeddings(sequence_length, d):
    
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
    return result

# CLASSES VIT

class ViT_c (nn.Module):
    def __init__(self, images_dim, input_channel, token_dim, n_heads, mlp_layer_size, t_blocks, patch_size, classification,out_dim=10):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.images_dim = images_dim
        self.c = input_channel
        self.patch_size = patch_size
        self.t_blocks   = t_blocks
        self.n_heads    = n_heads
        self.token_dim  = token_dim
        self.number_token = (self.images_dim//self.patch_size)**2+1
        self.classification = classification
        
        self.mlp_layer_size     =   mlp_layer_size

        self.linear_map         =   nn.Linear(self.c*(patch_size**2),token_dim)
        self.class_token        =   nn.Parameter(torch.rand(1, token_dim))
        self.blocks             =   nn.ModuleList([ViTBlock(token_dim, self.number_token, mlp_layer_size, n_heads) for _ in range(t_blocks)])
        self.output_pr          =   nn.Softmax()
        self.positional_embeddings = get_positional_embeddings(self.number_token,self.token_dim).to(self.device)

        if self.classification:
            self.linear_classifier = nn.Sequential(nn.Linear(self.token_dim, out_dim), nn.Softmax(dim=-1))

    def forward(self, images):

        self.n_images,self.c,self.h_image,self.w_image = images.shape
        all_class_token = self.class_token.repeat(self.n_images, 1, 1).to(self.device)
        patches     = patching(images, self.patch_size).to(self.device)
        linear_emb  = self.linear_map(patches)
        tokens      = torch.cat((all_class_token,linear_emb),dim=1)
        out         = tokens    +  self.positional_embeddings.repeat(self.n_images, 1, 1)    # positional embeddings will be added
        
        skip_connection_index = [3, 6, 9, 12]
        skip_connections = []

        for idx , block in enumerate(self.blocks):
            out = block(out)
            # out1=out
            # x = rearrange(out1[:,1:,:], "b (x y) c -> b c x y", x=8, y=8)  # 2x 1024 x 8 x 8
            # plt.subplot(3, 5, 6)
            # plt.imshow(x[1,1].detach().numpy(),cmap='gray')

            if (idx+1) in skip_connection_index:
                skip_connections.append(out[:,1:,:])

        if self.classification:
            self.linear_classifier(out[:, 0, :]) 
        else: 
            out=out[:, 1:, :]     

        return out,skip_connections

class ViTBlock(nn.Module):

    def __init__(self, token_dim, n_tokens, mlp_layer_size, num_heads):
        super().__init__() 

        self.token_dim      = token_dim
        self.num_heads      = num_heads
        self.mlp_layer_size = mlp_layer_size
        self.n_tokens       = n_tokens

        self.layer_norm1    = nn.LayerNorm(token_dim)
        self.layer_norm2    = nn.LayerNorm(token_dim) 
        self.msa            = MSA_Module(token_dim, n_tokens, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(token_dim, mlp_layer_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_layer_size, token_dim),
            nn.Dropout(0.1))
        self.dropout = nn.Dropout(0.1)


    def forward(self, x):
        # input = self.layer_norm1(x)
        # out = x + self.msa(input)
        # out = self.layer_norm2(out)
        # out = out + self.mlp(out)

        out = x + self.dropout(self.msa(x))
        out = self.layer_norm1(out)
        out = out + self.mlp(out)
        out = self.layer_norm2(out)
        
        return out
        
class MSA_Module(nn.Module):

    def __init__(self, token_dim, n_tokens, n_heads):
        super().__init__() 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.n_heads    = n_heads
        self.token_dim  = token_dim
        self.n_tokens   = n_tokens

        self.q_layers   = nn.ModuleList([nn.Linear(token_dim,token_dim) for _ in range(n_heads)])
        self.k_layers   = nn.ModuleList([nn.Linear(token_dim,token_dim) for _ in range(n_heads)])
        self.v_layers   = nn.ModuleList([nn.Linear(token_dim,token_dim) for _ in range(n_heads)])
        self.softmax    = nn.Softmax(dim=-1)
        self.linear_map = nn.Linear(n_heads * n_tokens,n_tokens)   

    def forward (self, tokens):
        
        self.n, self.number_tokens, self.token_size = tokens.shape
        result = torch.zeros(self.n, self.number_tokens*self.n_heads, self.token_size).to(self.device)

        for idx,token in enumerate(tokens):   # 128 batch. each of 65x16*16*3, token size : 50x8   --> 50x8            
            concat      = torch.zeros(self.n_heads, self.number_tokens, self.token_size)        
            for head in range(self.n_heads):        # number of heads : 4
                q_linear = self.q_layers[head]      # linear (512x512)  
                k_linear = self.k_layers[head]
                v_linear = self.v_layers[head]

                q  = q_linear(token)   # 65x512 
                k  = k_linear(token)   # 65x512 
                v  = v_linear(token)   # 65x512 
                                
                mat_mul = ((q @ k.T ) / ((self.number_tokens-1)**0.5))  # 65x65
                attention_mask  = self.softmax(mat_mul)   # 65x65

                attention        = attention_mask @ v       # 65*65 x 65*512   --> 65x512
                concat[head,:,:]  = attention             # 4x65x512
            result[idx,:,:]=torch.flatten(input=concat, start_dim=0, end_dim=1)
        result=torch.transpose(result,1,2) 
        result=self.linear_map(result)
        return torch.transpose(result,1,2)


class MyMSA(nn.Module):
    def __init__(self, d,token_dim,n_heads=2):
        super(MyMSA, self).__init__()
        self.d = d
        self.n_heads = n_heads

        assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"

        d_head = int(d / n_heads)
        self.q_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.k_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.v_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.d_head = d_head
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        # Sequences has shape (N, seq_length, token_dim)
        # We go into shape    (N, seq_length, n_heads, token_dim / n_heads)
        # And come back to    (N, seq_length, item_dim)  (through concatenation)
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]

                seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

                attention = self.softmax(q @ k.T / (self.d_head ** 0.5))
                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])

# class VİT_NN(nn.Module):
#     def __init__(self,images_dim, input_channel, token_dim, n_heads, mlp_layer_size, t_blocks, patch_size, classification,out_dim=10):
#         super().__init__()
        
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.images_dim = images_dim
#         self.c = input_channel
#         self.token_dim  = token_dim
#         self.patch_size = patch_size
#         self.t_blocks   = t_blocks
#         self.n_heads    = n_heads
#         self.mlp_layer_size     =   mlp_layer_size
#         self.out_dim = out_dim

#         self.number_token = (self.images_dim//self.patch_size)**2+1
#         self.classification = classification

#         self.trans_encoder_layers = []
#         self.linear_map         =   nn.Linear(self.c*(patch_size**2),token_dim)
#         self.positional_embeddings = get_positional_embeddings(self.number_token,self.token_dim).to(self.device)

#         # self.positions = torch.arange(start=0, end=self.number_token, step=1, dtype=torch.int32)
#         # self.pos_embed = nn.Embedding(self.number_token,self.token_dim)
        
#         for i in range(self.t_blocks):
#             layer = nn.TransformerEncoderLayer(
#                 d_model = self.token_dim,
#                 nhead   = self.n_heads ,
#                 dim_feedforward = self.mlp_layer_size ,
#                 dropout = 0,
#                 activation = nn.GELU(),
#                 batch_first = True).to( self.device)
#             self.trans_encoder_layers.append(layer)

#         self.class_token        =   nn.Parameter(torch.rand(1, token_dim)).to( self.device)

#         if self.classification:
#             self.linear_classifier = nn.Sequential(nn.Linear(self.token_dim, out_dim), nn.Softmax(dim=-1))

#     def forward(self,images):

#         n, c, h, w = images.shape
#         patches = patching(images,self.patch_size)
        
#         # Running linear layer tokenization
#         # Map the vector corresponding to each patch to the hidden size dimension
#         tokens = self.linear_map(patches)
        
#         # Adding classification token to the tokens
#         tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)
        
#         # Adding positional embedding
#         out = tokens   +  self.positional_embeddings.repeat(n, 1, 1)

#         # positions = self.positions
#         # pos_embed = self.pos_embed(positions)   ## [256, 768]
#         # out2 = tokens + pos_embed
        
        
#         # plt.figure()
#         # plt.subplot(1,2,1)
#         # plt.imshow(pos_embed[1:].detach().numpy(),cmap='gray')
#         # plt.subplot(1,2,2)
#         # plt.imshow(self.positional_embeddings.repeat(n, 1, 1)[0,1:].detach().numpy(),cmap='gray')

#         skip_connection_index = [3, 6, 9, 12]
#         skip_connections = []

#         for i in range(self.t_blocks):
#             layer = self.trans_encoder_layers[i]
#             out = layer(out)

#             if (i+1) in skip_connection_index:
#                 skip_connections.append(out[:,1:,:])

#         if self.classification:
#             self.linear_classifier(out[:, 0, :]) 
#         else: 
#             out=out[:, 1:, :]     

#         return out,skip_connections



if __name__ == '__main__':
    # Loading data

    model = ViT_c(images_dim=128,input_channel=3, token_dim=768,  n_heads=4, mlp_layer_size=1024, t_blocks=12, patch_size=8,classification=False)
    #model2 = VİT_NN(images_dim=128,input_channel=3, token_dim=768,  n_heads=4, mlp_layer_size=1024, t_blocks=12, patch_size=8,classification=False)

    print(model(torch.rand(2, 3, 128, 128))[0].shape)
    #print(model(torch.rand(2, 3, 128, 128))[0].shape)