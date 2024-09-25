##IMPORT 
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import os
from functools import partial
import sys
#from transformers import ViTImageProcessor, ViTForImageClassification



class Scale(nn.Module):
    """
    Scale vector by element multiplications.
    """
    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        return x * self.scale
        


class LayerNormGeneral(nn.Module):

    def __init__(self, affine_shape=None, normalized_dim=(-1, ), scale=True, 
        bias=True, eps=1e-5):
        super().__init__()
        self.normalized_dim = normalized_dim
        self.use_scale = scale
        self.use_bias = bias
        self.weight = nn.Parameter(torch.ones(affine_shape)) if scale else None
        self.bias = nn.Parameter(torch.zeros(affine_shape)) if bias else None
        self.eps = eps

    def forward(self, x):
        c = x - x.mean(self.normalized_dim, keepdim=True)
        s = c.pow(2).mean(self.normalized_dim, keepdim=True)
        x = c / torch.sqrt(s + self.eps)
        if self.use_scale:
            x = x * self.weight
        if self.use_bias:
            x = x + self.bias
        return x



class LayerNormWithoutBias(nn.Module):
    """
    Equal to partial(LayerNormGeneral, bias=False) but faster, 
    because it directly utilizes otpimized F.layer_norm
    """
    def __init__(self, normalized_shape, eps=1e-5, **kwargs):
        super().__init__()
        self.eps = eps
        self.bias = None
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape
    def forward(self, x):
        return F.layer_norm(x, self.normalized_shape, weight=self.weight, bias=self.bias, eps=self.eps)


class Mlp(nn.Module):
    """ MLP as used in MetaFormer models, eg Transformer, MLP-Mixer, PoolFormer, MetaFormer baslines and related networks.
    Mostly copied from timm.
    """
    def __init__(self, dim, mlp_ratio=4, out_features=None, act_layer=nn.GELU, drop=0., bias=False, **kwargs):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        drop_probs = (drop,drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.norm = nn.LayerNorm(dim)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.norm(x)
        x = self.drop2(x)
        return x.permute(0, 3, 1, 2)

class Attention(nn.Module):
    """
    Vanilla self-attention from Transformer: https://arxiv.org/abs/1706.03762.
    Modified from timm.
    """
    def __init__(self, dim, head_dim=32, num_heads=None, qkv_bias=False,
        attn_drop=0.2, proj_drop=0.2, proj_bias=False, **kwargs):
        super().__init__()

        self.head_dim = head_dim
        self.scale = head_dim ** -0.5   #embed_dim/head_number = head

        self.num_heads = num_heads if num_heads else dim // head_dim

        if self.num_heads == 0:
            self.num_heads = 1
        
        self.attention_dim = self.num_heads * self.head_dim

        self.qkv        = nn.Linear(dim, self.attention_dim * 3, bias=qkv_bias)
        self.attn_drop  = nn.Dropout(attn_drop)
        self.proj       = nn.Linear(self.attention_dim, dim, bias=proj_bias)
        self.proj_drop  = nn.Dropout(proj_drop)

        
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        N = H * W
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, self.attention_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x.permute(0, 3, 1, 2)


class upsampling(nn.Module):

    def __init__(self, in_channels, out_channels, 
        kernel_size, stride=1, padding=0, 
        pre_norm=None, post_norm=None, pre_permute=False):
        super().__init__()
        self.pre_norm = pre_norm(in_channels) if pre_norm else nn.Identity()
        self.pre_permute = pre_permute
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        #self.dilatedconv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding="same",dilation=3)
        #self.dilatedconv = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=3,padding="same") # 15x15

        self.norm = nn.LayerNorm(out_channels)
        #self.norm = nn.BatchNorm2d(out_channels)

        self.act= nn.GELU()
        self.up   = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.post_norm = post_norm(out_channels) if post_norm else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x= self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.act(x)
        x= self.up(x)
        return x

class ConvBlock(nn.Module):
    """ 

    """
    def __init__(self, dim, drop=0.):
        super().__init__()

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding='same',groups=dim) # depthwise conv

        #self.norm = nn.BatchNorm2d(dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop) if drop > 0. else nn.Identity()

    def forward(self, x):
        
        x = self.dwconv(x)#self.dwconv2(x)+self.dwconv3(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.act(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop_path(x)
        x = x.permute(0, 3, 1, 2)
        return x




class SepConv(nn.Module):
    """ 

    """
    def __init__(self, dim, drop=0.):
        super().__init__()

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding='same',groups=dim) # depthwise conv

        #self.norm = nn.BatchNorm2d(dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop) if drop > 0. else nn.Identity()

    def forward(self, x):
        
        x = self.dwconv(x)#self.dwconv2(x)+self.dwconv3(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.act(x)

        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop_path(x)
        x = x.permute(0, 3, 1, 2)
        return x

UPSAMPLE_LAYERS_FOUR_STAGES =[partial(upsampling,
                kernel_size=3, padding='same', 
                pre_norm=partial(LayerNormGeneral, bias=False, eps=1e-6)
            )]*3


class DecoderBlocks(nn.Module):
    """
    Implementation of one MetaFormer block.
    """
    def __init__(self, dim,
                 token_mixer=nn.Identity,
                 cblock=Mlp,
                 #norm_layer=nn.LayerNorm,
                 drop=0., drop_path=0.,
                 layer_scale_init_value=None, res_scale_init_value=None
                 ):

        super().__init__()

        #self.norm1          = norm_layer(dim)
        self.token_mixer    = token_mixer(dim=dim, drop=drop)
        self.drop_path1     = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        #self.layer_scale1   = Scale(dim=dim, init_value=layer_scale_init_value) if layer_scale_init_value else nn.Identity()

        #self.res_scale1     = Scale(dim=dim, init_value=res_scale_init_value) if res_scale_init_value else nn.Identity()

        #self.norm2          = norm_layer(dim)
        self.drop_path2     = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        #if self.token_mixer.__class__.__name__=='Attention':
        self.Cblock         = cblock(dim=dim, drop=0)

        #self.layer_scale2   = Scale(dim=dim, init_value=layer_scale_init_value) if layer_scale_init_value else nn.Identity()
        #self.res_scale2     = Scale(dim=dim, init_value=res_scale_init_value) if res_scale_init_value else nn.Identity()
        
    def forward(self, x):
        x = x + self.drop_path1(self.token_mixer(x))
        if self.token_mixer.__class__.__name__=='Attention':
            x = x + self.drop_path2(self.Cblock(x))
        else:
            x = x + self.drop_path2(self.token_mixer(x))

        return x
    

class Decoder(nn.Module):

    def __init__(self, num_classes=1000, 
                 depths=[1,1,1,1],
                 dims=[512,256,128,64],
                 up_layers=UPSAMPLE_LAYERS_FOUR_STAGES,
                 token_mixers=nn.Identity,
                 #norm_layers=partial(LayerNormWithoutBias, eps=1e-6), # partial(LayerNormGeneral, eps=1e-6, bias=False),
                 drop_path_rate=0.,
                 head_dropout=0.0, 
                 layer_scale_init_values=None,
                 res_scale_init_values=[None, None, 1.0, 1.0],
                 head_fn=nn.Linear,
                 **kwargs,
                 ):
        super().__init__()
        self.num_classes = num_classes

        if not isinstance(depths, (list, tuple)):
            depths = [depths] # it means the model has only one stage
        if not isinstance(dims, (list, tuple)):
            dims = [dims]

        num_stage      = len(depths)
        self.num_stage = num_stage

        if not isinstance(up_layers, (list, tuple)):
            up_layers = [up_layers] * num_stage
        
        self.up_layers = nn.ModuleList([up_layers[i](dims[i], dims[i+1]) for i in range(num_stage-1)])
        self.up_layers.append(up_layers[-1](dims[-1], dims[-1]))

        if not isinstance(token_mixers, (list, tuple)):
            token_mixers = [token_mixers] * num_stage


        #if not isinstance(norm_layers, (list, tuple)):
        #    norm_layers = [norm_layers] * num_stage
        
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        if not isinstance(layer_scale_init_values, (list, tuple)):
            layer_scale_init_values = [layer_scale_init_values] * num_stage
        if not isinstance(res_scale_init_values, (list, tuple)):
            res_scale_init_values = [res_scale_init_values] * num_stage

        self.stages = nn.ModuleList() # each stage consists of multiple  blocks
        cur = 0

        for i in range(num_stage):
            stage = nn.Sequential(
                *[DecoderBlocks(  dim=dims[i],
                                    token_mixer=token_mixers[i],
                                    #norm_layer=norm_layers[i],
                                    drop_path=dp_rates[cur + j],
                                    layer_scale_init_value=layer_scale_init_values[i],
                                    res_scale_init_value=res_scale_init_values[i],        ) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]


        if head_dropout > 0.0:
            self.head = head_fn(dims[-1], num_classes, head_dropout=head_dropout)
        else:
            self.head = head_fn(dims[-1], num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'norm'}

    def get_features(self, x, s):

        for i in range(self.num_stage):

            x = self.stages[i](x)
            x = self.up_layers[i](x)

            if i <3:
                x = s[i] + x

        return x

    def forward(self, x,s):
        decoder_output = self.get_features(x,s)
        return decoder_output


def decoder_function(pretrained=False,**kwargs):

    model = Decoder(
        depths=[1,1,1,1],
        dims=[512,256,128,64],
        token_mixers=[Attention, Attention, SepConv, SepConv],
        **kwargs)
    

    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url= model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)

    return model

if __name__ == '__main__':
    # Loading data

    model = decoder_function()
    #model2 = VİT_NN(images_dim=128,input_channel=3, token_dim=768,  n_heads=4, mlp_layer_size=1024, t_blocks=12, patch_size=8,classification=False)

    #print(model(torch.rand(2, 3, 224, 224))[0].shape)
    print(model(torch.rand(2, 3, 256, 256))[0].shape)