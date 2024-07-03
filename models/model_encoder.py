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


class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    """
    def __init__(self, scale_value=1.0, bias_value=0.0,
        scale_learnable=True, bias_learnable=True, 
        mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
            requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
            requires_grad=bias_learnable)
    def forward(self, x):
        return self.scale * self.relu(x)**2 + self.bias


class Attention(nn.Module):
    """
    Vanilla self-attention from Transformer: https://arxiv.org/abs/1706.03762.
    Modified from timm.
    """
    def __init__(self, dim, head_dim=32, num_heads=None, qkv_bias=False,
        attn_drop=0., proj_drop=0., proj_bias=False, **kwargs):
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
        return x


class SepConv(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """
    def __init__(self, dim, expansion_ratio=2,act1_layer=StarReLU,act2_layer=nn.Identity,bias=False,kernel_size=7,padding=3,**kwargs, ):
        super().__init__()

        med_channels    = int(expansion_ratio * dim)
        
        self.pwconv1    = nn.Linear(dim, med_channels, bias=bias)

        self.act1       = act1_layer()   # Relu
        
        self.dwconv     = nn.Conv2d(
                                    med_channels, med_channels, kernel_size=kernel_size,
                                    padding=padding, groups=med_channels, bias=bias) # depthwise conv
        
        self.act2       = act2_layer()
        self.pwconv2    = nn.Linear(med_channels, dim, bias=bias)

    def forward(self, x):
        x = self.pwconv1(x)
        x = self.act1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.act2(x)
        x = self.pwconv2(x)
        return x


class ConvBlock(nn.Module):
    """ ConvNeXtV2 Block.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, drop=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding='same', groups=dim) # depthwise conv

        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop) if drop > 0. else nn.Identity()

    def forward(self, x):
        
        input = x
        x = x.permute(0, 3, 1, 2) # (N, C, H, W) -> (N, H, W, C)
        x = self.dwconv(x)#self.dwconv2(x)+self.dwconv3(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.norm(input) + self.drop_path(x)
        return x

class Downsampling(nn.Module):
    """
    Downsampling implemented by a layer of convolution.
    """
    def __init__(self, in_channels, out_channels, 
        kernel_size, stride=1, padding=0, 
        pre_norm=None, post_norm=None, pre_permute=False):
        super().__init__()
        self.pre_norm = pre_norm(in_channels) if pre_norm else nn.Identity()
        self.pre_permute = pre_permute
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                              stride=stride, padding=padding)
        self.post_norm = post_norm(out_channels) if post_norm else nn.Identity()

    def forward(self, x):
        x = self.pre_norm(x)
        if self.pre_permute:
            # if take [B, H, W, C] as input, permute it to [B, C, H, W]
            x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1) # [B, C, H, W] -> [B, H, W, C]
        x = self.post_norm(x)
        return x

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


DOWNSAMPLE_LAYERS_FOUR_STAGES = [partial(Downsampling,
            kernel_size=7, stride=4, padding=2,
            post_norm=partial(LayerNormGeneral, bias=False, eps=1e-6)
            )] + \
            [partial(Downsampling,
                kernel_size=3, stride=2, padding=1, 
                pre_norm=partial(LayerNormGeneral, bias=False, eps=1e-6), pre_permute=True
            )]*3


class EncoderBlock(nn.Module):
    """
    Implementation of one MetaFormer block.
    """
    def __init__(self, dim,
                 token_mixer=nn.Identity,cblock=ConvBlock,
                 norm_layer=nn.LayerNorm,
                 drop=0., drop_path=0.,
                 layer_scale_init_value=None, res_scale_init_value=None
                 ):

        super().__init__()

        self.norm1          = norm_layer(dim)
        self.token_mixer    = token_mixer(dim=dim, drop=drop)
        self.drop_path1     = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale1   = Scale(dim=dim, init_value=layer_scale_init_value) if layer_scale_init_value else nn.Identity()
        
        self.res_scale1     = Scale(dim=dim, init_value=res_scale_init_value) if res_scale_init_value else nn.Identity()

        self.norm2          = norm_layer(dim)
        self.Cblock         = cblock(dim=dim, drop=0)

        self.drop_path2     = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.layer_scale2   = Scale(dim=dim, init_value=layer_scale_init_value) if layer_scale_init_value else nn.Identity()
        self.res_scale2     = Scale(dim=dim, init_value=res_scale_init_value) if res_scale_init_value else nn.Identity()
        
    def forward(self, x):
        x = self.res_scale1(x) + self.layer_scale1(self.drop_path1(self.token_mixer(self.norm1(x))))

        x = self.res_scale2(x) + self.layer_scale2(self.drop_path2(self.Cblock(self.norm2(x))))

        return x
    


class Encoder(nn.Module):

    def __init__(self, in_chans=3,  
                 depths=[2, 2, 6, 2],
                 dims=[64, 128, 320, 512],
                 downsample_layers=DOWNSAMPLE_LAYERS_FOUR_STAGES,
                 token_mixers=nn.Identity,
                 conv_blocks=ConvBlock,

                 norm_layers=partial(LayerNormWithoutBias, eps=1e-6), # partial(LayerNormGeneral, eps=1e-6, bias=False),
                 drop_path_rate=0.,
                 layer_scale_init_values=None,
                 res_scale_init_values=[None, None, 1.0, 1.0],
                 output_norm=partial(nn.LayerNorm, eps=1e-6), 
                 **kwargs,
                 ):
        super().__init__()

        if not isinstance(depths, (list, tuple)):
            depths = [depths] # it means the model has only one stage
        if not isinstance(dims, (list, tuple)):
            dims = [dims]

        num_stage      = len(depths)
        self.num_stage = num_stage

        if not isinstance(downsample_layers, (list, tuple)):
            downsample_layers = [downsample_layers] * num_stage
        
        down_dims = [in_chans] + dims
        self.downsample_layers = nn.ModuleList([downsample_layers[i](down_dims[i], down_dims[i+1]) for i in range(num_stage)])
        
        if not isinstance(token_mixers, (list, tuple)):
            token_mixers = [token_mixers] * num_stage

        if not isinstance(conv_blocks, (list, tuple)):
                    conv_blocks = [conv_blocks] * num_stage

        if not isinstance(norm_layers, (list, tuple)):
            norm_layers = [norm_layers] * num_stage


        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        if not isinstance(layer_scale_init_values, (list, tuple)):
            layer_scale_init_values = [layer_scale_init_values] * num_stage
        if not isinstance(res_scale_init_values, (list, tuple)):
            res_scale_init_values = [res_scale_init_values] * num_stage

        self.stages = nn.ModuleList() # each stage consists of multiple metaformer blocks
        cur = 0

        for i in range(num_stage):
            stage = nn.Sequential(
                *[EncoderBlock(  dim=dims[i],
                                    token_mixer=token_mixers[i],
                                    cblock=conv_blocks[i],
                                    norm_layer=norm_layers[i],
                                    drop_path=dp_rates[cur + j],
                                    layer_scale_init_value=layer_scale_init_values[i],
                                    res_scale_init_value=res_scale_init_values[i],        ) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = output_norm(dims[-1])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'norm'}

    def get_features(self, x):
        out=[]
        for i in range(self.num_stage):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            out_ = x.permute(0, 3, 1, 2)
            out.append(out_)
        return self.norm(x.mean([1, 2])),out

    def forward(self, x):
        x,features = self.get_features(x)
        return x,features


def encoder_function(config_res,training_mode=None,pretrained=False,**kwargs):

    model = Encoder(
        depths=[1, 1,1,1],
        dims=[64, 128, 320, 512],
        token_mixers=[Attention, Attention, Attention, Attention],
        **kwargs)
    

    if training_mode=="ssl_pretrained":
        
        if torch.cuda.is_available():
            ML_DATA_OUTPUT = os.environ["ML_DATA_OUTPUT"]
        else:
            ML_DATA_OUTPUT = os.environ["ML_DATA_OUTPUT_LOCAL"]
        checkpoint_path = ML_DATA_OUTPUT+str(model.__class__.__name__)+"["+str(config_res)+"]"
        
        if os.path.exists(checkpoint_path):
            if pretrained == True:
                print("SSL Pretrained SimCLR Imagenet pretrained weights are being used")
            else:
                print("SSL Pretrained SimCLR with random weights are being used")

            if torch.cuda.is_available():
                model.load_state_dict(torch.load(checkpoint_path))
            else: 
                model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
                
        else:
            raise Exception('No Model to Load for ssl_pretrained')
        return model


    elif training_mode=="supervised":
        if pretrained:
            state_dict = torch.hub.load_state_dict_from_url(
                url= model.default_cfg['url'], map_location="cpu", check_hash=True)
            model.load_state_dict(state_dict)
            print("Imagenet Pretrained weights are being used")

        else:
            print("random weights are being used...")
        return model

    elif training_mode=="ssl":
        if pretrained:
            state_dict = torch.hub.load_state_dict_from_url(
                url= model.default_cfg['url'], map_location="cpu", check_hash=True)
            model.load_state_dict(state_dict)
            print("Imagenet Pretrained weights are being used")

        else:
            print("random weights are being used...")
        return model

def convformer_s18_in21ft1k(pretrained=True, **kwargs):
    model = Encoder(
        depths=[3, 3, 9, 3],
        dims=[64, 128, 320, 512],
        token_mixers=[SepConv, SepConv, Attention, Attention],
        **kwargs)

    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url= model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)

    return model

if __name__ == '__main__':
    # Loading data

    model = convformer_s18_in21ft1k()
    #model2 = VİT_NN(images_dim=128,input_channel=3, token_dim=768,  n_heads=4, mlp_layer_size=1024, t_blocks=12, patch_size=8,classification=False)

    #print(model(torch.rand(2, 3, 224, 224))[0].shape)
    print(model(torch.rand(2, 3, 256, 256))[0].shape)