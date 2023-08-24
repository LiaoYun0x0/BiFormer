import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from detectron2.detectron2.layers import FrozenBatchNorm2d, ShapeSpec, get_norm
_NORM = 'BN'

class Conv2d_BN(nn.Module):
    """Convolution with BN module."""
    def __init__(
            self,
            in_ch,
            out_ch,
            kernel_size=1,
            stride=1,
            pad=0,
            dilation=1,
            groups=1,
            bn_weight_init=1,
            act_layer=None,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.conv = torch.nn.Conv2d(in_ch,
                                    out_ch,
                                    kernel_size,
                                    stride,
                                    pad,
                                    dilation,
                                    groups,
                                    bias=False
        )
        self.bn = get_norm(_NORM, out_ch)
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Note that there is no bias due to BN
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))

        self.act_layer = act_layer() if act_layer is not None else nn.Identity()

    def forward(self, x):
        # ain = self.in_ch
        # aout = self.out_ch

        x = self.conv(x)
        x = self.bn(x)
        x = self.act_layer(x)
        
        return x


class Dila_PRM(nn.Module):
    def __init__(self,in_embed,out_embed,kernel_size=4,downsample_ratio=1,dilations=[2,4,6],
                 fusion='cat'):
        super().__init__()
        
        self.dilations = dilations
        self.in_embed = in_embed
        # self.in_embeds=[self.in_embed,self.in_embed//2,self.in_embed//4]
        # self.out_embeds=[self.in_embed,self.in_embed//2,self.in_embed//4]
        self.out_embed = out_embed
        self.fusion = fusion
        self.kernel_size = kernel_size
        self.stride = downsample_ratio
        #self.out_size = img_size//downsample_ratio
        
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=self.in_embed, 
                        out_channels=self.in_embed, 
                        kernel_size=self.kernel_size, 
                        stride=self.stride, 
                        # padding=math.ceil(((self.kernel_size-1)*self.dilations[idx] + 1 - self.stride) / 2), 
                        padding=math.ceil(((self.kernel_size-1)*self.dilations[idx])/2),
                        dilation=self.dilations[idx]),
                    # nn.BatchNorm2d(self.in_embed),
                    nn.GELU()
                ) for idx in range(len(self.dilations))
            ]
        )
        
        if self.fusion == 'cat':
            self.outchans = self.in_embed * len(self.dilations)
            '''这里可以改一改，不同尺度的特征维度尽量少'''
            #self.aggerate = Conv2d_BN(self.in_embed*len(self.dilations),self.in_embed,act_layer=nn.Hardswish)
            self.aggerate = Conv2d_BN(self.in_embed*len(self.dilations),self.out_embed,act_layer=nn.Hardswish)
    
    def forward(self,x):
        B,C,H,W = x.shape   #1,3,320,320
        
        out = self.convs[0](x).unsqueeze(dim=-1)
        for i in range(1,len(self.dilations)):
            cur_out = self.convs[i](x).unsqueeze(dim=-1)
            
            out = torch.cat((cur_out,out),dim=-1)
            
        B, C, W, H, N = out.shape
        #cur_size = (W,H)
        if self.fusion=='cat':
            out = out.permute(0,4,1,2,3).reshape(B,N*C,W,H)
            out = self.aggerate(out)
            # out = out.flatten(2).transpose(1,2) #B,N,C
        
        return out