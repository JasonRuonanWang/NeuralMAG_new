import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

def manual_convtranspose2d(input, weight, stride: int, padding: int):
    output = F.conv_transpose2d(input, weight, stride=stride, padding=padding)
    return output
def manual_conv2d(input, weight, stride: int, padding: int):
    output = F.conv2d(input, weight, stride=stride, padding=padding)
    return output
    
class downsample(nn.Module):
    def __init__(self, in_channels, out_channels, kr, s, p, drop_out):
        super(downsample, self).__init__()
        self.stride = s
        self.padding = p
        self.pool = nn.AvgPool2d(2, 2)
        self.lrelu = nn.LeakyReLU(0.2, True)
        # 对于channel,h,w进行归一化
        self.norm = nn.GroupNorm(1, out_channels, affine=False)
        self.drop = nn.Dropout(drop_out) if drop_out>0 else nn.Identity()

    def forward(self, x, weight):
        pool_x = self.pool(x)
        lrelu_x = self.lrelu(pool_x)
        conv_x = manual_conv2d(lrelu_x, weight, stride=self.stride, padding=self.padding)
        norm_x = self.norm(conv_x)
        drop_x = self.drop(norm_x)
        return drop_x
    
class upsample(nn.Module):
    def __init__(self, in_channels, out_channels, kr, s, p, drop_out):
        super(upsample, self).__init__()
        self.stride = s
        self.padding = p
        self.relu = nn.ReLU(True)
        self.norm = nn.GroupNorm(1, out_channels, affine=False)
        
    def forward(self, x, weight):
        relu_x = self.relu(x)
        convt_x = manual_convtranspose2d(relu_x, weight, stride=self.stride, padding=self.padding)
        norm_x = self.norm(convt_x)
        return norm_x
        
class upsample_0(nn.Module):
    def __init__(self, in_channels, out_channels, kr, s, p, drop_out):
        super(upsample_0, self).__init__()
        self.up = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kr, stride=s, padding=p),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(drop_out) if drop_out>0 else nn.Identity()
            )
        
    def forward(self, x):
        return self.up(x)

# 32*32
class UNet(nn.Module):
    def __init__(self, kc=16, inc=3, ouc=3):
        super(UNet, self).__init__()
        self.s0 = nn.Conv2d(  inc,   1*kc,  3,1,1)
        self.s = downsample( 1*kc,  1*kc,  3,1,1, drop_out=0.0)

        self.weight_s  = nn.Parameter(torch.zeros(1*kc, 1*kc, 3, 3))
        self.weight_up = nn.Parameter(torch.zeros(2*kc, 1*kc, 4, 4))

        self.up = upsample( 2*kc, 1*kc, 4,2,1, drop_out=0.0)
        self.up0 = upsample_0( 2*kc, 1*kc, 3,1,1, drop_out=0.0)

        self.last_Conv = nn.Sequential(
            nn.ConvTranspose2d(kc+inc, 1*kc, 3,1,1),
            nn.BatchNorm2d(1*kc),
            nn.Tanh(),
            nn.Conv2d(1*kc, ouc, 1,1,0),
        )

        print("c_gn_test")
        self.init_weight()
    def init_weight(self):
        nn.init.kaiming_normal_(self.weight_s, mode='fan_out')
        nn.init.kaiming_normal_(self.weight_up, mode='fan_in')
        for w in self.modules():
            #判断层并且传参
            if isinstance(w, nn.Conv2d):
                #权重初始化
                nn.init.kaiming_normal_(w.weight, mode='fan_out')
                if w.bias is not None:
                    nn.init.zeros_(w.bias)
            elif isinstance(w, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(w.weight, mode='fan_in')
            elif isinstance(w, nn.BatchNorm2d):
                nn.init.ones_(w.weight)
                nn.init.zeros_(w.bias)
    
    def forward(self, x):
        # 获取x.size(2)中2的因子数量
        dim_size = x.size(2)
        layer_num = 0
        while (1 << layer_num) < dim_size:
            layer_num += 1
        
        layer_out: List[torch.Tensor] = [torch.tensor(0,device=torch.device("cuda:0"))] * (layer_num + 2)
        layer_out[0] = self.s0(x)
        for i in range(layer_num):
            layer_out[i+1] = self.s(layer_out[i], self.weight_s)

        layer_out[layer_num+1] = layer_out[layer_num]
        for i in range(layer_num):
            layer_out[layer_num-i] = self.up(torch.cat([layer_out[layer_num-i+1], layer_out[layer_num-i]], dim=1), self.weight_up)
        
        up_0 = self.up0(torch.cat([layer_out[1],layer_out[0]], dim=1))
        out  = self.last_Conv(torch.cat([up_0,x],dim=1))
        return torch.where(out >= 0, torch.exp(out)-1, -torch.exp(-out)+1)
    
