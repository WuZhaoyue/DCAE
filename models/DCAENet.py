import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.ops import deform_conv2d
from torch import nn
from models.NONLocalBlock2D import *


#####################2D deformable convolution###############
class DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=True):

        super(DeformableConv2d, self).__init__()

        self.padding = padding
        
        self.offset_conv = nn.Conv2d(in_channels, 
                                     2 * kernel_size * kernel_size,
                                     kernel_size=kernel_size, 
                                     stride=stride,
                                     padding=self.padding, 
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        
        self.modulator_conv = nn.Conv2d(in_channels, 
                                     1 * kernel_size * kernel_size,
                                     kernel_size=kernel_size, 
                                     stride=stride,
                                     padding=self.padding, 
                                     bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)
        
        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      bias=bias)
    
    def forward(self, x):

        h, w = x.shape[2:]
        max_offset = max(h, w)/4.

        offset = self.offset_conv(x).clamp(-max_offset, max_offset)
        modulator = 2.*torch.sigmoid(self.modulator_conv(x))
        
        x = deform_conv2d(input=x, 
                          offset=offset, 
                          weight=self.regular_conv.weight, 
                          bias=self.regular_conv.bias, 
                          padding=self.padding,
                          mask=modulator
                         )
        return x

####################################The main designed code#############################
class DCAENet(nn.Module):
    def __init__(self, in_dim, en_dim, kernels):
        super(DCAENet, self).__init__()
        

        self.CNN2d1 = nn.Sequential(
            nn.Conv2d(in_dim, round(in_dim/2), kernel_size=kernels, padding=math.floor(kernels/2)),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(round(in_dim/2)),
            # nn.Dropout(drop_out),
        )
        self.offset1 = DeformableConv2d(round(in_dim/2), round(in_dim/2), kernel_size=kernels, padding=math.floor(kernels/2))
        self.nl_1 = NONLocalBlock2D(round(in_dim/2))

        self.CNN2d2 = nn.Sequential(
            nn.Conv2d(round(in_dim/2), round(in_dim/4), kernel_size=kernels, padding=math.floor(kernels/2)),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(round(in_dim/4)),
            # nn.Dropout(drop_out),
        )
        self.offset2 = DeformableConv2d(round(in_dim/4), round(in_dim/4), kernel_size=kernels, padding=math.floor(kernels/2))
        self.nl_2 = NONLocalBlock2D(round(in_dim/4))

        self.CNN2d3 = nn.Sequential(
            nn.Conv2d(round(in_dim/4), en_dim, kernel_size=1),
        )
        self.nl_3 = NONLocalBlock2D(en_dim)

        self.encoderlayer = nn.Sequential(nn.Softmax(dim=1))
        self.decoderlayer1 = nn.Sequential(
            nn.Conv2d(en_dim, in_dim, kernel_size=(1, 1)),
        )
      
    def forward(self, x, args):
        
        if args.NLC & args.DC:
           x = self.CNN2d1(x) #torch.Size([1, 4, 100, 100])
           x = self.offset1(x)

           x = self.CNN2d2(x)
           x = self.offset2(x)

           x = self.CNN2d3(x)

           x = self.nl_3(x)
           
           encoded = self.encoderlayer(x) #torch.Size([1, 4, 100, 100])
           cons_x = self.decoderlayer1(encoded)
 
           return cons_x
        
        elif args.DC==False & args.NLC==False:
           x = self.CNN2d1(x)

           x = self.CNN2d2(x)

           x = self.CNN2d3(x)

       
           encoded = self.encoderlayer(x) #torch.Size([1, 4, 100, 100])
           cons_x = self.decoderlayer1(encoded)
       
           return cons_x
