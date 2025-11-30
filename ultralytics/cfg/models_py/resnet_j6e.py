# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import numpy as np

from ultralytics.nn.modules import Detect, Detect3D

def check_anchor_order(m):
    # Check anchor order against stride order for YOLOv5 Detect() module m, and correct if necessary
    a = m.anchor_grid.prod(-1).view(-1)  # anchor area
    da = a[-1] - a[0]  # delta a
    ds = m.stride[-1] - m.stride[0]  # delta s
    if da.sign() != ds.sign():  # same order
        print('Reversing anchor order')
        m.anchors[:] = m.anchors.flip(0)
        m.anchor_grid[:] = m.anchor_grid.flip(0)

class Conv(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size,
            stride=1,
            dilation=1,
            groups: int = 1,
            bias: bool = False,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        x = self.conv(x)
        return x

class ConvBN(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size,
            stride=1,
            dilation=1,
            groups: int = 1,
            bias: bool = False,
            mergeBN=False,
    ):
        super().__init__()
        self.mergeBN = mergeBN
        padding = kernel_size // 2
        if not self.mergeBN:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
            self.bn = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=True)

    def forward(self, x):
        x = self.conv(x)
        if not self.mergeBN:
            x = self.bn(x)
        return x


class ConvBNRelu(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size,
            stride=1,
            dilation=1,
            groups: int = 1,
            bias: bool = False,
            relu_layer=nn.ReLU(inplace=True),
            mergeBN=False,
    ):
        super().__init__()
        self.mergeBN = mergeBN
        padding = kernel_size // 2
        if not self.mergeBN:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
            self.bn = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=True)
        self.relu = relu_layer

    def forward(self, x):
        x = self.conv(x)
        if not self.mergeBN:
            x = self.bn(x)
        x = self.relu(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, inplane, outplane, groups = 1, downsample=False,mergeBN=False):
        super(ResBlock, self).__init__()
        self.downsample = downsample
        self.mergeBN = mergeBN
        self.groups = groups
        if self.downsample:
            self.conv_bn_1 = ConvBNRelu(in_channels=inplane, out_channels=outplane, kernel_size=3, stride=2,groups = self.groups,mergeBN=self.mergeBN)
            self.conv_bn_2 = ConvBNRelu(in_channels=outplane, out_channels=outplane, kernel_size=3, stride=1,groups = self.groups,mergeBN=self.mergeBN)
            self.downconv = ConvBNRelu(in_channels=inplane, out_channels=outplane, kernel_size=3, stride=2,groups = self.groups,mergeBN=self.mergeBN)
        else:
            self.conv_bn_1 = ConvBNRelu(in_channels=inplane, out_channels=outplane, kernel_size=3, stride=1,groups = self.groups,mergeBN=self.mergeBN)
            self.conv_bn_2 = ConvBNRelu(in_channels=outplane, out_channels=outplane, kernel_size=3, stride=1,groups = self.groups,mergeBN=self.mergeBN)

    def forward(self, x):
        """Forward function."""
        identity = x
        out = self.conv_bn_2(self.conv_bn_1(x))
        if self.downsample:
            out = out + self.downconv(identity)
        else:
            out = out + identity
        return out
    
class RefineBlock(nn.Module):
    def __init__(self, deeper_in_channels, side_in_channels, out_channels, relu_layer=nn.ReLU(inplace=True), groupnum=1,
                 mergeBN=False):
        super().__init__()
        # self.deconv_1 = nn.ConvTranspose2d(deeper_in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False) ######
        self.deconv_1 = nn.ConvTranspose2d(deeper_in_channels, out_channels, kernel_size=2, stride=2, padding=0,
                                           groups=groupnum, bias=False)
        self.relu = relu_layer
        self.conv_bn_2 = ConvBN(side_in_channels, out_channels, kernel_size=1, groups=groupnum, mergeBN=mergeBN)
        self.conv_bn_3 = ConvBN(out_channels, out_channels, kernel_size=1, groups=groupnum, mergeBN=mergeBN)

        self.res_4 = ResBlock(inplane=out_channels, outplane=out_channels, groups = groupnum,downsample=False,mergeBN=mergeBN)
      
        # self.res_5 = ResBlock(in_channels=out_channels, out_channels=out_channels, in_width=out_channels*2, relu_layer=relu_layer)

    def forward(self, x_deeper, x_side):
        x_deeper = self.deconv_1(x_deeper)
        x_deeper = self.relu(x_deeper)

        x_side = self.conv_bn_2(x_side)
        x_side = self.relu(x_side)

        x = x_deeper + x_side
        x = self.conv_bn_3(x)
        x = self.relu(x)

        x = self.res_4(x)
        # x = self.res_5(x)
        return x

class RefineBlock_DY(nn.Module):
    def __init__(self, deeper_in_channels, side_in_channels, out_channels, relu_layer=nn.ReLU(inplace=True), groupnum=1,
                 mergeBN=False):
        super().__init__()
        # self.deconv_1 = nn.ConvTranspose2d(deeper_in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv_1 = nn.ConvTranspose2d(deeper_in_channels, out_channels, kernel_size=2, stride=2, padding=0,
                                           groups=groupnum, bias=False)
        self.relu = relu_layer
        self.conv_bn_2 = ConvBN(side_in_channels, out_channels, kernel_size=1, groups=groupnum, mergeBN=mergeBN)

    def forward(self, x_deeper, x_side):
        x_deeper = self.deconv_1(x_deeper)
        x_deeper = self.relu(x_deeper)

        x_side = self.conv_bn_2(x_side)
        x_side = self.relu(x_side)

        x = x_deeper + x_side

        return x
class convbase(nn.Module):
    def __init__(self, in_channels=0, out_channels=0, kernel_size = 0, stride=0,padding=0,group=0,deploy = False):
        super(convbase, self).__init__()
        self.deploy = deploy
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=padding, dilation=1, groups=group, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        if self.deploy == False:
            self.bn1 = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True)

    def forward(self, x):
        if self.deploy:
            out = self.conv1(x)
            out = self.relu1(out)
        else:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu1(out)
        return out

class R34TDA4(nn.Module):
    """
    ImgFeExtractor backbone for TDA4 BEV Model Deploy
    """
    def __init__(self, n_class,mergeBN = False):
        super(R34TDA4, self).__init__()
        self.n_class = n_class
        self.inchannel = 3
        self.plan_lists = (np.array([3,6,8,16], dtype=int) * 16).tolist()
        self.groups = 1
        self.mergeBN = mergeBN
        self.stem_conv = ConvBNRelu(in_channels=self.inchannel * self.groups, out_channels=self.plan_lists[0] * self.groups, kernel_size=7, stride=2,groups = self.groups,mergeBN=self.mergeBN)
        self.stage0 = nn.Sequential(
            ConvBNRelu(in_channels=self.plan_lists[0] * self.groups, out_channels=self.plan_lists[0] * self.groups, kernel_size=3, stride=2,groups = self.groups,mergeBN=self.mergeBN),
            ResBlock(self.plan_lists[0] * self.groups, self.plan_lists[0] * self.groups, groups = self.groups,downsample=False,mergeBN=self.mergeBN),
            ResBlock(self.plan_lists[0] * self.groups, self.plan_lists[0] * self.groups, groups = self.groups,downsample=False,mergeBN=self.mergeBN),
            ResBlock(self.plan_lists[0] * self.groups, self.plan_lists[0] * self.groups, groups = self.groups,downsample=False,mergeBN=self.mergeBN)
        )

        self.stage1 = nn.Sequential(
            ResBlock(self.plan_lists[0] * self.groups, self.plan_lists[1] * self.groups, groups = self.groups,downsample=True,mergeBN=self.mergeBN),
            ResBlock(self.plan_lists[1] * self.groups, self.plan_lists[1] * self.groups, groups = self.groups,downsample=False,mergeBN=self.mergeBN),
            ResBlock(self.plan_lists[1] * self.groups, self.plan_lists[1] * self.groups, groups = self.groups,downsample=False,mergeBN=self.mergeBN),
            ResBlock(self.plan_lists[1] * self.groups, self.plan_lists[1] * self.groups, groups = self.groups,downsample=False,mergeBN=self.mergeBN),
        )

        self.stage2 = nn.Sequential(
            ResBlock(self.plan_lists[1] * self.groups, self.plan_lists[2] * self.groups, groups = self.groups,downsample=True,mergeBN=self.mergeBN),
            ResBlock(self.plan_lists[2] * self.groups, self.plan_lists[2] * self.groups, groups = self.groups,downsample=False,mergeBN=self.mergeBN),
            ResBlock(self.plan_lists[2] * self.groups, self.plan_lists[2] * self.groups, groups = self.groups,downsample=False,mergeBN=self.mergeBN),
            ResBlock(self.plan_lists[2] * self.groups, self.plan_lists[2] * self.groups, groups = self.groups,downsample=False,mergeBN=self.mergeBN),
            ResBlock(self.plan_lists[2] * self.groups, self.plan_lists[2] * self.groups, groups = self.groups,downsample=False,mergeBN=self.mergeBN),
            ResBlock(self.plan_lists[2] * self.groups, self.plan_lists[2] * self.groups, groups = self.groups,downsample=False,mergeBN=self.mergeBN),
        )

        self.stage3 = nn.Sequential(
            ResBlock(self.plan_lists[2] * self.groups, self.plan_lists[3] * self.groups, groups = self.groups,downsample=True,mergeBN=self.mergeBN),
            ResBlock(self.plan_lists[3] * self.groups, self.plan_lists[3] * self.groups, groups = self.groups,downsample=False,mergeBN=self.mergeBN),
            ResBlock(self.plan_lists[3] * self.groups, self.plan_lists[3] * self.groups, groups = self.groups,downsample=False,mergeBN=self.mergeBN),
        )
       
        self.refinenet2 = RefineBlock(deeper_in_channels=self.plan_lists[3] * self.groups, side_in_channels=self.plan_lists[2] * self.groups,
                                      out_channels=self.plan_lists[2] * self.groups, relu_layer=nn.ReLU(inplace=True), groupnum=self.groups,
                                      mergeBN=self.mergeBN)
        self.refinenet1 = RefineBlock_DY(deeper_in_channels=self.plan_lists[2] * self.groups, side_in_channels=self.plan_lists[1] * self.groups,
                                         out_channels=self.plan_lists[1] * self.groups, relu_layer=nn.ReLU(inplace=True), groupnum=self.groups,
                                         mergeBN=self.mergeBN)
        
        self.detect = Detect3D(nc=self.n_class, ch=[self.plan_lists[1]* self.groups, self.plan_lists[2]* self.groups, self.plan_lists[3]* self.groups])
        
        
    def forward(self, x):
        layer_1 = self.stem_conv(x)
        layer_1 = self.stage0(layer_1)
        layer_2 = self.stage1(layer_1)
        layer_3 = self.stage2(layer_2)

        path_3 = self.stage3(layer_3)
        path_2 = self.refinenet2(path_3, layer_3)
        path_1 = self.refinenet1(path_2, layer_2)

        detect = self.detect([path_1, path_2, path_3])

        return detect[0],detect[1], None

class ResNetJ6eModel(nn.Module):
    def __init__(self, n_class=10, mergeBN=False):  # model, input channels, number of classes
        super(ResNetJ6eModel, self).__init__()
        self.mergeBN = mergeBN
        self.model = R34TDA4(n_class=n_class, mergeBN=self.mergeBN)
        # self.model.anchors = anchors
        self.names = [str(i) for i in range(n_class)]

        ## Build strides, anchors
        if self.mergeBN==False:
            m = self.model.detect  # Detect()
            if isinstance(m, Detect3D):
                s = 256  # 2x min stride by input.shape = 1024
                _,tmp = self.forward(torch.zeros(1, 3, s, s))
                if isinstance(tmp, tuple):
                    tmp = tmp[0]
                m.stride = torch.tensor([s / x.shape[-2] for x in tmp])  # forward
                # m.anchors /= m.stride.view(-1, 1, 1) # anchors are scaled to the pyramid resolution
                # check_anchor_order(m)
                self.stride = m.stride
                print('Strides: %s' % m.stride.tolist())
            # self._initialize_biases()  # only run once

            # Init weights, biases
        # initialize_weights(self)

    def forward(self, x):
        layer_1 = self.model.stem_conv(x)
        layer_1 = self.model.stage0(layer_1)
        layer_2 = self.model.stage1(layer_1)
        layer_3 = self.model.stage2(layer_2)

        path_3 = self.model.stage3(layer_3)
        path_2 = self.model.refinenet2(path_3, layer_3)
        path_1 = self.model.refinenet1(path_2, layer_2)

        output = self.model.detect([path_1, path_2, path_3])

        return output
