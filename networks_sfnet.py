import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from networks_base import get_pad, ConvWithActivation, DeConvWithActivation
from network.nn.operators import AlignedModule, PSPModule
from network.nn.mynn import Norm2d, Upsample

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv3x3_bn_relu(in_planes, out_planes, stride=1, normal_layer=nn.BatchNorm2d):
    return nn.Sequential(
            conv3x3(in_planes, out_planes, stride),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
    )

class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, same_shape=True, **kwargs):
        super(Residual,self).__init__()
        self.same_shape = same_shape
        strides = 1 if same_shape else 2
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,stride=strides)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        if not same_shape:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                 stride=strides)
        self.batch_norm2d = nn.BatchNorm2d(out_channels)

    def forward(self,x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        if not self.same_shape:
            x = self.conv3(x)
        out = self.batch_norm2d(out + x)
        return F.relu(out)
def Norm2d(in_channels):
    """
    Custom Norm Function to allow flexible switching
    """
    normalization_layer = nn.BatchNorm2d(in_channels)
    return normalization_layer

class UperNetAlignHead(nn.Module):

    def __init__(self, inplane, num_class, norm_layer=nn.BatchNorm2d, fpn_inplanes=[256, 512, 1024, 2048], fpn_dim=256,
                 conv3x3_type="conv", fpn_dsn=False):
        super(UperNetAlignHead, self).__init__()

        self.ppm = PSPModule(inplane, norm_layer=norm_layer, out_features=fpn_dim)
        self.fpn_dsn = fpn_dsn
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]:
            self.fpn_in.append(
                nn.Sequential(
                    nn.Conv2d(fpn_inplane, fpn_dim, 1),
                    norm_layer(fpn_dim),
                    nn.ReLU(inplace=False)
                )
            )
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        self.fpn_out_align =[]
        self.dsn = []
        for i in range(len(fpn_inplanes) - 1):
            self.fpn_out.append(nn.Sequential(
                conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            ))

            if conv3x3_type == 'conv':
                self.fpn_out_align.append(
                AlignedModule(inplane=fpn_dim, outplane=fpn_dim//2)
                )

            if self.fpn_dsn:
                self.dsn.append(
                    nn.Sequential(
                        nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, stride=1, padding=1),
                        norm_layer(fpn_dim),
                        nn.ReLU(),
                        nn.Dropout2d(0.1),
                        nn.Conv2d(fpn_dim, num_class, kernel_size=1, stride=1, padding=0, bias=True)
                    )
                )

        self.fpn_out = nn.ModuleList(self.fpn_out)
        self.fpn_out_align = nn.ModuleList(self.fpn_out_align)

        if self.fpn_dsn:
            self.dsn = nn.ModuleList(self.dsn)

        self.conv_last = nn.Sequential(
            conv3x3_bn_relu(len(fpn_inplanes) * fpn_dim, fpn_dim, 1),
            nn.Conv2d(fpn_dim, 3, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, conv_out):
        psp_out = self.ppm(conv_out[-1])
        f = psp_out
        fpn_feature_list = [psp_out]
        out = []
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x)
            f = self.fpn_out_align[i]([conv_x, f])
            f = conv_x + f
            fpn_feature_list.append(self.fpn_out[i](f))
            if self.fpn_dsn:
                out.append(self.dsn[i](f))
        fpn_feature_list.reverse()  # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]

        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(nn.functional.interpolate(
                fpn_feature_list[i],
                output_size,
                mode='bilinear', align_corners=True))

        fusion_out = torch.cat(fusion_list, 1)
        x = self.conv_last(fusion_out)

        return x

class Pert(nn.Module):
    def __init__(self, use_GPU=True):
        super(Pert, self).__init__()
        self.iteration = 3
        self.use_GPU = use_GPU

        self.conv0 = nn.Sequential(
            nn.Conv2d(6, 32, 4, 2, 1),
            nn.ReLU()
            )
        self.convb = ConvWithActivation(32, 64, kernel_size=4, stride=2, padding=1)
        self.res1 = Residual(64, 64)
        self.res2 = Residual(64, 128, same_shape=False)
        self.res3 = Residual(128, 256, same_shape=False)
        self.res4 = Residual(256, 256)
        self.res5 = Residual(256, 512, same_shape=False)
        self.res6 = Residual(512, 512)
        self.conv2 = ConvWithActivation(512,512,kernel_size=1)
        self.deconv1 = DeConvWithActivation(512, 256, kernel_size=3, padding=1, stride=2)
        self.deconv2 = DeConvWithActivation(256 * 2, 128, kernel_size=3, padding=1, stride=2)
        self.deconv3 = DeConvWithActivation(128 * 2, 64, kernel_size=3, padding=1, stride=2)
        self.deconv4 = DeConvWithActivation(64 * 2, 32, kernel_size=3, padding=1, stride=2)
        self.deconv5 = DeConvWithActivation(64, 3, kernel_size=3, padding=1, stride=2)
        self.lateral_connection1 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1),
            nn.Conv2d(512, 256, kernel_size=1, padding=0, stride=1), )
        self.lateral_connection2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1),
            nn.Conv2d(256, 128, kernel_size=1, padding=0, stride=1), )
        self.lateral_connection3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),
            nn.Conv2d(128, 64, kernel_size=1, padding=0, stride=1), )
        self.lateral_connection4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1),
            nn.Conv2d(64, 32, kernel_size=1, padding=0, stride=1), )

        self.conv_o1 = nn.Conv2d(64, 3, kernel_size=1)
        self.conv_o2 = nn.Conv2d(32, 3, kernel_size=1)
        self.sf_net = UperNetAlignHead(512, num_class=1, norm_layer=Norm2d,
                                         fpn_inplanes=[64, 128, 256, 512], fpn_dim=128, fpn_dsn=False)

    def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)

        x = input
        h = Variable(torch.zeros(batch_size, 32, row, col))
        c = Variable(torch.zeros(batch_size, 32, row, col))

        if self.use_GPU:
            h = h.cuda()
            c = c.cuda()

        x_list = []
        mask_f_list = []
        x_o_list = []
        for i in range(self.iteration):
            x = torch.cat((input, x), 1)

            # down sample
            x = self.conv0(x) # 32 256
            mask_in = []
            con_x1 = x
            x = self.convb(x) # 64  // 128
            mask_in.append(x)
            x = self.res1(x) # 64
            con_x2 = x
            x = self.res2(x) # 128 // 64
            con_x3 = x
            mask_in.append(x)
            x = self.res3(x) # 256 // 32
            con_x4 = x
            mask_in.append(x)
            x = self.res4(x) #
            x = self.res5(x) # 512 // 16
            mask_in.append(x)
            x = self.res6(x) # 512
            x = self.conv2(x)
            # BRN
            x = self.deconv1(x)
            x = F.interpolate(x,size = con_x4. shape[2:], mode = 'bilinear')
            x = torch.cat([self.lateral_connection1(con_x4), x], dim=1)
            x = self.deconv2(x)

            x = F.interpolate(x,size = con_x3.shape[2:], mode = 'bilinear')
            x = torch.cat([self.lateral_connection2(con_x3), x], dim=1)
            x = self.deconv3(x)
            xo1 = x

            x = F.interpolate(x,size = con_x2.shape[2:], mode = 'bilinear')
            x = torch.cat([self.lateral_connection3(con_x2), x], dim=1)
            x = self.deconv4(x)
            xo2 = x

            x = F.interpolate(x,size = con_x1.shape[2:], mode = 'bilinear')
            x = torch.cat([self.lateral_connection4(con_x1), x], dim=1)
            x = self.deconv5(x)

            # GGN
            mm_temo = self.sf_net(mask_in)
            mm = Upsample(mm_temo, (row, col))
            x = Upsample(x, (row, col))

            x = mm * x + (1 - mm) * input
            x_list.append(x)

            mask_f_list.append(mm)
        xo1 = self.conv_o1(xo1)
        xo2 = self.conv_o2(xo2)

        # RegionMS
        mm_o1 = mm_temo
        in_o1 = F.interpolate(input, scale_factor=0.25)
        mm_o2 = F.interpolate(mm_temo, scale_factor=2)
        in_o2 = F.interpolate(input, scale_factor=0.5)
        xo1 = mm_o1 * xo1 + (1 - mm_o1) * in_o1
        xo2 = mm_o2 * xo2 + (1 - mm_o2) * in_o2

        x_o_list.append(xo1)
        x_o_list.append(xo2)
        return x, x_list, mm, mask_f_list, x_o_list
