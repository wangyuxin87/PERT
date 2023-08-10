import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
from torch import nn
from tensorboardX import SummaryWriter
from models.Model import VGG16FeatureExtractor

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2
    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2
    C1 = 0.01**2
    C2 = 0.03**2
    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def gram_matrix(feat):
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram
def dice_loss(input, target):
    input = input.contiguous().view(input.size()[0], -1)
    target = target.contiguous().view(target.size()[0], -1)
    input = input
    target = target
    a = torch.sum(input * target, 1)
    b = torch.sum(input * input, 1) + 0.001
    c = torch.sum(target * target, 1) + 0.001
    d = (2 * a) / (b + c)
    dice_loss = torch.mean(d)
    return 1 - dice_loss

def L2(f_):
    return (((f_ ** 2).sum(dim=1)) ** 0.5).reshape(f_.shape[0], 1, f_.shape[2], f_.shape[3]) + 1e-8

def similarity(feat):
    feat = feat.float()
    tmp = L2(feat).detach()
    feat = feat / tmp
    feat = feat.reshape(feat.shape[0], feat.shape[1], -1)
    return torch.einsum('icm,icn->imn', [feat, feat])  # non-local

def sim_dis_compute(f_S, f_T):
    sim_err = ((similarity(f_T) - similarity(f_S)) ** 2) / ((f_T.shape[-1] * f_T.shape[-2]) ** 2) / f_T.shape[0]
    sim_dis = sim_err.sum()
    return sim_dis

class loss_pert(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(loss_pert, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)
        self.l1 = nn.L1Loss()
        self.extractor = VGG16FeatureExtractor()
        self.scale = [0.5, 0.25, 0.125]
        self.criterion = sim_dis_compute

    def forward(self, img1, img2, mask_1, mask_2, mask_list, out_list, x_list):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel
        # mask loss
        mask_loss = 0
        for ind_mask in range(len(mask_list)):
            mask_loss += dice_loss(mask_list[ind_mask], mask_2)
        mask_loss = mask_loss/len(mask_list)
        # rs loss
        mask_re = 1 - mask_2
        holeLoss = 13 * self.l1((1 - mask_re) * img2, (1 - mask_re) * img1)
        validAreaLoss = 2 * self.l1(mask_re * img2, mask_re * img1)
        loss_h_v = holeLoss + validAreaLoss

        imgs1 = F.interpolate(img1, scale_factor=0.25)
        imgs2 = F.interpolate(img1, scale_factor=0.5)
        masks_a = F.interpolate(mask_re, scale_factor=0.25)
        masks_b = F.interpolate(mask_re, scale_factor=0.5)
        x_o1 = out_list[0]
        x_o2 = out_list[1]
        msrloss = 12 * self.l1((1 - masks_b) * x_o2, (1 - masks_b) * imgs2) + 1 * self.l1(masks_b * x_o2,
                                                                                          masks_b * imgs2) + \
                  10 * self.l1((1 - masks_a) * x_o1, (1 - masks_a) * imgs1) + 0.8 * self.l1(masks_a * x_o1,
                                                                                            masks_a * imgs1)
        rs_loss = loss_h_v + msrloss

        output_comp = mask_re * img1 + (1 - mask_re) * img2
        feat_output_comp = self.extractor(output_comp)
        feat_output = self.extractor(img2)
        feat_gt = self.extractor(img1)
        prcLoss = 0.0
        for i in range(3):
            prcLoss += 0.01 * self.l1(feat_output[i], feat_gt[i])
            prcLoss += 0.01 * self.l1(feat_output_comp[i], feat_gt[i])
        styleLoss = 0.0
        for i in range(3):
            styleLoss += 120 * self.l1(gram_matrix(feat_output[i]),
                                       gram_matrix(feat_gt[i]))
            styleLoss += 120 * self.l1(gram_matrix(feat_output_comp[i]),
                                       gram_matrix(feat_gt[i]))
        loss_vgg = styleLoss + prcLoss
        
        # gs loss
        gs_loss = 0
        for i in range(len(self.scale)):
            feat_S = feat_output[i]
            feat_T = feat_gt[i]
            feat_T.detach()
            total_w, total_h = feat_T.shape[2], feat_T.shape[3]
            patch_w, patch_h = int(total_w * self.scale[i]), int(total_h * self.scale[i])
            maxpool = nn.MaxPool2d(kernel_size=(patch_w, patch_h), stride=(patch_w, patch_h), padding=0,
                                   ceil_mode=True)  # change
            gs_loss_temp = self.criterion(maxpool(feat_S), maxpool(feat_T))
            gs_loss += gs_loss_temp
        rg_loss = gs_loss * 0.5 + rs_loss

        ssim_loss = _ssim(img1, img2, window, self.window_size, channel, self.size_average)
        print('mask: {:.4f}, ssim: {:.4f}, rgloss: {:.4f},  vgg: {:.4f}'.format(float(mask_loss), float(ssim_loss), float(rg_loss), float(loss_vgg)))
        loss_total_all = ssim_loss + rg_loss * -1 + loss_vgg * -1 + mask_loss * -1
        return loss_total_all