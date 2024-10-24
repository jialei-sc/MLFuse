#!/usr/bin/python
# -*- encoding: utf-8 -*-
import math
import torch.nn as nn
from ssim import SSIM
import torch
import torch.nn.functional as F


class EdgLoss(nn.Module):
    def __init__(self):
        super(EdgLoss, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self, image_vis, generate_img):
        vis_grad = self.sobelconv(image_vis)
        generate_img_grad = self.sobelconv(generate_img)
        loss_grad = F.l1_loss(generate_img_grad, vis_grad)

        return loss_grad

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)


class SSDLoss(nn.Module):
    def __init__(self):
        super(SSDLoss, self).__init__()

    def forward(self, image_ir, generate_img, m):
        g_vsm = m * generate_img
        ir_vsm = m * image_ir
        loss_ssd = F.l1_loss(g_vsm, ir_vsm)

        return loss_ssd


class CMKLoss(nn.Module):
    def __init__(self):
        super(CMKLoss, self).__init__()

    def forward(self, img1, img2, f_img):
        loss_1 = F.mse_loss(f_img, img1)
        loss_2 = F.mse_loss(f_img, img2)
        
        loss_cmk = loss_1 + loss_2

        return loss_cmk


class FusionLoss(nn.Module):
    def __init__(self):
        super(FusionLoss, self).__init__()
        self.ssim_loss = SSIM()

    def forward(self, img1, img2, f_img):
        ssim_1 = (1 - self.ssim_loss(img1, f_img)) / 2
        ssim_2 = (1 - self.ssim_loss(img2, f_img)) / 2
        loss_fus = (ssim_1 + ssim_2) * 0.1
        return loss_fus
