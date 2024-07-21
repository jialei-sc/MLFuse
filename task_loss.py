#!/usr/bin/python
# -*- encoding: utf-8 -*-
import math
import torch.nn as nn
from ssim import SSIM
import torch
import torch.nn.functional as F


# e_loss_vis
class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
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

# d_loss_ir
class DetLoss(nn.Module):
    def __init__(self):
        super(DetLoss, self).__init__()

    def forward(self, image_ir, generate_img, m):
        g_vsm = m * generate_img
        ir_vsm = m * image_ir
        ssim_det = F.l1_loss(g_vsm, ir_vsm)

        return ssim_det

# c_loss
class CDALoss(nn.Module):
    def __init__(self):
        super(CDALoss, self).__init__()

    def forward(self, img1, img2, f_img):
        loss_1 = F.mse_loss(f_img, img1)
        loss_2 = F.mse_loss(f_img, img2)
        
        loss_cda = loss_1 + loss_2

        return loss_cda

# f_loss
class FusionLoss(nn.Module):
    def __init__(self):
        super(FusionLoss, self).__init__()
        self.ssim_loss = SSIM()

    def forward(self, img1, img2, f_img):
        ssim_1 = (1 - self.ssim_loss(img1, f_img)) / 2
        ssim_2 = (1 - self.ssim_loss(img2, f_img)) / 2
        loss_fus = (ssim_1 + ssim_2) * 0.1
        return loss_fus



if __name__ == '__main__':
    img1 = torch.randn(4,1,144,144)
    img2 = torch.randn(4,1,144,144)
    img3 = torch.randn(4,1,144,144)

    #     d_loss = DetLoss()
    #     x = d_loss(img1,img2)  # x:  tensor(2.0139)
    #     print('x: ', x)  # x:  tensor(0.6464, device='cuda:0', grad_fn=<NllLoss2DBackward>)
    c_loss = CDALoss()
    x = c_loss(img1,img2,img3)
    print('x: ', x)     # x:  tensor(1.7374, dtype=torch.float64)
