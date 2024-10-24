import torch
import torch.nn as nn
import torch.nn.functional as F

from net.ssd_net import SSDNet
from net.cmk_net import CMKNet
from net.edg_net import EdgNet
from task_loss import CMKLoss, EdgLoss, FusionLoss, SSDLoss


class MConv_Dou(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MConv_Dou, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # ouc
        self.ouc_mconv = nn.Conv2d(out_channels, 1, kernel_size=1)

    def forward(self, x, y):
        z_cat = torch.cat((x, y), dim=1)
        fea_conv = self.conv_block(z_cat)
        l_conv = F.tanh(self.ouc_mconv(fea_conv))
        return fea_conv, l_conv


class MConv_Sig(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MConv_Sig, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # ouc
        self.ouc_mconv = nn.Conv2d(out_channels, 1, kernel_size=1)

    def forward(self, x):
        fea_conv = self.conv_block(x)
        l_conv = F.tanh(self.ouc_mconv(fea_conv))
        return fea_conv, l_conv


def save_grad(grads, name):
    def hook(grad):
        grads[name] = grad

    return hook


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.double_conv(x)
    
    
class SingleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.double_conv(x)


class OutConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return F.tanh(self.conv(x))


class MLFNet(nn.Module):

    def __init__(self):
        super(MLFNet, self).__init__()
        self.inc_edg = SingleConv(1, 32)
        self.inc_ssd = SingleConv(1, 32)    
        self.inc_cmk_1 = SingleConv(1, 32)
        self.inc_cmk_2 = SingleConv(1, 32)
        self.layer = DoubleConv(in_channels=96, out_channels=16, mid_channels=48)
        self.outc = OutConv(16, 1)

        self.cmk_net = CMKNet(icha=32)
        self.edg_net = EdgNet()
        self.ssd_net = SSDNet(inplanes=32,planes=32)
        
        self.c_loss = CMKLoss()
        self.e_loss = EdgLoss()
        self.d_loss = SSDLoss()
        self.f_loss = FusionLoss()

        

    def forward(self, img1, img2):
        ie_img = self.inc_edg(img1)
        id_img = self.inc_ssd(img2)
        ic_img1 = self.inc_cmk_1(img1)   
        ic_img2 = self.inc_cmk_2(img2)

        fea_edg, l_edg = self.edg_net(ie_img)
        fea_ssd, l_ssd = self.ssd_net(id_img)
        fea_cmk, l_cmk = self.cmk_net(ic_img1, ic_img2)
        
        # concat
        c = torch.cat((fea_edg, fea_cmk, fea_ssd), dim=1)
        c = self.layer(c) 
        f = self.outc(c)

        return f, l_edg, l_cmk, l_ssd

    def _closs(self, x, y, l):
        closs = self.c_loss(x, y, l)
        return closs

    def _eloss(self, y, l):
        eloss = self.e_loss(y, l)
        return eloss

    def _dloss(self, x, l, m):
        dloss = self.d_loss(x, l, m)
        return dloss

    def _floss(self, x, y, z):
        floss = self.f_loss(x, y, z)
        return floss