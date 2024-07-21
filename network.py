import torch
import torch.nn as nn
import torch.nn.functional as F
from detail_sdl import SDL_attention
from cda import Consistency_Difference_Aggregation
from edge_attention import Weight_Prediction_Network
from task_loss import CDALoss, EdgeLoss, FusionLoss, DetLoss


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

# MLFuse-network
class TransNet(nn.Module):

    def __init__(self):
        super(TransNet, self).__init__()
        self.inc_edge = SingleConv(1, 32)
        self.inc_deta = SingleConv(1, 32)    
        self.inc_cda_1 = SingleConv(1, 32)
        self.inc_cda_2 = SingleConv(1, 32)
        self.layer = DoubleConv(in_channels=96, out_channels=16, mid_channels=48)
        self.outc = OutConv(16, 1)

        self.cda_net = Consistency_Difference_Aggregation(icha=32)   # 32 1
        # self.cda_net = MConv_Dou(in_channels=64,out_channels=32)
        self.edge_net = Weight_Prediction_Network()  # 32 1
        # self.det_net = Detail_Trans()  # 1
        self.det_net = SDL_attention(inplanes=32,planes=32)  # 32 1
        # self.edge_net = MConv_Sig(in_channels=32,out_channels=32) # 32 1
        # self.det_net = MConv_Sig(in_channels=32,out_channels=32) # 32 1
        
        self.c_loss = CDALoss()
        self.e_loss = EdgeLoss()
        self.d_loss = DetLoss()
        self.f_loss = FusionLoss()


    def forward(self, img1, img2):
        # img1: vis、mrict、mripet、mrispect
        # img2: ir、ct、pet、spect

        # edge-inconv    
        ie_img = self.inc_edge(img1) # ic=1 oc=32
        # deta-inconv 
        id_img = self.inc_deta(img2)   # ic=1 oc=32
        # cda-inconv 
        ic_img1 = self.inc_cda_1(img1)   
        ic_img2 = self.inc_cda_2(img2)
         
        # edge-net output
        fea_edge, l_edge = self.edge_net(ie_img)
        # detail-net output
        fea_det, l_det = self.det_net(id_img)
        # cda-net output
        fea_cda, l_cda = self.cda_net(ic_img1, ic_img2)
        
        # concat
        c = torch.cat((fea_edge, fea_cda, fea_det), dim=1)
        c = self.layer(c) 
        f = self.outc(c)

        return f, l_edge, l_cda, l_det

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



if __name__ == '__main__':
    model = TransNet()
    img_ir = torch.rand(2, 1, 144, 144)
    img_vis = torch.rand(2, 1, 144, 144)
    img_ct = torch.rand(2, 1, 144, 144)
    img_mrict = torch.rand(2, 1, 144, 144)

    a1, b1, c1, d1 = model(img_vis.float(), img_ir.float())
    a2, b2, c2, d2 = model(img_mrict.float(), img_ct.float())

    print(a1.shape)        # torch.Size([2, 1, 144, 144])
    print(b1.shape)      # torch.Size([2, 1, 144, 144])
