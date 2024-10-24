import torch
import torch.nn.functional as F
from torch import nn


# Cross-Modal Knowledge Reinforcing Network
class CMKNet(nn.Module):

    def __init__(self, icha=32):
        super(CMKNet, self).__init__()
        self.fuse5_conv1 = nn.Sequential(nn.Conv2d(icha, icha, kernel_size=3, padding=1), nn.BatchNorm2d(icha), nn.PReLU())   # defult:512
        self.fuse5_conv2 = nn.Sequential(nn.Conv2d(icha, icha, kernel_size=3, padding=1), nn.BatchNorm2d(icha), nn.PReLU())
        self.fuse5_conv3 = nn.Sequential(nn.Conv2d(icha, icha, kernel_size=3, padding=1), nn.BatchNorm2d(icha), nn.PReLU())
        self.fuse5_conv4 = nn.Sequential(nn.Conv2d(icha, icha, kernel_size=3, padding=1), nn.BatchNorm2d(icha), nn.PReLU())
        self.fuse5_conv5 = nn.Sequential(nn.Conv2d(icha, icha, kernel_size=3, padding=1), nn.BatchNorm2d(icha), nn.PReLU())
        # self.sideout5 = nn.Sequential(nn.Conv2d(64, 1, kernel_size=3, padding=1))

        # self.fuse4_conv1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU())
        # self.fuse4_conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU())
        # self.fuse4_conv3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU())
        # self.fuse4_conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU())
        # self.fuse4_conv5 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU())

        # # input conv
        # self.inc_cda_1 = DoubleConv(1, 32)
        # self.inc_cda_2 = DoubleConv(1, 32)

        #output conv
        self.ouc_cda = nn.Conv2d(icha, 1, kernel_size=1)

    def forward(self, fea_1, fea_2):

        # cda --> get S
        certain_feature5 = fea_1 * fea_2
        fuse5_conv1 = self.fuse5_conv1(fea_1 + certain_feature5)
        fuse5_conv2 = self.fuse5_conv2(fea_2 + certain_feature5)
        fuse5_certain = self.fuse5_conv3(fuse5_conv1 + fuse5_conv2)

        uncertain_feature5 = self.fuse5_conv4(torch.abs(fea_1 - fea_2))
        fuse5 = self.fuse5_conv5(fuse5_certain + uncertain_feature5)
        # sideout5 = self.sideout5(fuse5)

        # cda -- with S
        # certain_feature4 = F.sigmoid(sideout5) * certain_feature5
        # fuse4_conv1 = self.fuse4_conv1(fea_1 + certain_feature4)
        # fuse4_conv2 = self.fuse4_conv2(fea_2 + certain_feature4)
        # fuse4_certain = self.fuse4_conv3(fuse4_conv1 + fuse4_conv2)
        # uncertain_feature4 = self.fuse4_conv4(F.sigmoid(sideout5) * torch.abs(fea_1 - fea_2))
        # fuse4 = self.fuse4_conv5(fuse4_certain + uncertain_feature4)

        # fuse4_l = self.ouc_cda(fuse4)
        fuse5_l = F.tanh(self.ouc_cda(fuse5))

        return fuse5, fuse5_l




if __name__ == '__main__':
    img_ct = torch.randn(1, 32, 144, 144)
    img_mrict = torch.randn(1, 32, 144, 144)

    img_vis = torch.randn(1, 32, 144, 144)
    img_ir = torch.randn(1, 32, 144, 144)

    cmk = CMKNet()

    fuse_ct_mri_fea, fuse_ct_mri_l = cmk(img_mrict,img_ct)
    fuse_vis_ir_fea, fuse_vis_ir_l = cmk(img_vis,img_ir)

    print(fuse_ct_mri_fea.shape)
    print(fuse_ct_mri_l.shape)
    print(fuse_vis_ir_fea.shape)
    print(fuse_vis_ir_l.shape)
    # torch.Size([4, 64, 144, 144])
    # torch.Size([4, 1, 144, 144])
    # torch.Size([4, 64, 144, 144])
    # torch.Size([4, 1, 144, 144])