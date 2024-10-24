import torch
from torch import nn
import torch.nn.functional as F


# Edge-Guided Learning Network
class EdgNet(nn.Module):
    def __init__(self,n_feats=32):
        super(EdgNet, self).__init__()
        f = n_feats // 2  # 64 / 4 = 16
        self.conv1 = nn.Conv2d(n_feats, f, kernel_size=1)
        self.conv_f = nn.Conv2d(f, f, kernel_size=1)
        self.conv_max = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv3_ = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        self.conv_dilation = nn.Conv2d(f, f, kernel_size=3, padding=1,
                                        stride=3, dilation=2)
        # input conv
        # self.inc_edge = DoubleConv(1, 32)
        # output conv
        self.ouc_edge = nn.Conv2d(32,1,kernel_size=1)

    def forward(self, x): # x is the input feature
        # input operation1
        # _, c, _, _ = x.size()    # c=1 or c=3
        # if c == 1:
        #     x = self.inc_edge_sig(x)   # [B, 64, H, W]
        # else:
        #     x = self.inc_edge_mul(x)   # [B, 64, H, W]

        # input operation2
        # x = self.inc_edge(x)  # output [B 64 H W]

        # edge attention
        x = self.conv1(x)   # input:[B, 64, H, W]  output:[B, 16, H, W]
        shortCut = x     # [B, 16, H, W]
        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=7, stride=3)
        x = self.relu(self.conv_max(x))
        x = self.relu(self.conv3(x))
        x = self.conv3_(x)
        x = F.interpolate(x, (shortCut.size(2), shortCut.size(3)),
                          mode='bilinear', align_corners=False)
        shortCut = self.conv_f(shortCut)
        x = self.conv4(x+shortCut)
        x_fea = self.sigmoid(x)
        
        x_l = F.tanh(self.ouc_edge(x_fea))

        return x_fea, x_l

if __name__ == '__main__':
    img1 = torch.randn(2, 32, 144, 144)

    edge_attention = EdgNet()
    edge_fea, edge_l = edge_attention(img1)

    print(edge_fea.shape)
    print(edge_l.shape)
    # torch.Size([2, 64, 144, 144])
    # torch.Size([2, 1, 144, 144])