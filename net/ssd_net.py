import torch
import torch.nn as nn
import torch.nn.functional as F

_NORM_BONE = False

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


# Spatial-Spectral Domain Optimizing Network
class SSDNet(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=1, stride=1):
        super(SSDNet, self).__init__()

        self.inplanes = inplanes
        self.inter_planes = planes // 2
        self.planes = planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size-1)//2

        self.conv_q_right = nn.Conv2d(self.inplanes, 1, kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv_v_right = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv_up = nn.Conv2d(self.inter_planes, self.planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax_right = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()

        self.conv_q_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0, bias=False)   #g
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_v_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0, bias=False)   #theta
        self.softmax_left = nn.Softmax(dim=2)

        self.reset_parameters()

    def reset_parameters(self):
        kaiming_init(self.conv_q_right, mode='fan_in')
        kaiming_init(self.conv_v_right, mode='fan_in')
        kaiming_init(self.conv_q_left, mode='fan_in')
        kaiming_init(self.conv_v_left, mode='fan_in')

        self.conv_q_right.inited = True
        self.conv_v_right.inited = True
        self.conv_q_left.inited = True
        self.conv_v_left.inited = True

        # inc
        # self.inc_det = DoubleConv(1, 32)
        # ouc
        self.ouc_cda = nn.Conv2d(32, 1, kernel_size=1)

    # HR spatial attention
    def spatial_attention(self, x):
        input_x = self.conv_v_right(x)
        batch, channel, height, width = input_x.size()

        input_x = input_x.view(batch, channel, height * width)
        context_mask = self.conv_q_right(x)
        context_mask = context_mask.view(batch, 1, height * width)
        context_mask = self.softmax_right(context_mask)

        context = torch.matmul(input_x, context_mask.transpose(1,2))
        context = context.unsqueeze(-1)
        context = self.conv_up(context)

        mask_ch = self.sigmoid(context)

        out = x * mask_ch

        return out
    # HR spectral attention
    def spectral_attention(self, x):

        g_x = self.conv_q_left(x)
        batch, channel, height, width = g_x.size()

        avg_x = self.avg_pool(g_x)
        batch, channel, avg_x_h, avg_x_w = avg_x.size()

        avg_x = avg_x.view(batch, channel, avg_x_h * avg_x_w).permute(0, 2, 1)
        theta_x = self.conv_v_left(x).view(batch, self.inter_planes, height * width)
        context = torch.matmul(avg_x, theta_x)
        context = self.softmax_left(context)
        context = context.view(batch, 1, height, width)

        mask_sp = self.sigmoid(context)

        out = x * mask_sp

        return out

    def forward(self, x):
        # x = self.inc_det(x)
        context_spectral = self.spectral_attention(x)
        context_spatial = self.spatial_attention(x)
        out = context_spatial + context_spectral
        
        # ablation-wo-context_spatial
        # out = context_spectral
        
        # ablation-wo-context_spectral
        # out = context_spatial
        
        out_l = F.tanh(self.ouc_cda(out))
        return out, out_l

if __name__ == '__main__':
    x1 = torch.randn(4,32,144,144)
    x2 = torch.randn(4,32,144,144)
    # inc_det = DoubleConv(1, 64)
    # x1 = inc_det(x1)
    ssd = SSDNet(inplanes=32,planes=32)
    z, z_l = ssd(x1)
    print(z.shape)
    print(z_l.shape)
    # torch.Size([4, 64, 144, 144])




