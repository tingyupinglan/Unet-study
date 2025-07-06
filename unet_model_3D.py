import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(nn.Conv3d(in_channels,out_channels,kernel_size=3,stride=1,padding=1),
                                         nn.BatchNorm3d(out_channels),
                                         nn.ReLU(inplace=True),
                                         nn.Conv3d(out_channels,out_channels,kernel_size=3,padding=1),
                                         nn.BatchNorm3d(out_channels),
                                         nn.ReLU(inplace=True))

    def forward(self,x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.max_pool_conv = nn.Sequential(nn.MaxPool3d(2),
                                           DoubleConv(in_channels, out_channels))

    def forward(self,x):
        return self.max_pool_conv(x)

class Up(nn.Module):
    # n_skip_channels是跳跃连接，降采样特征图与上采样解码图拼接时额外的通道数
    def __init__(self,in_channels,n_skip_channels,out_channels,bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2,mode = 'trilinear',align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels,in_channels // 2,kernel_size=2,stride=2)

        self.conv = DoubleConv(in_channels + n_skip_channels, out_channels)
    #x1是降采样的特征图，x2是原始图
    def forward(self,x1,x2):
        x1 = self.up(x1)

        diffZ = torch.tensor(x2.size()[2] - x1.size()[2])
        diffY = torch.tensor(x2.size()[3] - x1.size()[3])
        diffX = torch.tensor(x2.size()[4] - x1.size()[4])
        x1 = F.pad(x1,[diffX // 2, diffX - diffX // 2,diffY // 2,diffY - diffY // 2,diffZ//2, diffZ - diffZ // 2],mode='constant',value = 0)

        x = torch.cat([x2,x1],1)
        x = self.conv(x)
        return x
class OutConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        #1*1卷积，相当于在通道方向上加权
        self.conv = nn.Conv3d(in_channels,out_channels,kernel_size=1)

    def forward(self,x):
        return self.conv(x)

class Unet(nn.Module):

    def __init__(self,n_channels,n_classes,bilinear=False):
        super().__init__()
        self.classes = n_classes
        self.n_channels = n_channels
        self.bilinear = bilinear

        # self.inc = DoubleConv(n_channels,64)
        # self.down1 = Down(64,128)
        # self.down2 = Down(128, 256)
        # self.down3 = Down(256, 512)
        # self.down4 = Down(512, 1024)
        #
        # self.up1 = Up(1024,512,512)
        # self.up2 = Up(512,256, 256)
        # self.up3 = Up(256,128, 128)
        # self.up4 = Up(128, 64,64)
        self.inc = DoubleConv(n_channels,16)
        self.down1 = Down(16,32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        self.down4 = Down(128, 256)

        self.up1 = Up(256,128,128)
        self.up2 = Up(128,64, 64)
        self.up3 = Up(64,32, 32)
        self.up4 = Up(32, 16,16)
        self.outc = OutConv(16,n_classes)

    def forward(self,x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5,x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits

# if __name__ == '__main__':
#     net = Unet(n_channels=3, n_classes=1)
#     print(net)