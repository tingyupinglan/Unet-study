import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1),
                                         nn.BatchNorm2d(out_channels),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1),
                                         nn.BatchNorm2d(out_channels),
                                         nn.ReLU(inplace=True))

    def forward(self,x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.max_pool_conv = nn.Sequential(nn.MaxPool2d(2),
                                           DoubleConv(in_channels, out_channels))

    def forward(self,x):
        return self.max_pool_conv(x)

class Up(nn.Module):
    def __init__(self,in_channels,skip_channels,out_channels,bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2,mode = 'bilinear',align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels,in_channels // 2,kernel_size=2,stride=2)

        self.conv = DoubleConv(in_channels + skip_channels, out_channels)
    #x1是降采样的特征图，x2是原始图
    def forward(self,x1,x2):
        x1 = self.up(x1)
        diffY = torch.tensor(x2.size()[2] - x1.size()[2])
        diffX = torch.tensor(x2.size()[3] - x1.size()[3])
        x1 = F.pad(x1,[diffX // 2, diffX - diffX // 2,diffY // 2,diffY - diffY // 2],mode='constant',value = 0)

        x = torch.cat([x2,x1],1)
        x = self.conv(x)
        return x
class OutConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        #1*1卷积，相当于在通道方向上加权
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=1)

    def forward(self,x):
        return self.conv(x)

class Unet(nn.Module):
    def __init__(self,n_channels,n_classes,bilinear=False):
        super().__init__()
        self.classes = n_classes
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels,64)
        self.down1 = Down(64,128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        self.up1 = Up(1024,512,512)
        self.up2 = Up(512, 256,256)
        self.up3 = Up(256, 128,128)
        self.up4 = Up(128, 64,64)
        self.outc = OutConv(64,n_classes)

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
#unet分类任务
class Unet_Classify(nn.Module):
    def __init__(self,n_channels,n_classes,bilinear=False):
        super().__init__()
        self.classes = n_classes
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels,64)
        self.down1 = Down(64,128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        self.up1 = Up(1024,512,512)
        self.up2 = Up(512, 256,256)
        self.up3 = Up(256, 128,128)
        self.up4 = Up(128, 64,64)
        self.classifier = torch.nn.Sequential(nn.AdaptiveAvgPool2d(1),#全局平均池化，每个特征图池化成1个点，通道数保持不变
                                              nn.Flatten(),#展平
                                              nn.Linear(64,n_classes)#全连接层
                                              )

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

        logits = self.classifier(x)
        return logits
