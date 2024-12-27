#model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class UNetAutoencoder(nn.Module):
    def __init__(self):
        super(UNetAutoencoder, self).__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            SEBlock(32)  # 添加SE模块
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            SEBlock(64)  # 添加SE模块
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            SEBlock(128)  # 添加SE模块
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
            SEBlock(256)  # 添加SE模块
        )

        # Decoder
        self.up4 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1)
        self.dec4 = nn.Sequential(
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            SEBlock(128)  # 添加SE模块
        )
        self.up3 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.dec3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            SEBlock(64)  # 添加SE模块
        )
        self.up2 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.dec2 = nn.Sequential(
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            SEBlock(32)  # 添加SE模块
        )
        self.up1 = nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1)
        self.final_layer = nn.Sequential(
            nn.Conv2d(2, 1, 3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        # Decoder
        up4 = self.up4(enc4)
        dec4 = torch.cat((up4, enc3), dim=1)
        dec4 = self.dec4(dec4)

        up3 = self.up3(dec4)
        dec3 = torch.cat((up3, enc2), dim=1)
        dec3 = self.dec3(dec3)

        up2 = self.up2(dec3)
        dec2 = torch.cat((up2, enc1), dim=1)
        dec2 = self.dec2(dec2)

        up1 = self.up1(dec2)
        final_output = torch.cat((up1, x), dim=1)
        final_output = self.final_layer(final_output)

        return final_output, enc4  # 返回解码后的输出和编码器的最后一层输出
