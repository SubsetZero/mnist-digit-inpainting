"""Small convolutional encoder-decoder for 28x28 grayscale inpainting."""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class InpaintingCNN(nn.Module):
    """U-Net-style encoder-decoder with skip connections.

    Input: masked image, concatenated with the binary mask as a second
    channel (so the network is told exactly where the gap is).
    Output: the reconstructed full image.
    """

    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(ConvBlock(2, 32), ConvBlock(32, 32))
        self.pool1 = nn.MaxPool2d(2)  # 28 -> 14

        self.enc2 = nn.Sequential(ConvBlock(32, 64), ConvBlock(64, 64))
        self.pool2 = nn.MaxPool2d(2)  # 14 -> 7

        self.bottleneck = nn.Sequential(ConvBlock(64, 128), ConvBlock(128, 128))

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  # 7 -> 14
        self.dec2 = nn.Sequential(ConvBlock(128, 64), ConvBlock(64, 64))

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)  # 14 -> 28
        self.dec1 = nn.Sequential(ConvBlock(64, 32), ConvBlock(32, 32))

        self.out = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, masked_image, mask):
        x = torch.cat([masked_image, mask], dim=1)

        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        b = self.bottleneck(p2)

        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return torch.sigmoid(self.out(d1))
