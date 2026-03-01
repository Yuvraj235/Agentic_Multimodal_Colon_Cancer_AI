import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)

        # In case of size mismatch
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ConvNeXtUNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        # -----------------------
        # Encoder
        # -----------------------
        self.encoder = timm.create_model(
            "convnext_tiny",
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3)
        )

        enc_channels = self.encoder.feature_info.channels()
        # Example: [96, 192, 384, 768]

        # -----------------------
        # Decoder
        # -----------------------
        self.up4 = UpBlock(enc_channels[3], enc_channels[2], 384)
        self.up3 = UpBlock(384, enc_channels[1], 192)
        self.up2 = UpBlock(192, enc_channels[0], 96)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(96, 64, 2, 2),
            DoubleConv(64, 64)
        )

        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):

        # Encoder
        f1, f2, f3, f4 = self.encoder(x)

        # Decoder
        d4 = self.up4(f4, f3)
        d3 = self.up3(d4, f2)
        d2 = self.up2(d3, f1)
        d1 = self.up1(d2)

        out = self.final(d1)
        out = F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)

        return out  # IMPORTANT: no sigmoid here