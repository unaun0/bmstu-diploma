from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class CoordAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 32) -> None:
        super().__init__()
        mip = max(8, channels // reduction)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(
            channels, mip, 
            kernel_size=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(
            mip, channels, 
            kernel_size=1, bias=False
        )
        self.conv_w = nn.Conv2d(
            mip, channels, 
            kernel_size=1, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        _, _, height, width = x.shape
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [height, width], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        return identity \
            * torch.sigmoid(self.conv_h(x_h)) \
                * torch.sigmoid(self.conv_w(x_w))


class DenseSkipFusion(nn.Module):
    def __init__(
        self,
        c2_channels: int,
        c3_channels: int,
        aspp_channels: int,
        *,
        branch_channels: int = 48,
        out_channels: int = 304,
    ) -> None:
        super().__init__()
        self.c2_proj = _conv_bn_relu(c2_channels, branch_channels)
        self.c3_proj = _conv_bn_relu(c3_channels, branch_channels)
        self.aspp_proj = _conv_bn_relu(aspp_channels, branch_channels)
        self.refine = _separable_conv(branch_channels * 3, out_channels)

    def forward(
        self,
        feat_c2: torch.Tensor,
        feat_c3: torch.Tensor,
        aspp_features: torch.Tensor,
    ) -> torch.Tensor:
        target = feat_c2.shape[2:]
        c2 = self.c2_proj(feat_c2)
        c3 = F.interpolate(self.c3_proj(feat_c3), size=target, mode="bilinear", align_corners=False)
        aspp = self.aspp_proj(aspp_features)
        if aspp.shape[2:] != target:
            aspp = F.interpolate(
                aspp, size=target, mode="bilinear", align_corners=False
            )
            
        return self.refine(torch.cat([c2, c3, aspp], dim=1))


class DeepLabV3PlusDecoderWithAsppDSFCA(nn.Module):
    """ASPP → upsample → DSF → CA(fusion) → block2."""

    def __init__(
        self,
        smp_decoder: nn.Module,
        aspp_channels: int,
        *,
        c2_channels: int,
        c3_channels: int,
        fusion_channels: int,
    ) -> None:
        super().__init__()
        self.aspp = smp_decoder.aspp
        self.up = smp_decoder.up
        self.block2 = smp_decoder.block2

        self.dense_skip = DenseSkipFusion(
            c2_channels=c2_channels,
            c3_channels=c3_channels,
            aspp_channels=aspp_channels,
            out_channels=fusion_channels,
        )
        self.fusion_ca = CoordAttention(fusion_channels, reduction=32)

    def forward(self, *features: torch.Tensor) -> torch.Tensor:
        aspp_features = self.aspp(features[-1])
        aspp_features = self.up(aspp_features)
        fused = self.dense_skip(features[2], features[3], aspp_features)
        fused = self.fusion_ca(fused)
        return self.block2(fused)


class DeepLabV3PlusDSFCA(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.encoder = backbone.encoder
        aspp_channels = backbone.decoder.aspp[1][0].in_channels
        fusion_channels = backbone.decoder.block2[0][1].in_channels

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 32, 32)
            enc_feats = self.encoder(dummy)
        c2_channels = enc_feats[2].shape[1]
        c3_channels = enc_feats[3].shape[1]

        self.decoder = DeepLabV3PlusDecoderWithAsppDSFCA(
            backbone.decoder,
            aspp_channels,
            c2_channels=c2_channels,
            c3_channels=c3_channels,
            fusion_channels=fusion_channels,
        )
        self.segmentation_head = backbone.segmentation_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        decoded = self.decoder(*features)
        return self.segmentation_head(decoded)


def _conv_bn_relu(in_ch: int, out_ch: int, *, kernel_size: int = 1) -> nn.Sequential:
    padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


def _separable_conv(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False),
        nn.Conv2d(in_ch, out_ch, 1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )
