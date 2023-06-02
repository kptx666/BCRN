from torch import nn
from model.blocks import blueprint_conv_layer
from model.blocks import Blocks
from model.blocks import ESA
from model.blocks import pixelshuffle_block
import torch
from bsconv.pytorch import BSConvU
from model.blocks import activation
from model.blocks import CCALayer


class BluePrintShortcutBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = blueprint_conv_layer(in_channels, out_channels, kernel_size)
        self.convNextBlock = Blocks(out_channels, kernel_size)
        self.esa = ESA(out_channels, BSConvU)
        self.cca = CCALayer(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.convNextBlock(x)
        x = self.esa(x)
        x = self.cca(x)
        return x


class BluePrintConvNeXt_SR(nn.Module):
    def __init__(self, upscale_factor=2):
        super().__init__()
        self.conv1 = blueprint_conv_layer(3, 64, 3)
        self.convNext1 = BluePrintShortcutBlock(64, 64, 3)
        self.convNext2 = BluePrintShortcutBlock(64, 64, 3)
        self.convNext3 = BluePrintShortcutBlock(64, 64, 3)
        self.convNext4 = BluePrintShortcutBlock(64, 64, 3)
        self.convNext5 = BluePrintShortcutBlock(64, 64, 3)
        self.convNext6 = BluePrintShortcutBlock(64, 64, 3)
        # self.convNext7 = BluePrintShortcutBlock(64, 64, 3)
        # self.convNext8 = BluePrintShortcutBlock(64, 64, 3)
        self.conv2 = blueprint_conv_layer(64*6, 64, 3)
        self.upsample_block = pixelshuffle_block(64, 3, upscale_factor)
        self.activation = activation(act_type='gelu')

    def forward(self, x):
        out_fea = self.conv1(x)
        out_C1 = self.convNext1(out_fea)
        out_C2 = self.convNext2(out_C1)
        out_C3 = self.convNext3(out_C2)
        out_C4 = self.convNext4(out_C3)
        out_C5 = self.convNext5(out_C4)
        out_C6 = self.convNext6(out_C5)
        # out_C7 = self.convNext7(out_C6)
        # out_C8 = self.convNext8(out_C7)
        out_C = self.activation(self.conv2(torch.cat([out_C1, out_C2, out_C3, out_C4, out_C5, out_C6], dim=1)))
        out_lr = out_C + out_fea
        output = self.upsample_block(out_lr)
        return output


