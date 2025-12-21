import torch
import numpy as np
import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import time
import math

import torchvision
import torch.nn as nn
from torch import flatten
from torch.nn.functional import kl_div, log_softmax
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Pad, Resize
from torchvision.io import decode_image
from torchinfo import summary as model_summary
from torchvision.models.segmentation import lraspp_mobilenet_v3_large
from torchvision.models.segmentation.lraspp import LRASPPHead

gpu = torch.device('cuda:0')

def n_slice(lists, n):
    if lists == []:
        return []
    return [lists[:n]] + n_slice(lists[n:], n)

class SigmoidWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        out = self.base_model(x)
        out["out"] = torch.sigmoid(out["out"])
        return out

def OneOutLRASPP():
    model = lraspp_mobilenet_v3_large(num_classes=1)

    old_in = model.backbone["0"][0]
    old_batch_norm1 = model.backbone["0"][1]
    old_conv2 = model.backbone["1"].block[0][0]
    old_bn2 = model.backbone["1"].block[0][1]
    old_conv3 = model.backbone["1"].block[1][0]
    old_bn3 = model.backbone["1"].block[1][1]
    old_conv4 = model.backbone["2"].block[0][0]

    new_in = nn.Conv2d(in_channels=6, out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    batch_norm1 = nn.BatchNorm2d(32, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
    conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
    bn2 = nn.BatchNorm2d(32, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
    conv3 = nn.Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
    bn3 = nn.BatchNorm2d(32, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
    conv4 = nn.Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)

    with torch.no_grad():
        new_in.weight[16:, 3:, :, :] = old_in.weight
        new_in.weight[:16, :3, :, :] = old_in.weight
        batch_norm1.weight = nn.parameter.Parameter( torch.cat((old_batch_norm1.weight, old_batch_norm1.weight)) )
        conv2.weight[16: , :, :, :] = old_conv2.weight
        conv2.weight[:16 , :, :, :] = old_conv2.weight
        bn2.weight = nn.parameter.Parameter( torch.cat((old_bn2.weight, old_bn2.weight)) )
        conv3.weight[16: , 16:, :, :] = old_conv3.weight
        conv3.weight[:16 , :16, :, :] = old_conv3.weight
        bn3.weight = nn.parameter.Parameter( torch.cat((old_bn3.weight, old_bn3.weight)) )
        conv4.weight[: , 16:, :, :] = old_conv4.weight
        conv4.weight[: , :16, :, :] = old_conv4.weight
    
    model.backbone["0"][0] = new_in
    model.backbone["0"][1] = batch_norm1
    model.backbone["1"].block[0][0] = conv2
    model.backbone["1"].block[0][1] = bn2
    model.backbone["1"].block[1][0] = conv3
    model.backbone["1"].block[1][1] = bn3
    model.backbone["2"].block[0][0] = conv4


    #model.classifier = LRASPPHead(low_channels=128, high_channels=960, num_classes=1, inter_channels=256)

    return SigmoidWrapper(model)





def headless_resnet():
    resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
    return nn.Sequential(*list(resnet.children())[:-1])

class DoubleResnet50(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.resnet1 = headless_resnet()
        self.resnet2 = headless_resnet()

        self.fc1 = nn.Linear(4096, 2048)
        self.fc2 = nn.Linear(2048, 4)

    def forward(self, x):
        x1, x2 = torch.split(x, 3, dim=1)
        
        f1 = self.resnet1(x1)  # (N, 2048, 1, 1)
        f2 = self.resnet2(x2)  # (N, 2048, 1, 1)

        f1 = torch.flatten(f1, 1)  # (N, 2048)
        f2 = torch.flatten(f2, 1)  # (N, 2048)

        f = torch.cat((f1, f2), dim=1)  # (N, 4096)
        f = self.fc1(f) # (N, 2048)
        o = self.fc2(f) # (N, 4)

        return o

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        #print(f"\nDown module with in_c:{self.in_channels} and out_c:{self.out_channels}")
        #print(f"input shape: {x.shape}")
        o = self.maxpool_conv(x)
        #print(f"output shape: {o.shape}")
        return o


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        print(f"\nUp module, in_c={self.in_channels}, out_c={self.out_channels}")
        print(f"x1 shape BEFORE up {x1.shape}")
        x1 = self.up(x1)
        print(f"x1 shape AFTER up {x1.shape}")
        print(f"x2 shape AFTER up {x2.shape}")
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        print(f"x1 PATCHED shape {x1.shape}")
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)

        print(f"concatted X: {x.shape}")
        return self.conv(x)
    
class MulUp(nn.Module):
    """Up module with multiple inputs"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.obj_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.bg_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels * 2, out_channels, in_channels // 2)
        else:
            self.obj_up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.bg_up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels * 2, out_channels, mid_channels=((in_channels+out_channels)//2))

    def forward(self, x1, x2, x3, x4):
        #print(f"\nUp module, in_c={self.in_channels}, out_c={self.out_channels}")
        #print(f"x1 shape BEFORE up {x1.shape}")
        #print(f"x3 shape BEFORE up {x3.shape}")
        x1 = self.obj_up(x1)
        x3 = self.bg_up(x3)
        #print(f"x1 shape AFTER up {x1.shape}")
        #print(f"x3 shape AFTER up {x3.shape}")
        #print(f"x2 shape {x2.shape}")
        # input is CHW
        """diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x3 = F.pad(x3, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        """
        #print(f"x1 PATCHED shape {x1.shape}")
        #print(f"x3 PATCHED shape {x3.shape}")
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1, x4, x3], dim=1)
        del x1
        del x2
        del x3
        del x4
        torch.cuda.empty_cache()

        #print(f"concatted X: {x.shape}")
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

def base_vit_6_channels():
    "modified ViT_B_16 with 6 input channels and 4 output nodes"
    model = torchvision.models.vit_b_16(weights="IMAGENET1K_SWAG_E2E_V1") 
    new_conv_proj = torch.nn.modules.Conv2d(6, 768, kernel_size=(16, 16), stride=(16, 16))
    new_head = torch.nn.modules.Linear(in_features=768, out_features=4, bias=True)
    old_conv_weight = model.conv_proj.weight
    with torch.no_grad():
        new_conv_proj.weight[:, :3, :, :], new_conv_proj.weight[:, 3:, :, :] = old_conv_weight,old_conv_weight
    model.conv_proj = new_conv_proj
    model.heads.head = new_head
    return model

class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2048, 1024)
        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()
        self.out = nn.Linear(1024, 2)
        

    def forward(self, x):
        x = self.fc(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.out(x)
        return x
    
class ResnetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torchvision.models.resnet101(weights="DEFAULT")        

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        #x = self.backbone.avgpool(x)
        return x

class fusion_v2(nn.Module):
    def __init__(self):
        super(fusion_v2, self).__init__()
        #self.vit = torchvision.models.vit_b_16(weights="IMAGENET1K_SWAG_E2E_V1")
        #self.vit.head = torch.nn.Identity()
        self.resnet = torchvision.models.resnet101(weights="DEFAULT")
        self.resnet.fc = nn.Identity()

        self.en = torchvision.models.efficientnet_v2_s(weights="IMAGENET1K_V1")
        self.en.classifier = torch.nn.Identity()
        #self.linear = torch.nn.Linear(2280, 1024, bias=True)
        #self.head = torch.nn.Linear(1024, 4, bias=True)

        self.head = nn.Sequential(
            nn.Linear(3328, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(1024, 4)
        )

    def forward(self, x):
        bg = x[:, :3, :, :]
        fg = x[:, 3:, :, :]

        bg = self.resnet(bg)
        fg = self.en(fg)

        x = torch.cat((bg,fg),1)
        
        #x = self.linear(x)
        x = self.head(x)
        return x
    
class fusion_v5(nn.Module):
    def __init__(self):
        super(fusion_v5, self).__init__()
        self.scene_encoder = ResnetEncoder()
        self.obj_encoder = ENEncoder()
        self.decoder = torch.nn.Sequential(
            DecoderBlock(3328, 3328),
            DecoderBlock(3328, 3328),
            DecoderBlock(3328, 3328)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.head = nn.Sequential(
            nn.Linear(3328, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(1024, 4)
        )
        

    def forward(self, x):
        bg = x[:, :3, :, :]
        fg = x[:, 3:, :, :]

        bg = self.scene_encoder(bg)
        fg = self.obj_encoder(fg)

        x = torch.cat((bg,fg),1)
        x = self.decoder(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.head(x)
        return x
    
class ENEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.en = torchvision.models.efficientnet_v2_s(weights="IMAGENET1K_V1")
        

    def forward(self, x):
        x = self.en.features(x)
        #x = self.en.avgpool(x)
        # x = torch.flatten(x,start_dim=-2)
        # x = torch.permute(x, (0,2,1))
        #print("EN output:", x.mean(), x.std(), x.max())
        return x
    
class ViTEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = torchvision.models.vit_b_16(weights="IMAGENET1K_SWAG_E2E_V1")

    def forward(self, x):
        if torch.isnan(x).any():
            print("nan in VIT before process")
        x = self.vit._process_input(x)
        if torch.isnan(x).any():
            print("nan in VIT after process")
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        if torch.isnan(batch_class_token).any():
            print("nan in VIT in batch class token")
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.vit.encoder(x)
        if torch.isnan(x).any():
            print("nan in VIT after encoder")
        
        #print("VIT output:", x.mean(), x.std(), x.max())
        return x
    
class DecoderBlock(nn.Module):
    def __init__(self, d, o):
        super().__init__()
        i = int(d/4)
        self.conv1 = nn.Conv2d(d, i, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(i, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2 = nn.Conv2d(i, i, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(i, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3 = nn.Conv2d(i, o, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(o, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        if o < d:
            self.upsample = nn.Sequential(
                nn.Conv2d(d, o, kernel_size=(1, 1), stride=(1, 1), bias=False),
                nn.BatchNorm2d(o, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
        else:
            self.upsample = None

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        if self.upsample is not None:
            identity = self.upsample(identity)
        return self.relu(x + identity)
    
class CrossAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.fg_norm = nn.LayerNorm(768)
        self.fg_norm2 = nn.LayerNorm(768)
        self.mha = nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(768, 3072),
            nn.GELU(),
            # dropout
            nn.Linear(3072, 768)
        )
    def forward(self, fg, bg):
        #fg = self.fg_norm(fg)
        #bg = self.bg_norm(bg)
        
        attn_out, _ = self.mha(fg, bg, bg, need_weights=False)
        #print(w.mean().item(), w.std().item())

        fg = fg + attn_out
        fg = self.fg_norm(fg)
        fg = fg + self.mlp(fg)

        return self.fg_norm2(fg)
    
class Decoder(nn.Module):
    def __init__(self, num_queries, N):
        super().__init__()
        self.N = N
        self.num_queries = 1
        self.query = nn.Embedding(num_queries, 768)
        self.self_attn = nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True)
        self.norm2 = nn.LayerNorm(768)
        self.norm3 = nn.LayerNorm(768)
        self.norm4 = nn.LayerNorm(768)
        self.mlp2 = nn.Sequential(
            nn.Linear(768, 2048),
            nn.ReLU(),
            # dropout
            nn.Linear(2048, 768)
        )

    def forward(self, fg):
        q = self.query.weight.unsqueeze(1).expand(-1, fg.shape[0] ,-1).transpose(0, 1) # (N, q, d)

        s_attn_out, _ = self.self_attn(q, q, q, need_weights=False)
        q = q + s_attn_out
        q = self.norm2(q)
        c_attn_out, _ = self.cross_attn(q, fg, fg, need_weights=False)
        #print(w.mean().item(), w.std().item())
        q = q + c_attn_out
        q = q + self.mlp2(q)

        return self.norm4(q)

    
class fusion_v3(nn.Module):
    def __init__(self, num_queries, batch_size):
        super(fusion_v3, self).__init__()
        self.vit = ViTEncoder()
        self.en = ENEncoder()
        self.attn = CrossAttention()
        self.decoder = Decoder(num_queries=num_queries, N=batch_size)
        self.head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4)
        )
        self.linear = nn.Linear(1280, 768)

    def forward(self, x):

        # N = batch size
        # D = embedding dim = 768     
        # Q = number of queries   

        bg = x[:, :3, :, :] # (N, 3, H, W)
        fg = x[:, 3:, :, :] # (N, 3, H, W)

        bg = self.vit(bg) # (N, 577, D)
        fg = self.en(fg) # (N, 144, 1280)
        fg = self.linear(fg) # (N, 144, D)
        #print("linear output:", fg.mean(), fg.std(), fg.max())
        fg = self.attn(fg, bg) # (N, 144, D)
        #print("self attn:", fg.mean(), fg.std(), fg.max())

        query = self.decoder(fg) # (N, Q, D)
        #print("query attn:", query.mean(), query.std(), query.max())
        o = self.head(query) # (N, Q, 4)
        #print("query attn:", o.mean(), o.std(), o.max())

        return o
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class DinoEncoder(nn.Module):
    def __init__(self):
        super(DinoEncoder, self).__init__()
        self.dino = dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    
    def forward(self, x):
        with torch.cuda.amp.autocast(dtype=torch.float16):
            y = self.dino(x, is_training=True)["x_norm_patchtokens"]
        return y
    
class DenseDecoder(nn.Module):
    def __init__(self, seq, dim):
        super(DenseDecoder, self).__init__()
        self.pre_flat = nn.Linear(dim, 256)
        self.dropout = nn.Dropout(p=0.1)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(256*seq, 512)
        self.dropout1 = nn.Dropout(p=0.2)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(512, 512)
        self.dropout2 = nn.Dropout(p=0.2)
        self.relu2 = nn.ReLU()
        self.out = nn.Linear(512, 4)

    def forward(self, x):
        x = self.pre_flat(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        #x = self.linear3(x)
        return self.out(x)
    
class fusion_v4(nn.Module):
    def __init__(self, num_queries, batch_size):
        super(fusion_v4, self).__init__()
        self.scene_encoder = DinoEncoder()
        self.obj_encoder = DinoEncoder()
        self.decoder = DenseDecoder(784, 768)
        self.fusion1 = CrossAttention()
        self.fusion2 = CrossAttention()

    def forward(self, x):

        # N = batch size
        # D = embedding dim = 768     
        # Q = number of queries  
        
        bg = x[:, :3, :, :] # (N, 3, H, W)
        fg = x[:, 3:, :, :] # (N, 3, H, W)

        bg = self.scene_encoder(bg) # (N, 577, D)
        fg = self.obj_encoder(fg) # (N, 577, D)

        #print("linear output:", fg.mean(), fg.std(), fg.max())
        fg = self.fusion1(fg, bg) # (N, 144, D)
        fg = self.fusion2(fg, bg) # (N, 144, D)
        o = self.decoder(fg)
        #o = self.head(q)
        return o





def pad_to_n(img, n=500):
    down = n - img.shape[1]
    right = n - img.shape[2]
    pad = Pad((0,0,right,down))
    return pad(img)

def square_pad(img):
    h, w = img.shape[1], img.shape[2]
    if h > w:
        return Pad((0, 0, h - w, 0))(img)
    else:
        return Pad((0, 0, 0, w - h))(img)

def pad_to_640(img):
    return pad_to_n(img, 640)

def normalize_mask(t):
    return torch.div(t, 255)

LRASPP_preprocess = torchvision.transforms.Compose([
    pad_to_640,
    Resize([520,520]),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

LRASPP_target_preprocess = torchvision.transforms.Compose([
    pad_to_640,
    Resize([520,520]),
    normalize_mask
])

unet_preprocess = torchvision.transforms.Compose([
    pad_to_640,
    Resize([512,512]),
    normalize_mask
])

unet_target_preprocess = torchvision.transforms.Compose([
    pad_to_640,
    Resize([512,512],interpolation=torchvision.transforms.InterpolationMode.NEAREST),
    normalize_mask
])

vit_transform = torchvision.transforms.Compose([
    pad_to_640,
    Resize([384,384], interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
    #normalize_mask,
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

efficient_net_transform = torchvision.transforms.Compose([
    Resize((392,)),
    #normalize_mask,
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

vit_transform_2 = torchvision.transforms.Compose([
    square_pad,
    Resize([384,384], interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
    #normalize_mask,
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
        
# Subset of VOC'2012 with object removed that CONTAIN 2 IMAGES
# We stack the tensors for the image and obj together
class TwoChannelCustomDataset(Dataset):
    def __init__(self, label_dir, img_dir, obj_dir, transform=None, target_transform=None, label_names=("img_name","xmin","ymin","xmax","ymax")):
        self.bbox_label = pd.read_csv(label_dir, names=label_names)
        self.img_dir = img_dir
        self.obj_dir = obj_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.bbox_label)

    def __getitem__(self, idx):    
        img_name = self.bbox_label.iloc[idx, 0]
        obj_name = img_name[:-3] + "jpg"
        img_path = os.path.join(self.img_dir, img_name)
        obj_path = os.path.join(self.obj_dir, obj_name)       
        image = decode_image(img_path).type(torch.float32)
        obj = decode_image(obj_path).type(torch.float32)
        label = torch.tensor([self.bbox_label.iloc[idx, i] for i in range(1,5)]).type(torch.float32)
        if self.transform:
            image = pad_to_n(self.transform(image), 500)
            obj = self.transform(obj)
        if self.target_transform:
            label = self.target_transform(label)
        example = torch.cat((image,obj),0)
        
        return example.to(gpu), label.to(gpu), img_name

    def get_imgname(self, idx):
        return self.bbox_label.iloc[idx, 0]

# Torch dataset class for loading OPA with variable label length
class OPADataset(Dataset):
    def __init__(self, label_dir, img_dir, obj_dir, transform=None, target_transform=None):
        with open(label_dir, "r") as f:
            self.bbox_label = f.readlines()
        self.pad_label = 55 # length of label (see how label is padded in __getitem__)
        self.img_pad_size = 640
        self.img_size = [384, 384] # target size of image
        self.img_dir = img_dir
        self.obj_dir = obj_dir
        self.transform = transform
        self.target_transform = target_transform
        self.resize_ratio = self.img_size[0] / self.img_pad_size

    def __len__(self):
        return len(self.bbox_label)

    def __getitem__(self, idx):
        label_line = self.bbox_label[idx].split(",")
        img_id = label_line[1]
        obj_id = label_line[0]
        img_name = img_id + ".jpg"
        obj_name = obj_id + ".jpg"
        cat = label_line[2]
        name = obj_id + "c" + img_id

        img_path = os.path.join(self.img_dir, cat, img_name)
        obj_path = os.path.join(self.obj_dir, cat, obj_name)       
        image = decode_image(img_path).type(torch.float32)
        obj = decode_image(obj_path).type(torch.float32)

        label = label_line[3:]
        label = [float(x) for x in label]
        label = n_slice(label, 4)
        for sublabel in label:
            for i, element in enumerate(sublabel):
                sublabel[i] = element * self.resize_ratio
        while len(label) < self.pad_label:
            label.append([float('inf')]*4)
        label = torch.tensor(label).type(torch.float32)
        # Dimension of (each) label is (n, 4) where we pad label to conain n = self.pad_label bboxes.
        # Example: [[x1, y1, w1, h1],
        #           [x2, y2, w2, h2],
        #           ...
        #           [xn, yn, wn, hn]]

        resize = Resize(self.img_size)
        if self.transform:
            image = self.transform(image)
            obj = self.transform(obj)
        if self.target_transform:
            label = self.target_transform(label)
        #print(image.shape)
        #print(obj.shape)
        example = torch.cat((image,obj),0)
        
        return example.to(gpu), label.to(gpu), name
    
class OPADataset_2(Dataset):
    def __init__(self, label_dir, img_dir, obj_dir, transform=None, target_transform=None, obj_transform=None):
        with open(label_dir, "r") as f:
            self.bbox_label = f.readlines()
        self.pad_label = 55 # length of label (see how label is padded in __getitem__)
        self.img_pad_size = 640
        self.img_size = [384, 384] # target size of image
        self.img_dir = img_dir
        self.obj_dir = obj_dir
        self.transform = transform
        self.obj_transform = obj_transform
        self.target_transform = target_transform
        self.resize_ratio = self.img_size[0] / self.img_pad_size

    def __len__(self):
        return len(self.bbox_label)

    def __getitem__(self, idx):
        label_line = self.bbox_label[idx].split(",")
        img_id = label_line[1]
        obj_id = label_line[0]
        img_name = img_id + ".jpg"
        obj_name = obj_id + ".jpg"
        cat = label_line[2]
        name = obj_id + "c" + img_id

        img_path = os.path.join(self.img_dir, cat, img_name)
        obj_path = os.path.join(self.obj_dir, cat, obj_name)       
        image = decode_image(img_path).type(torch.float32)
        obj = decode_image(obj_path).type(torch.float32)

        label = label_line[3:]
        label = [float(x) for x in label]
        label = n_slice(label, 4)
        for sublabel in label:
            for i, element in enumerate(sublabel):
                sublabel[i] = element * self.resize_ratio
        while len(label) < self.pad_label:
            label.append([float('inf')]*4)
        label = torch.tensor(label).type(torch.float32)
        # Dimension of (each) label is (n, 4) where we pad label to conain n = self.pad_label bboxes.
        # Example: [[x1, y1, w1, h1],
        #           [x2, y2, w2, h2],
        #           ...
        #           [xn, yn, wn, hn]]

        if self.transform:
            image = self.transform(image)
            obj = self.obj_transform(obj)
        if self.target_transform:
            label = self.target_transform(label)
        #print(image.shape)
        #print(obj.shape)
        example = torch.cat((image,obj),0)
        
        return example.to(gpu), label.to(gpu), name
    
class OPADataset_3(Dataset):
    def __init__(self, label_dir, img_dir, obj_dir, transform=None, target_transform=None, obj_transform=None, classes=None):
        with open(label_dir, "r") as f:
            bbox_label = f.readlines()
        self.pad_label = 55 # length of label (see how label is padded in __getitem__)
        self.img_pad_size = 640
        self.img_size = [384, 384] # target size of image
        self.img_dir = img_dir
        self.obj_dir = obj_dir
        self.transform = transform
        self.obj_transform = obj_transform
        self.target_transform = target_transform
        self.resize_ratio = self.img_size[0] / self.img_pad_size
        new_label = []
        if classes is not None:
            for line in bbox_label:
                list_line = line.split(",")
                if list_line[2] in classes:
                    new_label.append(line)
            self.bbox_label = new_label
        else:
            self.bbox_label = bbox_label

        


    def __len__(self):
        return len(self.bbox_label)

    def __getitem__(self, idx):
        label_line = self.bbox_label[idx].split(",")
        img_id = label_line[1]
        obj_id = label_line[0]
        img_name = img_id + ".jpg"
        obj_name = obj_id + ".jpg"
        cat = label_line[2]
        name = obj_id + "c" + img_id

        img_path = os.path.join(self.img_dir, cat, img_name)
        obj_path = os.path.join(self.obj_dir, cat, obj_name)       
        image = decode_image(img_path).type(torch.float32)
        obj = decode_image(obj_path).type(torch.float32)

        label = label_line[3:]
        label = [float(x) for x in label]
        label = n_slice(label, 4)
        for sublabel in label:
            for i, element in enumerate(sublabel):
                sublabel[i] = element * self.resize_ratio
        while len(label) < self.pad_label:
            label.append([float('inf')]*4)
        label = torch.tensor(label).type(torch.float32)
        # Dimension of (each) label is (n, 4) where we pad label to conain n = self.pad_label bboxes.
        # Example: [[x1, y1, w1, h1],
        #           [x2, y2, w2, h2],
        #           ...
        #           [xn, yn, wn, hn]]

        if self.transform:
            image = self.transform(image)
            obj = self.obj_transform(obj)
        if self.target_transform:
            label = self.target_transform(label)
        #print(image.shape)
        #print(obj.shape)
        example = torch.cat((image,obj),0)
        
        return example.to(gpu), label.to(gpu), name
    
class BGDataset(Dataset):
    def __init__(self, label_dir, img_dir, obj_dir, transform=None, target_transform=None, obj_transform=None):
        with open(label_dir, "r") as f:
            self.bbox_label = f.readlines()
        self.img_pad_size = 640
        self.img_size = [320, 320] # target size of image
        self.img_dir = img_dir
        self.obj_dir = obj_dir
        self.transform = transform
        self.obj_transform = obj_transform
        self.target_transform = target_transform
        self.resize_ratio = self.img_size[0] / self.img_pad_size

    def __len__(self):
        return len(self.bbox_label)

    def __getitem__(self, idx):
        label_line = self.bbox_label[idx].split(",")
        cat = label_line[1]
        bg_id = label_line[0]
        label = label_line[2:]
        img_name = bg_id + ".jpg"

        img_path = os.path.join(self.img_dir, cat, img_name)    
        image = decode_image(img_path).type(torch.float32)

        label = [int(x) for x in label]
        label = n_slice(label, 2)
        #label.sort(key=lambda x: x[0] + x[1])
        for sublabel in label:
            if sublabel[0] == -1:
                break
            for i, element in enumerate(sublabel):
                sublabel[i] = element * self.resize_ratio
        label = torch.tensor(label).type(torch.float32)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        #print(image.shape)
        #print(obj.shape)
        
        return image.to(gpu), label.to(gpu), img_name
    
class OPADistDataset(Dataset):
    def __init__(self, label_dir, img_dir, obj_dir, mask_dir, transform=None, target_transform=None):
        with open(label_dir, "r") as f:
            self.bbox_label = f.readlines()
        self.img_dir = img_dir
        self.obj_dir = obj_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.bbox_label)

    def __getitem__(self, idx):
        label_line = self.bbox_label[idx].split(",")        
        cat = label_line[2]
        img_id = label_line[1]
        obj_id = label_line[0]
        img_name = img_id + ".jpg"
        obj_name = obj_id + ".jpg"
        comp_name = obj_id + "c" + img_id

        img_path = os.path.join(self.img_dir, cat, img_name)
        obj_path = os.path.join(self.obj_dir, cat, obj_name)    
        mask_path = os.path.join(self.mask_dir, cat, comp_name + ".jpg")
        image = decode_image(img_path).type(torch.uint8).float()
        obj = decode_image(obj_path).type(torch.uint8).float()
        mask = decode_image(mask_path).type(torch.uint8)

        if self.transform:
            image = self.transform(image)
            obj = self.transform(obj)
        if self.target_transform:
            label = self.target_transform(mask)
        example = torch.cat((image,obj),0)
        
        return example, label, comp_name
    
def get_imgname(revoc, img):
    for idx in range(len(revoc)):
        ds_img, _ = revoc[idx]
        if id(img) == id(ds_img):
            return revoc.get_imgname(idx)
        """
        if torch.equal(img, ds_img):
            return revoc.get_imgname(idx)"""

resnet_preprocess = torchvision.transforms.Compose([
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def matcher(outputs, targets):
    """
    Finds the optimal matching between the set of outputs and targets.

    This snippet is taken from the original code:
    https://github.com/facebookresearch/detr/blob/main/models/matcher.py

    outputs: This is a dict that contains at least these entries:
        1. "pred_logits": Tensor [batch_size, num_queries, num_classes]
        with the classification logits. 
        num_queries = 100 by default.
        2. "pred_boxes": Tensor [batch_size, num_queries, 4] with the
            predicted box coordinates.

    targets: This is a list of targets (len(targets) = batch_size), 
        where each target is a dict containing:
        1. "labels": Tensor [num_target_boxes] (where num_target_boxes
            is the number of ground-truth objects in the target) 
            containing the class labels
        2. "boxes": Tensor of dim [num_target_boxes, 4] containing the 
            target box coordinates
    """
    # Step 1. Prepare outputs and targets for matching.
    # By default num_queries = 100.
    # batch_size must be at least 1.
    batch_size, num_queries = outputs["pred_logits"].shape[:2]

    # Flatten to compute the cost matrices in a batch.
    # [batch_size * num_queries, num_classes].
    out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1) 
    # [batch_size * num_queries, 4].
    out_bbox = outputs["pred_boxes"].flatten(0, 1)

    # [num_target_boxes]
    tgt_ids = torch.cat([v["labels"] for v in targets])
    # [num_target_boxes]
    tgt_bbox = torch.cat([v["boxes"] for v in targets])


    # Step 2. Compute the classification and bounding box losses.
    # Compute the classification cost using 1 - proba[target class].
    # 1 is a constant that does not change the matching, 
    # and can be ommitted.
    # We only consider the object classes present in the current
    # batch of targets. 
    # [batch_size * num_queries, len(unique(tgt_ids))],
    # where len(unique(tgt_ids)) < num_classes.
    cost_class = -out_prob[:, tgt_ids]

    # Compute the L1 distance between boxes. The L1 distance is more 
    # robust to outliers than the L2 distance.
    cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1) # (N, q, 4)
    # Compute the giou cost betwen boxes.
    cost_giou = -generalized_box_iou(
        box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox)
    )

    # Final classification + bounding box cost matrix.
    C = cost_bbox + cost_class + cost_giou
    C = C.view(batch_size, num_queries, -1).cpu()


    # Step 3. Perform matching using the Hungarian algorithm 
    # (linear sum assignment). The resulting indices are the optimal
    # assignments between the predictions and targets. These indices
    # are passed on to the loss function downstream to calculate the
    # losses for back propagation.
    sizes = [len(v["boxes"]) for v in targets]
    indices = [
        linear_sum_assignment(c[i]) 
        for i, c in enumerate(C.split(sizes, -1))
    ]

    # Returns a list of size batch_size, containing tuples of 
    # (index_i, index_j) where:
    #   - index_i is the indices of the selected predictions (in order)
    #   - index_j is the indices of the corresponding selected targets 
    #     (in order)
    return [
        (
            torch.as_tensor(i, dtype=torch.int64), 
            torch.as_tensor(j, dtype=torch.int64)
        ) for i, j in indices
    ]

def matching_loss(pred, target, loss=None):
    """ pred: tensor (N, q, 4)
        target: tensor (N, q, 4)"""
    if loss is None:
        loss = torch.nn.MSELoss(reduction="sum")
    N, q = pred.shape[:2]
    batch_loss = list()
    """
    pred[:, :, [1, 3]] = pred[:, :, [3, 1]]
    target[:, :, [1, 3]] = target[:, :, [3, 1]]
    """
    for k in range(N):
        current_loss = list()
        _pred = pred[k]
        _tgt = target[k]
        loss_matrix = torch.zeros((q, q))
        for i in range(q):
            for j in range(q):
                if _tgt[j][0] == -1.0:
                    continue
                    #loss_matrix[i][j] = 0 
                else:
                    loss_matrix[i][j] = loss(_pred[i], _tgt[j]).item()
        row_idx, col_idx = scipy.optimize.linear_sum_assignment(loss_matrix)
        #non_zero_count = 0

        opt_pred = []
        opt_tgt = []
        for i in range(q):
            row, col = row_idx[i], col_idx[i]
            if _tgt[col][0] == -1.0:
                continue
            else:
                opt_pred.append(_pred[row])
                opt_tgt.append(_tgt[col])
        opt_pred = torch.stack(opt_pred, dim=0)
        opt_tgt = torch.stack(opt_tgt, dim=0)
        
        batch_loss.append(loss(opt_pred, opt_tgt))
    return sum(batch_loss) / N


def var_mse_min(pred, label):
    """perform mse for each sublabel in label and return the minimum one"""

    mse = torch.nn.MSELoss(reduction='sum')
    result = []
    
    for i in range(len(pred)):

        min_loss_value = float("inf")
        min_loss = None

        subpred = pred[i]
        sublabel = label[i]

        for candidate in sublabel:

            if candidate[0].item() == float("inf"):
                break
            
            #print(f"Calculating loss between {subpred} and {candidate}")
            loss = mse(subpred, candidate)
            loss_value = loss.item()

            if loss_value < min_loss_value:
                min_loss_value = loss_value
                min_loss = loss

        #print(min_loss)
        result.append(min_loss)
    return sum(result) / len(pred)

def var_ciou_min(pred, label):
    """test ciou for each sublabel in label and return the minimum one
    pred: (N, 4)
    label: (N, q, 4)"""

    pred[:, [1, 3]] = pred[:, [3, 1]]
    label[:, :, [1, 3]] = label[:, :, [3, 1]]

    loss = torchvision.ops.complete_box_iou_loss
    N = len(pred)
    batch_loss = list()
    for i in range(N):
        _pred = pred[i]
        _label = label[i]
        msk = torch.isfinite(_label).all(dim=1)
        _label = _label[msk]
        _pred = _pred.unsqueeze(0).expand(len(_label), -1)
        losses = loss(_pred, _label)
        batch_loss.append(losses.min())
    return sum(batch_loss) / N

    
    for i in range(len(pred)):

        min_loss_value = float("inf")
        min_loss = None

        subpred = pred[i]
        sublabel = label[i]

        for candidate in sublabel:

            if candidate[0].item() == float('inf'):
                break
            
            #print(f"Calculating loss between {subpred} and {candidate}")
            loss = mse(subpred, candidate)

            if loss.item() < min_loss_value:
                min_loss_value = loss.item()
                min_loss = loss

        #print(min_loss)
        result.append(min_loss)
    return sum(result)

to_logspace = lambda x: flatten(log_softmax(flatten(x, 2), 2), 1)
def kldiv(x, y):
    x = to_logspace(x)
    y = to_logspace(y)
    return sum([kl_div(input=x[i], target=y[i],log_target=True, reduction="sum") for i in range(len(x))])

def dice_loss(pred, target, smooth=1):
    """
    Computes the Dice Loss for binary segmentation.
    https://medium.com/data-scientists-diary/implementation-of-dice-loss-vision-pytorch-7eef1e438f68
    Args:
        pred: Tensor of predictions (batch_size, 1, H, W).
        target: Tensor of ground truth (batch_size, 1, H, W).
        smooth: Smoothing factor to avoid division by zero.
    Returns:
        Scalar Dice Loss.
    """
    # Apply sigmoid to convert logits to probabilities
    pred = torch.sigmoid(pred)
    
    # Calculate intersection and union
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    
    # Compute Dice Coefficient
    dice = (2. * intersection + smooth) / (union + smooth)
    
    # Return Dice Loss
    return 1 - dice.mean()


def train_loop(dataloader, model, loss_fn, optimizer, result_dict, t=None):
    batch_num = len(dataloader)
    model.train()
    total_loss = 0
    for batch, (X, y, _) in enumerate(dataloader):
        # Compute prediction and loss
        start_forward = time.time()
        X = X.to(gpu)
        y = y.to(gpu)
        if batch % 100 == 0:
            print("batch num: ",batch, " out of ",batch_num)
        pred = model(X)
        #print("forward pass: ", time.time() - start_forward)
        start_loss = time.time()
        loss = loss_fn(pred, y)
        #print("calculate loss: ", time.time() - start_loss)
        

        # Backpropagation
        #loss.requires_grad = True
        #loss.to(gpu)
        #print(loss)
        #print()
        start_backward = time.time()
        loss.backward()
        total_loss += loss.item()
        #print("backward: ", time.time() - start_backward)
        """
        for each_loss in loss:
            each_loss.backward(retain_graph=True)"""
        
        start_step = time.time()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        optimizer.zero_grad()
        #print("opt step: ", time.time() - start_step)
    print(f"epoch: {t + 1}, avg train loss: {total_loss/batch_num}")
    #print(f"pred shape: ",pred["out"].shape)
    #print(f"y shape: {y.shape}")
    if t is not None:
        result_dict[t] = {"train": total_loss/batch_num} # create dict because we run train BEFORE test in each epoch

def test_loop(dataloader, model, loss_fn,result_dict, t=None):
    model.eval()
    batch_num = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y,_ in dataloader:
            X = X.to(gpu)
            y = y.to(gpu)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()    
    print(f"epoch: {t + 1}, avg test loss: {test_loss/batch_num}")
    #print(f"pred shape: ",pred["out"].shape)
    #print(f"y shape: {y.shape}")
    if t is not None:
        result_dict[t]["test"] = test_loss/batch_num # edit dict because we run test AFTER train in each epoch

def save_eval(dataset, model, loss_fn, full_dataset=None, out_dir="output/"):
    # Do prediction one by one so we can save result
    model.eval()
    total_loss = 0

    with open(os.path.join(out_dir, "output.csv"), "a") as out_file:
    
        # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
        # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
        with torch.no_grad():
            for X, y, n in dataset:
                pred = model(X)
                out_name = n
                out_label = []
                out_pred = []
                """
                for x in X:
                    pass
                    #name = n
                    #name = get_imgname(full_dataset, x)
                    # name = "placeholder"
                    #out_name.append(name)
                for label in y:
                    label = [x for x in label if x != [float("inf")]*4]
                    xmin, ymin, xmax, ymax = [str(int(x)) for x in label.tolist()]
                    out_label.append(",".join([xmin, ymin, xmax, ymax]))"""
                for label in pred:
                    pred_xmin, pred_ymin, pred_xmax, pred_ymax = [str(int(x)) for x in label.tolist()]
                    out_pred.append(",".join([pred_xmin, pred_ymin, pred_xmax, pred_ymax]))
                for i in range(len(out_name)):
                    out_file.write(out_name[i] + "," + out_pred[i] + "\n")

def save_dist_eval(dataset, model, loss_fn, out_dir="output/"):
    model.eval()
    total_loss = 0    
    with torch.no_grad():
        for X, y, n in dataset:
            pred = model(X)
            for i, x in enumerate(pred):
                fn = os.path.join(out_dir, n[i] + ".png")
                x = torch.clamp(x[0], min=0,max=1)
                img = (x * 255).cpu().numpy().astype('uint8')
                cv2.imwrite(fn, img)
        
        