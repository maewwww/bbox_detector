import torch
import numpy as np
import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt

import torchvision
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Pad
from torchvision.io import decode_image
from torchinfo import summary as model_summary

gpu = torch.device('cuda:0')

def headless_resnet():
    resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
    return nn.Sequential(*list(resnet.children())[:-1])

class double_resnet(nn.Module):
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
    
def pad_to_500(img):
    down = 500 - img.shape[1]
    right = 500 - img.shape[2]
    pad = Pad((0,0,right,down))
    return pad(img)
        
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
        obj_name = img_name[:-3] + "png"
        img_path = os.path.join(self.img_dir, img_name)
        obj_path = os.path.join(self.obj_dir, obj_name)       
        image = decode_image(img_path).type(torch.float32)
        obj = decode_image(obj_path).type(torch.float32)
        label = torch.tensor([self.bbox_label.iloc[idx, i] for i in range(1,5)]).type(torch.float32)
        if self.transform:
            image = pad_to_500(self.transform(image))
            obj = self.transform(obj)
        if self.target_transform:
            label = self.target_transform(label)
        example = torch.cat((image,obj),0)
        
        return example.to(gpu), label.to(gpu)

    def get_imgname(self, idx):
        return self.bbox_label.iloc[idx, 0]
    
def get_imgname(revoc, img):
    for idx in range(len(revoc)):
        ds_img, _ = revoc[idx]
        if torch.equal(img, ds_img):
            return revoc.get_imgname(idx)

resnet_preprocess = torchvision.transforms.Compose([
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])