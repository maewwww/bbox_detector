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
        obj_name = img_name[:-3] + "jpg"
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
        
        return example.to(gpu), label.to(gpu), img_name

    def get_imgname(self, idx):
        return self.bbox_label.iloc[idx, 0]
    
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

def train_loop(dataloader, model, loss_fn, optimizer, result_dict, t=None):
    batch_num = len(dataloader)
    model.train()
    total_loss = 0
    for batch, (X, y, _) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        total_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"epoch: {t + 1}, avg train loss: {total_loss/batch_num}")
    if t is not None:
        result_dict[t] = {"train": total_loss/batch_num} # create dict because we run train BEFORE test in each epoch

def test_loop(dataloader, model, loss_fn,result_dict, t=None):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    batch_num = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y,_ in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()    
    print(f"epoch: {t + 1}, avg test loss: {test_loss/batch_num}")
    if t is not None:
        result_dict[t]["test"] = test_loss # edit dict because we run test AFTER train in each epoch

def save_eval(dataset, model, loss_fn, full_dataset, out_dir="output/"):
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
                for x in X:
                    pass
                    #name = n
                    #name = get_imgname(full_dataset, x)
                    # name = "placeholder"
                    #out_name.append(name)
                for label in y:
                    xmin, ymin, xmax, ymax = [str(int(x)) for x in label.tolist()]
                    out_label.append(",".join([xmin, ymin, xmax, ymax]))
                for label in pred:
                    pred_xmin, pred_ymin, pred_xmax, pred_ymax = [str(int(x)) for x in label.tolist()]
                    out_pred.append(",".join([pred_xmin, pred_ymin, pred_xmax, pred_ymax]))
                for i in range(len(out_name)):
                    out_file.write(out_name[i] + "," + out_label[i] + "," + out_pred[i] + "\n")
        
        