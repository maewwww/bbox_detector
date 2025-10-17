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

def n_slice(lists, n):
    if lists == []:
        return []
    return [lists[:n]] + n_slice(lists[n:], n)

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
    
def pad_to_n(img, n=500):
    down = n - img.shape[1]
    right = n - img.shape[2]
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
            image = pad_to_n(self.transform(image), 500)
            obj = self.transform(obj)
        if self.target_transform:
            label = self.target_transform(label)
        example = torch.cat((image,obj),0)
        
        return example.to(gpu), label.to(gpu), img_name

    def get_imgname(self, idx):
        return self.bbox_label.iloc[idx, 0]

"""x = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
x = torch.tensor(x)
for i in x: print(i)"""

# Torch dataset class for loading OPA with variable label length
class OPADataset(Dataset):
    def __init__(self, label_dir, img_dir, obj_dir, transform=None, target_transform=None):
        with open(label_dir, "r") as f:
            self.bbox_label = f.readlines()
        self.pad_label = 55
        self.img_dir = img_dir
        self.obj_dir = obj_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.bbox_label)

    def __getitem__(self, idx):
        label_line = self.bbox_label[idx].split(",")

        img_name = label_line[1] + ".jpg"
        obj_name = label_line[0] + ".jpg"
        cat = label_line[2]
        name = img_name + "c" + obj_name

        img_path = os.path.join(self.img_dir, cat, img_name)
        obj_path = os.path.join(self.obj_dir, cat, obj_name)       
        image = decode_image(img_path).type(torch.float32)
        obj = decode_image(obj_path).type(torch.float32)

        label = label_line[3:]
        label = [float(x) for x in label]
        label = n_slice(label, 4)
        while len(label) < self.pad_label:
            label.append([float('inf')]*4)
        label = torch.tensor(label).type(torch.float32)
        # Dimension of label is (n, 4) where we pad label to be n = self.pad_label bboxes.
        # Example: [[x1, y1, w1, h1],
        #           [x2, y2, w2, h2],
        #           ...
        #           [xn, yn, wn, hn]]

        if self.transform:
            image = pad_to_n(self.transform(image), 640)
            obj = pad_to_n(self.transform(obj), 640)
        if self.target_transform:
            label = self.target_transform(label)
        example = torch.cat((image,obj),0)
        
        return example.to(gpu), label.to(gpu), name

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
            
            #print(f"Calculating loss between {subpred} and {candidate}")
            loss = mse(subpred, candidate)

            if loss.item() < min_loss_value:
                min_loss_value = loss.item()
                min_loss = loss

        #print(min_loss)
        result.append(min_loss)
    return sum(result)

def train_loop(dataloader, model, loss_fn, optimizer, result_dict, t=None):
    batch_num = len(dataloader)
    model.train()
    total_loss = 0
    for batch, (X, y, _) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        # total_loss += loss.item() FIXME

        # Backpropagation
        #loss.requires_grad = True
        #loss.to(gpu)
        #print(loss)
        #print()
        loss.backward()
        """
        for each_loss in loss:
            each_loss.backward(retain_graph=True)"""
        optimizer.step()
        optimizer.zero_grad()
    print(f"epoch: {t + 1}, avg train loss: {total_loss/batch_num}")
    if t is not None:
        result_dict[t] = {"train": total_loss/batch_num} # create dict because we run train BEFORE test in each epoch

def test_loop(dataloader, model, loss_fn,result_dict, t=None):
    model.eval()
    batch_num = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y,_ in dataloader:
            pred = model(X)
            #test_loss += loss_fn(pred, y).item()    
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
        
        