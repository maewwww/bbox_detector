import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import json

def compute_iou(xmin, ymin, xmax, ymax, xmin2, ymin2, xmax2, ymax2):
    if xmin2 > xmax2 or ymin2 > ymax2:
        return 0
    A1 = (xmax - xmin) * (ymax - ymin)
    A2 = (xmax2 - xmin2) * (ymax2 - ymin2)
    Ai = (min(xmax,xmax2) - max(xmin,xmin2)) * (min(ymax,ymax2) - max(ymin,ymin2))
    return max(0, Ai / (A1 + A2 - Ai))

def get_corners(x_origin, y_origin, height, width):
    xmin = x_origin - (width/2)
    xmax = x_origin + (width/2)
    ymin = y_origin - (height/2)
    ymax = y_origin + (height/2)
    return xmin, ymin, xmax, ymax

# Read result from filename (csv) and return a list containing IOU values (float)
def get_iou_list(file_path, alt_label=False):
    with open(file_path, "r") as f:
        file_csv = f.readlines()
    result = []
    for line in file_csv:
        _, label_xmin, label_ymin, label_xmax, label_ymax, xmin, ymin, xmax, ymax = line.split(",")
        ymax = ymax.strip()
        #convert coordinates to int
        label_xmin, label_ymin, label_xmax, label_ymax, xmin, ymin, xmax, ymax = [int(x) for x in [label_xmin, label_ymin, label_xmax, label_ymax, xmin, ymin, xmax, ymax]]
        
        if alt_label:
            xmin, ymin, xmax, ymax = get_corners(xmin, ymin, xmax, ymax)
            label_xmin, label_ymin, label_xmax, label_ymax = get_corners(label_xmin, label_ymin, label_xmax, label_ymax)
        result.append(compute_iou(label_xmin, label_ymin, label_xmax, label_ymax, xmin, ymin, xmax, ymax))
    return result

# plot a histogram of IOU distribution for each result in the list
def plot_iou_dist(iou_lists, names, bins=30, save=True):
    assert len(iou_lists) == len(names)
    nrows = len(names)
    fig, axes = plt.subplots(nrows=nrows, ncols=1)
    fig.tight_layout()

    for i in range(nrows):
        plt_idx = i + 1
        ious, name = iou_lists[i], names[i]
        plt.subplot(nrows, 1, plt_idx)
        plt.hist(ious, bins=bins)
        plt.title(f"IOU distribution of {name}")
        plt.ylabel("count")
        #plt.xlabel("IOU")
        plt.xticks([0,0.2,0.4,0.6,0.8,1])
        if save:
            plt.savefig("histogram.png")
        plt.show()

# Count the number of entries that surpass the threshold
def count_correct(iou_lists,
                  thresholds=[0.5, 0.75, 0.8, 0.9]):
    result = [[]] * len(iou_lists) # Create a list with of empty lists
    for threshold in thresholds:
        for i in range(len(iou_lists)):
            result[i].append(
                len([x for x in iou_lists[i] if x >= threshold])
            )
    return result

def plot_correct_per_threshold(correct_list, names, save=True,
                               thresholds=[0.5, 0.75, 0.8, 0.9]):
    assert len(correct_list) == len(names)
    for i in range(len(names)):
        correct,name = correct_list[i], names[i]
        plt.plot(thresholds, correct, label=name)
    plt.ylabel("Correct predictions")
    plt.xlabel("IOU threshold")
    plt.title("Correct prediction @ IOU thresholds")
    plt.legend(loc="upper right")
    if save:
        plt.savefig("line.png")
    plt.show()


