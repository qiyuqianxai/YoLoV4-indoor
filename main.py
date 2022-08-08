import os
import time
import copy
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from nets.yolo4 import YoloBody,froze_params
from nets.yolo_training import YOLOLoss, Generator
from matplotlib import pyplot as plt
from PIL import Image


# 定义展示图片函数并展示几张图片
def show_image(image_path):
    plt.figure(figsize=(16, 10))
    plt.xticks([])
    plt.yticks([])
    image = Image.open(image_path)
    plt.imshow(image)

#---------------------------------------------------#
#   获得类和先验框
#---------------------------------------------------#
print("定义函数")
def get_classes(classes_path):
    # 读取类别文件
    with open(classes_path) as f:
        class_names = f.readlines()
    # 获取分类并返回
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    # 读取先验框文件
    with open(anchors_path) as f:
        anchors = f.readline()
    # 获取先验框并返回
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape([-1,3,2])[::-1,:,:]
    
# if __name__ == "__main__":
#     main()
