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
from nets.yolo4 import YoloBody
from nets.yolo_training import YOLOLoss, Generator

#---------------------------------------------------#
#   获得类和先验框
#---------------------------------------------------#
def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape([-1,3,2])[::-1,:,:]


#---------------------------------------------------#
#   训练一个epoch
#---------------------------------------------------#
def fit_one_epoch(net, yolo_losses, epoch, epoch_size, epoch_size_val, gen,genval, Epoch, cuda, optimizer, lr_scheduler):
    total_loss = 0
    val_loss = 0
    print('\n' + '-' * 10 + 'Train one epoch.' + '-' * 10)
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Start Training.')
    net.train()
    for iteration in range(epoch_size):
        start_time = time.time()
        images, targets = next(gen)
        with torch.no_grad():
            if cuda:
                images = Variable(torch.from_numpy(images).type(torch.FloatTensor)).cuda()
                targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]
            else:
                images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
                targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]
        optimizer.zero_grad()
#         with torch.no_grad():
#             outputs = net(images)
        outputs = net(images)
        losses = []
        for i in range(3):
            loss_item = yolo_losses[i](outputs[i], targets)
            losses.append(loss_item[0])
        loss = sum(losses)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss
        waste_time = time.time() - start_time
        if iteration == 0 or (iteration+1) % 10 == 0:
            print('step:' + str(iteration+1) + '/' + str(epoch_size) + ' || Total Loss: %.4f || %.4fs/step' % (total_loss/(iteration+1), waste_time))
    print('Finish Training.')
    '''        
    print('Start Validation.')
    net.eval()
    for iteration in range(epoch_size_val):
        images_val, targets_val = next(genval)

        with torch.no_grad():
            if cuda:
                images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor)).cuda()
                targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets_val]
            else:
                images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor))
                targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets_val]
            optimizer.zero_grad()
            outputs = net(images_val)
            losses = []
            for i in range(3):
                loss_item = yolo_losses[i](outputs[i], targets_val)
                losses.append(loss_item[0])
            loss = sum(losses)
            val_loss += loss
    print('Finish Validation')
    '''
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss/(epoch_size+1), val_loss/(epoch_size_val+1)))
    
    return total_loss/(epoch_size+1), val_loss/(epoch_size_val+1)
	
#-------------------------------#
#   输入的shape大小
#   显存比较小可以使用416x416
#   显存比较大可以使用608x608
#-------------------------------#
# input_shape = (416,416)
input_shape = (608, 608)

#-------------------------------#
#   tricks的使用设置
#-------------------------------#
Cosine_lr = True
mosaic = True
# 用于设定是否使用cuda
Cuda = True
smoooth_label = 0.03

#-------------------------------#
#   获得训练集和验证集的annotations
#   
#-------------------------------#
train_annotation_path = 'model_data/ele_train.txt'
val_annotation_path = 'model_data/ele_test.txt'

#-------------------------------#
#   获得先验框和类
#-------------------------------#
anchors_path = 'model_data/yolo_anchors.txt'
classes_path = 'model_data/mask_classes.txt'
class_names = get_classes(classes_path)
anchors = get_anchors(anchors_path)
num_classes = len(class_names)

# 创建模型
model = YoloBody(len(anchors[0]), num_classes)
model_path = "model_data/yolov4_coco_pretrained_weights.pth"
#model_path = "model_data/yolov4_maskdetect_weights0.pth"
# model_path = "model_data/yolov4_maskdetect_weights626B.pth"
# 加快模型训练的效率
print('Loading pretrained model weights.')
model_dict = model.state_dict()
pretrained_dict = torch.load(model_path)
pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
print('Finished!')

if Cuda:
    net = torch.nn.DataParallel(model)
    cudnn.benchmark = True
    net = net.cuda()
else:
    net = torch.nn.DataParallel(model)

# 建立loss函数
yolo_losses = []
for i in range(3):
    yolo_losses.append(YOLOLoss(np.reshape(anchors, [-1,2]), num_classes, \
                                (input_shape[1], input_shape[0]), smoooth_label, Cuda))
# read train lines and val lines
with open(train_annotation_path) as f:
    train_lines = f.readlines()
with open(val_annotation_path) as f:
    val_lines = f.readlines()
num_train = len(train_lines)
num_val = len(val_lines)

#------------------------------------#
#   先冻结backbone训练
#------------------------------------#
lr = 1e-3
Batch_size = 4
Init_Epoch = 0
Freeze_Epoch = 300
        
optimizer = optim.Adam(net.parameters(), lr, weight_decay=5e-4)
if Cosine_lr:
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
else:
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

gen = Generator(Batch_size, train_lines, (input_shape[0], input_shape[1])).generate(mosaic = mosaic)
gen_val = Generator(Batch_size, val_lines, (input_shape[0], input_shape[1])).generate(mosaic = False)
                        
epoch_size = int(max(1, num_train//Batch_size//2.5)) if mosaic else max(1, num_train//Batch_size)
epoch_size_val = num_val//Batch_size
for param in model.backbone.parameters():
    param.requires_grad = False

best_loss = 99999999.0
best_model_weights = copy.deepcopy(net.state_dict())
for epoch in range(Init_Epoch, Freeze_Epoch):
    total_loss, val_loss = fit_one_epoch(net, yolo_losses, epoch, epoch_size, epoch_size_val, gen, gen_val, 
                                         Freeze_Epoch, Cuda, optimizer, lr_scheduler)
    if total_loss < best_loss:
        best_loss = total_loss
        best_model_weights = copy.deepcopy(model.state_dict())
    with open('loss_fro101.csv', mode='a+') as total_loss_file:
        total_loss_file.write(str(total_loss.item()) + '\n')
    #with open('val_loss.csv', mode='a+') as val_loss_file:
    #    val_loss_file.write(str(val_loss.item()) + '\n')
torch.save(best_model_weights, 'model_data/yolov4_ele_weights_OCT.pth')

#------------------------------------#
#   解冻backbone后训练
#------------------------------------#
lr = 1e-4
Batch_size = 4
Freeze_Epoch = 300
Unfreeze_Epoch = 600

optimizer = optim.AdamW(net.parameters(), lr, weight_decay=5e-4)
if Cosine_lr:
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
else:
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

gen = Generator(Batch_size, train_lines, (input_shape[0], input_shape[1])).generate(mosaic = mosaic)
gen_val = Generator(Batch_size, val_lines, (input_shape[0], input_shape[1])).generate(mosaic = False)
                        
epoch_size = int(max(1, num_train//Batch_size//2.5)) if mosaic else max(1, num_train//Batch_size)
epoch_size_val = num_val//Batch_size
for param in model.backbone.parameters():
    param.requires_grad = False

for epoch in range(Freeze_Epoch, Unfreeze_Epoch):
    total_loss, val_loss = fit_one_epoch(net, yolo_losses, epoch, epoch_size, epoch_size_val, gen, gen_val, 
                                         Unfreeze_Epoch, Cuda, optimizer, lr_scheduler)
    if total_loss < best_loss:
        best_loss = total_loss
        best_model_weights = copy.deepcopy(model.state_dict())
    with open('loss_unfro101.csv', mode='a+') as total_loss_file:
        total_loss_file.write(str(total_loss.item()) + '\n')
    #with open('val_loss.csv', mode='a+') as val_loss_file:
    #    val_loss_file.write(str(val_loss.item() + '\n')
torch.save(best_model_weights, 'model_data/yolov4_ele_weights_OCT.pth', _use_new_zipfile_serialization=False)