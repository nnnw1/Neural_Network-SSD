import argparse
import os
import numpy as np
import time
import cv2

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

from dataset import *
from model import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true')
args = parser.parse_args()
#please google how to use argparse
#a short intro:
#to train: python main.py
#to test:  python main.py --test


class_num = 4 #cat dog person background

num_epochs = 100
batch_size = 32


boxs_default = default_box_generator([10,5,3,1], [0.2,0.4,0.6,0.8], [0.1,0.3,0.5,0.7])


#Create network
network = SSD(class_num)
network.cuda()
cudnn.benchmark = True

# TP = np.zeros([batch_size, 3])
# FP = np.zeros([batch_size, 3])
# FN_TP = np.zeros([batch_size, 3])

if not args.test:
    dataset = COCO("data/train/images/", "data/train/annotations/", class_num, boxs_default, train = True, image_size=320, val = False)
    dataset_test = COCO("data/train/images/", "data/train/annotations/", class_num, boxs_default, train = False, image_size=320, val = True)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=0)
    
    optimizer = optim.Adam(network.parameters(), lr = 1e-4)
    # optimizer = optim.SGD(network.parameters(), lr = 1e-3, momentum = 0.99)
    #feel free to try other optimizers and parameters.
    
    start_time = time.time()

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        #TRAINING
        network.train()

        avg_loss = 0
        avg_count = 0
        for i, data in enumerate(dataloader, 0):
            images_, ann_box_, ann_confidence_, ori_height, ori_width, name, _ = data
            images = images_.cuda()
            ann_box = ann_box_.cuda()
            ann_confidence = ann_confidence_.cuda()
            # print("training")
            optimizer.zero_grad()
            pred_confidence, pred_box = network(images)
            # pred_confidence = torch.softmax(pred_confidence, dim = 2)
            loss_net = SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box)
            loss_net.backward()
            optimizer.step()
            
            avg_loss += loss_net.data
            avg_count += 1

        print('[%d] time: %f train loss: %f' % (epoch, time.time()-start_time, avg_loss/avg_count))
        #append loss
        train_loss = avg_loss/avg_count
        train_loss = train_loss.detach().cpu()
        train_losses.append(train_loss)

        #visualize
        pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
        pred_box_ = pred_box[0].detach().cpu().numpy()

        pred_confidence_,pred_box_ = non_maximum_suppression(pred_confidence_,pred_box_,boxs_default)

        visualize_pred("train", pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default, name)
        
        
        #VALIDATION
        network.eval()
        
        # TODO: split the dataset into 90% training and 10% validation
        # use the training set to train and the validation set to evaluate
        TP = np.zeros([batch_size, 3])
        FP = np.zeros([batch_size, 3])
        FN_TP = np.zeros([batch_size, 3])
        for i, data in enumerate(dataloader_test, 0):
            images_, ann_box_, ann_confidence_, ori_height, ori_width, name, class_stat = data
            images = images_.cuda()
            ann_box = ann_box_.cuda()
            ann_confidence = ann_confidence_.cuda()

            pred_confidence, pred_box = network(images)
            
            pred_confidence_ = pred_confidence.detach().cpu().numpy()
            pred_box_ = pred_box.detach().cpu().numpy()

            #compute val loss
            loss_net = SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box)
            
            avg_loss += loss_net.data
            avg_count += 1
            
            #optional: implement a function to accumulate precision and recall to compute mAP or F1.
            TP_ = np.zeros([batch_size, 3])
            FP_ = np.zeros([batch_size, 3])
            FN_TP_ = np.zeros([batch_size, 3])
            pred_confidence_ = pred_confidence.detach().cpu().numpy()
            pred_box_ = pred_box.detach().cpu().numpy()
            for idx in range(batch_size):
                pred_confidence_[idx], pred_box_[idx] = non_maximum_suppression(pred_confidence_[idx], pred_box_[idx], boxs_default)
            class_stat = class_stat.detach().numpy()
            TP_, FP_, FN_TP_ = update_precision_recall(pred_confidence_, pred_box_, ann_confidence_.numpy(), ann_box_.numpy(), boxs_default,TP_,FP_,FN_TP_, class_stat, thres=0.5)
            TP += TP_
            FP += FP_
            FN_TP += FN_TP_
        # TP = np.sum(TP, axis=0)
        # FP = np.sum(FP, axis=0)
        # FN_TP = np.sum(FN_TP, axis=0)

        print('[%d] time: %f val loss: %f' % (epoch, time.time()-start_time, avg_loss/avg_count))
        val_loss = avg_loss/avg_count
        val_loss = val_loss.detach().cpu()
        val_losses.append(val_loss)
        #visualize
        pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
        pred_box_ = pred_box[0].detach().cpu().numpy()

        pred_confidence_,pred_box_ = non_maximum_suppression(pred_confidence_,pred_box_,boxs_default)
        
        visualize_pred("val", pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default, name)
        
        #optional: compute F1
        # precision = TP / (TP + FP)
        # recall = TP / FN_TP
        # print(precision, recall)
        # for c_id in range(3):
        #     F1score = 2*precision[c_id]*recall[c_id]/np.maximum(precision[c_id]+recall[c_id],1e-8)
        #     print(F1score)
        
        #save weights
        if epoch%10==9:
            #save last network
            print('saving net...')
            torch.save(network.state_dict(), 'network.pth')
    
    #plot train_loss and val_loss
    plt.plot(train_losses, color='b', label = 'train loss')
    plt.plot(val_losses, color='r', label = 'val loss')
    plt.legend()
    plt.xticks(np.arange(num_epochs))
    plt.show()

    TP = np.sum(TP, axis=0)
    FP = np.sum(FP, axis=0)
    FN_TP = np.sum(FN_TP, axis=0)
    
    FP = np.maximum(FP, 1e-8)
    FN_TP = np.maximum(FN_TP, 1e-8)
    print(FN_TP)
    print(TP)
    print(FP)

    precision = TP / (TP + FP)
    recall = TP / FN_TP
    print(precision, recall)
    for c_id in range(3):
        F1score = 2*precision[c_id]*recall[c_id]/np.maximum(precision[c_id]+recall[c_id],1e-8)
        print(F1score)


else:
    #TEST
    dataset_test = COCO("data/train/images/", "data/train/annotations/", class_num, boxs_default, train = False, image_size=320)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0)
    network.load_state_dict(torch.load('network.pth'))
    network.eval()
    
    for i, data in enumerate(dataloader_test, 0):
        images_, ann_box_, ann_confidence_, ori_height, ori_width, name, _ = data
        images = images_.cuda()
        ann_box = ann_box_.cuda()
        ann_confidence = ann_confidence_.cuda()

        pred_confidence, pred_box = network(images)

        pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
        pred_box_ = pred_box[0].detach().cpu().numpy()
        
        pred_confidence_,pred_box_ = non_maximum_suppression(pred_confidence_,pred_box_,boxs_default)

        #TODO: save predicted bounding boxes and classes to a txt file.
        #you will need to submit those files for grading this assignment
        path = "output/training/"
        output(path, name, boxs_default, pred_confidence_, pred_box_, ori_height, ori_width)
        
        visualize_pred("test", pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default, name)
        cv2.waitKey(10)



