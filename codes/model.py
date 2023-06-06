import os
import random
import numpy as np

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




def SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box):
    #input:
    #pred_confidence -- the predicted class labels from SSD, [batch_size, num_of_boxes, num_of_classes]
    #pred_box        -- the predicted bounding boxes from SSD, [batch_size, num_of_boxes, 4]
    #ann_confidence  -- the ground truth class labels, [batch_size, num_of_boxes, num_of_classes]
    #ann_box         -- the ground truth bounding boxes, [batch_size, num_of_boxes, 4]
    #
    #output:
    #loss -- a single number for the value of the loss function, [1]
    
    #TODO: write a loss function for SSD
    #
    #For confidence (class labels), use cross entropy (F.cross_entropy)
    #You can try F.binary_cross_entropy and see which loss is better
    #For box (bounding boxes), use smooth L1 (F.smooth_l1_loss)
    #
    #Note that you need to consider cells carrying objects and empty cells separately.
    #I suggest you to reshape confidence to [batch_size*num_of_boxes, num_of_classes]
    #and reshape box to [batch_size*num_of_boxes, 4].
    #Then you need to figure out how you can get the indices of all cells carrying objects,
    #and use confidence[indices], box[indices] to select those cells.
    pred_confidence = pred_confidence.reshape((-1, 4))
    pred_box = pred_box.reshape((-1, 4))
    ann_confidence = ann_confidence.reshape((-1, 4))
    ann_box =  ann_box.reshape((-1, 4))

    obj = torch.where(ann_confidence[:, 3] == 0)
    nonObj = torch.where(ann_confidence[:, 3] == 1)
    idx = obj[0]
    nonIdx = nonObj[0]
    target = torch.where(ann_confidence[idx] == 1)[1]
    nontarget = torch.where(ann_confidence[nonIdx] == 1)[1]

    confLoss = F.cross_entropy(pred_confidence[idx], target) + 3 * F.cross_entropy(pred_confidence[nonIdx], nontarget)
    boxLoss = F.smooth_l1_loss(pred_box[obj], ann_box[obj])
    
    return confLoss + boxLoss




class SSD(nn.Module):

    def __init__(self, class_num):
        super(SSD, self).__init__()
        
        self.class_num = class_num #num_of_classes, in this assignment, 4: cat, dog, person, background
        
        #TODO: define layers
        self.prelayer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1), #1
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), #2
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), #3
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), #4
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), #5
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), #6
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), #7
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), #8
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), #9
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1), #10
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), #11
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), #12
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1), #13
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        
        self.BranchOne_1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.BranchOne_2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.BranchOne_3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.BranchOne_left = nn.Conv2d(256, 16, kernel_size=1, stride=1, padding=0)
        self.BranchOne_right = nn.Conv2d(256, 16, kernel_size=1, stride=1, padding=0)

        self.BranchTwo_left_1 = nn.Conv2d(256, 16, kernel_size=3, stride=1, padding=1)
        self.BranchTwo_right_1 = nn.Conv2d(256, 16, kernel_size=3, stride=1, padding=1)
        
        self.BranchTwo_left_2 = nn.Conv2d(256, 16, kernel_size=3, stride=1, padding=1)
        self.BranchTwo_right_2 = nn.Conv2d(256, 16, kernel_size=3, stride=1, padding=1)

        self.BranchTwo_left_3 = nn.Conv2d(256, 16, kernel_size=3, stride=1, padding=1)
        self.BranchTwo_right_3 = nn.Conv2d(256, 16, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        #input:
        #x -- images, [batch_size, 3, 320, 320]
        
        x = x/255.0 #normalize image. If you already normalized your input image in the dataloader, remove this line.
        
        #TODO: define forward
        x1 = self.prelayer(x)
        x_one1 = self.BranchOne_1(x1)
        x_one2 = self.BranchOne_2(x_one1)
        x_one3 = self.BranchOne_3(x_one2)

        x_oneleft = self.BranchOne_left(x_one3)
        x_oneright = self.BranchOne_right(x_one3)
        x_oneleft = x_oneleft.reshape((-1, 16, 1))
        x_oneright = x_oneright.reshape((-1, 16, 1))

        x_twoleft1 = self.BranchTwo_left_1(x1)
        x_tworight1 = self.BranchTwo_right_1(x1)
        x_twoleft1 = x_twoleft1.reshape((-1, 16, 100))
        x_tworight1 = x_tworight1.reshape((-1, 16, 100))

        x_twoleft2 = self.BranchTwo_left_2(x_one1)
        x_tworight2 = self.BranchTwo_right_2(x_one1)
        x_twoleft2 = x_twoleft2.reshape((-1, 16, 25))
        x_tworight2 = x_tworight2.reshape((-1, 16, 25))

        x_twoleft3 = self.BranchTwo_left_3(x_one2)
        x_tworight3 = self.BranchTwo_right_3(x_one2)
        x_twoleft3 = x_twoleft3.reshape((-1, 16, 9))
        x_tworight3 = x_tworight3.reshape((-1, 16, 9))

        x_left = torch.cat((x_twoleft1, x_twoleft2, x_twoleft3, x_oneleft), dim = 2)
        x_left = x_left.permute((0, 2, 1))
        bboxes = x_left.reshape((-1, 540, 4))

        x_right = torch.cat((x_tworight1, x_tworight2, x_tworight3, x_oneright), dim = 2)
        x_right = x_right.permute((0, 2, 1))
        x_right = x_right.reshape((-1, 540, 4))
        confidence = torch.softmax(x_right, dim = 2)
        #should you apply softmax to confidence? (search the pytorch tutorial for F.cross_entropy.) If yes, which dimension should you apply softmax?
        
        #sanity check: print the size/shape of the confidence and bboxes, make sure they are as follows:
        #confidence - [batch_size,4*(10*10+5*5+3*3+1*1),num_of_classes]
        #bboxes - [batch_size,4*(10*10+5*5+3*3+1*1),4]
        
        return confidence, bboxes


# input = torch.rand(4, 3, 320, 320)
# ssd = SSD(4)
# conf, bbox = ssd(input)
# print("conf: ")
# print(conf.size())
# print("bbox: ")
# print(bbox.size())







