from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import cv2
from scipy import misc

from zod import ZOD_Dataset
import config as cnf
from kitti_bev_utils import *
import utils


def drawRect(img, pt1, pt2, pt3, pt4, color, lineWidth):
    cv2.line(img, pt1, pt2, color, lineWidth)
    cv2.line(img, pt2, pt3, color, lineWidth)
    cv2.line(img, pt3, pt4, color, lineWidth)
    cv2.line(img, pt1, pt4, color, lineWidth)


def get_region_boxes(x, conf_thresh, num_classes, anchors, num_anchors):
    if x.dim() == 3:
        x = x.unsqueeze(0)
    assert (x.size(1) == (10 + num_classes) * num_anchors) # Changed to 10 from 7 to match the output of the model

    nA = num_anchors  # num_anchors = 5
    nB = x.data.size(0)
    nC = num_classes  # num_classes = 5
    nH = x.data.size(2)  # nH  16
    nW = x.data.size(3)  # nW  32

    # Tensors for cuda support
    FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
    ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

    prediction = x.view(nB, nA, 10+nC, nH, nW).permute(0, 1, 3, 4, 2).contiguous()

    # Get outputs
    x = torch.sigmoid(prediction[..., 0])  # Center x
    y = torch.sigmoid(prediction[..., 1])  # Center y
    w = prediction[..., 2]  # Width
    l = prediction[..., 3]  # Height
    im = prediction[..., 4]  # Im
    re = prediction[..., 5]  # Re
    pred_conf = torch.sigmoid(prediction[..., 6])  # Conf
    pred_cls = torch.sigmoid(prediction[..., 7:])  # Cls pred.
    # TODO: Here we could set z and h based on the average of the training set if we want 3D bounding boxes

    # Calculate offsets for each grid
    grid_x = torch.arange(nW).repeat(nH, 1).view([1, 1, nH, nW]).type(FloatTensor)
    grid_y = torch.arange(nH).repeat(nW, 1).t().view([1, 1, nH, nW]).type(FloatTensor)
    scaled_anchors = FloatTensor([(a_w , a_h ) for a_w, a_h in anchors])
    anchor_w = scaled_anchors[:, 0:1].view((1, nA, 1, 1))
    anchor_h = scaled_anchors[:, 1:2].view((1, nA, 1, 1))

    # Add offset and scale with anchors
    pred_boxes = FloatTensor(prediction.shape)
    pred_boxes[..., 0] = x.data + grid_x
    pred_boxes[..., 1] = y.data + grid_y
    pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
    pred_boxes[..., 3] = torch.exp(l.data) * anchor_h
    pred_boxes[..., 4] = im.data
    pred_boxes[..., 5] = re.data
    pred_boxes[..., 6] = pred_conf
    pred_boxes[..., 7:(10 + nC) ] = pred_cls

    pred_boxes = utils.convert2cpu(pred_boxes.transpose(0, 1).contiguous().view(-1, (10 + nC)))  # torch.Size([2560, 15])

    all_boxes = []
    for i in range(2560):
        if pred_boxes[i][6] > conf_thresh:
            all_boxes.append(pred_boxes[i])
            # print(pred_boxes[i])
    return all_boxes


# classes
# class_list = ['Car', 'Van' , 'Truck' , 'Pedestrian' , 'Person_sitting' , 'Cyclist' , 'Tram' ]


bc = cnf.boundary

if __name__ == '__main__':
    dataset=ZOD_Dataset(root='./minzod_mmdet3d',set='train')
    data_loader = torch.utils.data.DataLoader(dataset, 1, shuffle=False)
    for batch_idx, (rgb_map, targets) in enumerate(data_loader):

        model = torch.load('ComplexYOLO_epoch29999.pt')
        model.cuda()
        model.eval()
        output = model(rgb_map.float().cuda())  # torch.Size([1, 75, 16, 32])

        # eval result
        conf_thresh = 0.7
        nms_thresh = 0.4
        num_classes = int(5)
        num_anchors = int(5)

        all_boxes = get_region_boxes(output, conf_thresh, num_classes, utils.anchors, num_anchors)

        targets = targets.squeeze(0)
        # TODO: Bounding boxes seems quite long and small
        targets[:, 1] *= cnf.BEV_WIDTH # x 0.1 
        targets[:, 2] *= cnf.BEV_HEIGHT # y 0.1 
        targets[:, 3] *= cnf.BEV_WIDTH * ((bc["maxY"]-bc["minY"]) / (bc["maxX"]-bc["minX"])) / (cnf.BEV_WIDTH / cnf.BEV_HEIGHT)  # 5/3
        targets[:, 4] *= cnf.BEV_HEIGHT *  (cnf.BEV_WIDTH / cnf.BEV_HEIGHT) / ((bc["maxY"]-bc["minY"]) / (bc["maxX"]-bc["minX"]))  # 3/5
        # targets[:, 3] *= cnf.BEV_HEIGHT # w
        # targets[:, 4] *= cnf.BEV_WIDTH # l

        # Get yaw angle
        targets[:, 5] = torch.atan2(targets[:, 5], targets[:, 6])
        img_bev = rgb_map.squeeze() * 255
        img_bev = img_bev.permute(1, 2, 0).numpy().astype(np.uint8)
        img_bev = cv2.resize(img_bev, (cnf.BEV_WIDTH, cnf.BEV_HEIGHT)) # TODO: Resize but maintain aspect ratio

        for c, x, y, w, l, yaw in targets[:, 0:6].numpy(): # targets = [cl, y1, x1, w1, l1, math.sin(float(yaw)), math.cos(float(yaw))]
            # Draw rotated box
            drawRotatedBox(img_bev, x, y, w, l, yaw, cnf.colors[int(c)])

        for i in range(len(all_boxes)):
            x = int(all_boxes[i][0] * cnf.BEV_WIDTH / 32.0)  # 32 cell = 1024 pixels
            y = int(all_boxes[i][1] * cnf.BEV_HEIGHT / 16.0)  # 16 cell = 512 pixels
            w = int(all_boxes[i][2] * cnf.BEV_WIDTH * ((bc["maxY"]-bc["minY"]) / (bc["maxX"]-bc["minX"])) / (cnf.BEV_WIDTH / cnf.BEV_HEIGHT)  / 32.0)  # 32 cell = 1024 pixels
            l = int(all_boxes[i][3] * cnf.BEV_HEIGHT * (cnf.BEV_WIDTH / cnf.BEV_HEIGHT) / ((bc["maxY"]-bc["minY"]) / (bc["maxX"]-bc["minX"])) / 16.0)  # 16 cell = 512 pixels
            im = all_boxes[i][4].detach().numpy()
            re = all_boxes[i][5].detach().numpy()
            yaw = np.arctan2(im, re)
            drawRotatedBox(img_bev, x, y, w, l, yaw, (0, 0, 255), 2, (165, 0, 255))
            # rect_top1 = int(x - l / 2)
            # rect_top2 = int(y - w / 2)
            # rect_bottom1 = int(x + l / 2)
            # rect_bottom2 = int(y + w / 2)
            # cv2.rectangle(img_bev, (rect_top1, rect_top2), (rect_bottom1, rect_bottom2), (0, 0, 255), 2)

        img_bev = cv2.rotate(img_bev, cv2.ROTATE_180)
        cv2.imshow('single_sample', img_bev)

        key = cv2.waitKey(0) & 0xFF  # Ensure the result is an 8-bit integer
        if key == 27:  # Check if 'Esc' key is pressed
            cv2.destroyAllWindows() 
            break
        elif key == 110:
            print(f"\nShowing image {batch_idx}\n")
            show_next_image = True  # Set the flag to False to avoid showing the same image again
            continue  # Skip the rest of the loop and go to the next iteration
        # break
    print('done')