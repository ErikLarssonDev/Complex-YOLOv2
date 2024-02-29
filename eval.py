from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable
import copy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import cv2
from scipy import misc
from region_loss import RegionLoss

from zod import ZOD_Dataset
import config as cnf
from kitti_bev_utils import *
import utils
import eval_utils


def drawRect(img, pt1, pt2, pt3, pt4, color, lineWidth):
    cv2.line(img, pt1, pt2, color, lineWidth)
    cv2.line(img, pt2, pt3, color, lineWidth)
    cv2.line(img, pt3, pt4, color, lineWidth)
    cv2.line(img, pt1, pt4, color, lineWidth)


def get_region_boxes(x, conf_thresh, num_classes, anchors, num_anchors):
    if x.dim() == 3:
        x = x.unsqueeze(0)
    assert (x.size(1) == (7 + num_classes) * num_anchors)
    nA = num_anchors  # num_anchors = 5
    nB = x.data.size(0)
    nC = num_classes  # num_classes = 5
    nH = x.data.size(2)  # nH  16
    nW = x.data.size(3)  # nW  32

    # Tensors for cuda support
    FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
    ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

    prediction = x.view(nB, nA, 7+nC, nH, nW).permute(0, 1, 3, 4, 2).contiguous()
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
    pred_boxes = FloatTensor(prediction.shape[0],prediction.shape[1],prediction.shape[2],prediction.shape[3],8)
    pred_boxes[..., 0] = x.data + grid_x
    pred_boxes[..., 1] = y.data + grid_y
    pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
    pred_boxes[..., 3] = torch.exp(l.data) * anchor_h
    pred_boxes[..., 4] = im.data
    pred_boxes[..., 5] = re.data
    pred_boxes[..., 6] = pred_conf
    pred_boxes[..., 7] = torch.argmax(pred_cls)
    print(f"pred_boxes.shape: {pred_boxes.shape}")

    pred_boxes = utils.convert2cpu(pred_boxes.transpose(0, 1).contiguous().view(-1, (8))).detach().numpy()  # torch.Size([1, num_boxes, 8])
    
    all_boxes = []
    for i in range(pred_boxes.shape[0]):
        if pred_boxes[i][6] > conf_thresh:
            all_boxes.append(pred_boxes[i])
    return all_boxes


def evaluate_mAP(predictions, targets, iou_thresholds=[0.1, 0.3, 0.5, 0.7, 0.9]):
    sample_metrics = eval_utils.get_image_statistics_rotated_bbox(predictions, targets, iou_thresholds=iou_thresholds)
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = eval_utils.ap_per_class(true_positives, pred_scores, pred_labels, targets[:, :, 0])
    return precision, recall, AP, f1, ap_class

def evaluate_image(predictions, targets, iou_thresholds=[0.1, 0.3, 0.5, 0.7, 0.9]):
    sample_metrics = eval_utils.get_image_statistics_rotated_bbox(predictions, targets, iou_thresholds=iou_thresholds)
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]

    return tp, tn, fp, fn # These have shape [num_classes, IoU_thresholds]
    
bc = cnf.boundary

if __name__ == '__main__':
    dataset=ZOD_Dataset(root='./minzod_mmdet3d',set='train')
    data_loader = torch.utils.data.DataLoader(dataset, 1, shuffle=False)

    # define loss function
    region_loss = RegionLoss(num_classes=5, num_anchors=5)

    for batch_idx, (rgb_map, targets) in enumerate(data_loader):
        model = torch.load('ComplexYOLO_latest_euler_nC5.pt')
        model.cuda()
        model.eval()
        output = model(rgb_map.float().cuda())  # torch.Size([1, 60, 16, 32])

        # eval result
        conf_thresh = 0.5
        nms_thresh = 0.5
        num_classes = int(5)
        num_anchors = int(5)
        # for all images in a batch
        for image_idx in range(output.size(0)):
            image_boxes = get_region_boxes(output[image_idx], conf_thresh, num_classes, utils.anchors, num_anchors)
            image_targets = targets[image_idx]
            # get pred boxes
            # get gt boxes
            # count total tp tn fp fn per class per IoU threshold, each metric is [num_classes, IoU_thresholds]
        # Sum up all metrics and return
            

            image_targets[:, 1] *= cnf.BEV_WIDTH # x 0.1 
            image_targets[:, 2] *= cnf.BEV_HEIGHT # y 0.1 
            image_targets[:, 3] *= cnf.BEV_WIDTH * ((bc["maxY"]-bc["minY"]) / (bc["maxX"]-bc["minX"])) / (cnf.BEV_WIDTH / cnf.BEV_HEIGHT)  # 5/3
            image_targets[:, 4] *= cnf.BEV_HEIGHT *  (cnf.BEV_WIDTH / cnf.BEV_HEIGHT) / ((bc["maxY"]-bc["minY"]) / (bc["maxX"]-bc["minX"]))  # 3/5
            image_targets[:, 5] = torch.atan2(image_targets[:, 5], image_targets[:, 6]) # Get yaw angle

            img_bev = rgb_map[image_idx] * 255
            img_bev = img_bev.permute(1, 2, 0).numpy().astype(np.uint8)
            img_bev = cv2.resize(img_bev, (cnf.BEV_WIDTH, cnf.BEV_HEIGHT)) # TODO: Resize but maintain aspect ratio

            for c, x, y, w, l, yaw in image_targets[:, 0:6].numpy(): # targets = [cl, y1, x1, w1, l1, math.sin(float(yaw)), math.cos(float(yaw))]
                # Draw rotated box
                drawRotatedBox(img_bev, x, y, w, l, yaw, cnf.colors[int(c)])
       
            for i in range(len(image_boxes)):
                image_boxes[i][0] *= cnf.BEV_WIDTH / 32.0 # 32 cell = 1024 pixels
                image_boxes[i][1] *= cnf.BEV_HEIGHT / 16.0  # 16 cell = 512 pixels
                image_boxes[i][2] *= cnf.BEV_WIDTH * ((bc["maxY"]-bc["minY"]) / (bc["maxX"]-bc["minX"])) / (cnf.BEV_WIDTH / cnf.BEV_HEIGHT)  / 32.0  # 32 cell = 1024 pixels
                image_boxes[i][3] *= cnf.BEV_HEIGHT * (cnf.BEV_WIDTH / cnf.BEV_HEIGHT) / ((bc["maxY"]-bc["minY"]) / (bc["maxX"]-bc["minX"])) / 16.0  # 16 cell = 512 pixels
                image_boxes[i][4] = np.arctan2(image_boxes[i][4], image_boxes[i][5])
                image_boxes[i][5] = image_boxes[i][6]
                image_boxes[i][6] = image_boxes[i][7] 
                drawRotatedBox(img_bev,
                               image_boxes[i][0],
                               image_boxes[i][1],
                               image_boxes[i][2],
                               image_boxes[i][3],
                               image_boxes[i][4],
                               (0, 0, 255),
                               2,
                               (165, 0, 255))

            img_bev = cv2.rotate(img_bev, cv2.ROTATE_180)
            cv2.imshow('single_sample', img_bev)
            print(f"\nShowing image {batch_idx}\n")
            # TODO: image_boxes and image_targets are in BEV format, should be metric space
            precision, recall, AP, f1, ap_class = evaluate_image(np.array(image_boxes)[:, :7], image_targets[:, :6])
            print(f"precision: {precision}")
            print(f"recall: {recall}")
            print(f"AP: {AP}")
            print(f"f1: {f1}")
            print(f"ap_class: {ap_class}")

        key = cv2.waitKey(0) & 0xFF  # Ensure the result is an 8-bit integer
        if key == 27:  # Check if 'Esc' key is pressed
            cv2.destroyAllWindows() 
            break
        elif key == 110:
            show_next_image = True  # Set the flag to False to avoid showing the same image again
            continue  # Skip the rest of the loop and go to the next iteration
        # break
    print('done')