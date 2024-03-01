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

from zod import ZOD_Dataset, inverse_yolo_targets
import config as cnf
from kitti_bev_utils import *
import utils
import eval_utils
import argparse

def parse_train_configs():
    parser = argparse.ArgumentParser(description='The Implementation of Complex YOLOv2')
    parser.add_argument("--show_results", action="store_true", help="Show results")
    parser.add_argument("--save_results", action="store_true", help="Save results")
    parser.add_argument("--conf_thresh", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--nms_thresh", type=float, default=0.5, help="Non-maximum suppression threshold")


    return parser.parse_args()


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
    pred_boxes = FloatTensor(prediction.shape)
    pred_boxes[..., 0] = x.data + grid_x
    pred_boxes[..., 1] = y.data + grid_y
    pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
    pred_boxes[..., 3] = torch.exp(l.data) * anchor_h
    pred_boxes[..., 4] = im.data
    pred_boxes[..., 5] = re.data
    pred_boxes[..., 6] = pred_conf
    pred_boxes[..., 7:] = pred_cls

    pred_boxes = utils.convert2cpu(pred_boxes.transpose(0, 1).contiguous().view(-1, (12))).detach().numpy()  # torch.Size([1, num_boxes, 8])
    
    all_boxes = np.array([]).reshape(0, 8)
    for i in range(pred_boxes.shape[0]):
        if pred_boxes[i][6] > conf_thresh:
            pred_boxes[i][0:8] = np.insert(pred_boxes[i], 0, np.argmax(pred_boxes[i][7:]), axis=0)[0:8]
            all_boxes = np.append(all_boxes, [pred_boxes[i][0:8]], axis=0)
    return all_boxes


def evaluate_mAP(predictions, targets, iou_thresholds=[0.1, 0.3, 0.5, 0.7, 0.9]):
    sample_metrics = eval_utils.get_image_statistics_rotated_bbox(predictions, targets, iou_thresholds=iou_thresholds)
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = eval_utils.ap_per_class(true_positives, pred_scores, pred_labels, targets[:, :, 0])
    return precision, recall, AP, f1, ap_class

def evaluate_image(predictions, targets, iou_thresholds=[0.1, 0.3, 0.5, 0.7, 0.9]):
    tp, fp, fn = eval_utils.get_image_statistics_rotated_bbox(predictions, targets, iou_thresholds=iou_thresholds)
    return tp, fp, fn # These have shape [num_classes, IoU_thresholds]

def precision_recall_f1(tp, fp, fn):
    precision = tp / (tp + fp + 1e-16)
    recall = tp / (tp + fn + 1e-16)
    f1 = 2 * precision * recall / (precision + recall + 1e-16)
    return precision, recall, f1
    
bc = cnf.boundary

if __name__ == '__main__':
    args = parse_train_configs()
    batch_size=1 
    dataset=ZOD_Dataset(root='./minzod_mmdet3d',set='train')
    data_loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=False)

    # define loss function
    region_loss = RegionLoss(num_classes=5, num_anchors=5)

    # eval result
    iou_thresholds=[0.01, 0.1, 0.3, 0.5, 0.7, 0.9]
    conf_thresh = args.conf_thresh
    nms_thresh = args.nms_thresh # TODO: Non-maximum suppression threshold is not done yet
    num_classes = int(5)
    num_anchors = int(5)
    true_positives_bev = np.zeros((batch_size, len(cnf.class_list), len(iou_thresholds)))
    false_positives_bev = np.zeros((batch_size, len(cnf.class_list), len(iou_thresholds)))
    false_negatives_bev = np.zeros((batch_size, len(cnf.class_list), len(iou_thresholds)))
    true_positives = np.zeros((batch_size, len(cnf.class_list), len(iou_thresholds)))
    false_positives = np.zeros((batch_size, len(cnf.class_list), len(iou_thresholds)))
    false_negatives = np.zeros((batch_size, len(cnf.class_list), len(iou_thresholds)))
    total_inference_time = 0
    gt_class_counts = np.zeros(len(cnf.class_list))
    pred_class_counts = np.zeros(len(cnf.class_list))
    for batch_idx, (rgb_map, targets) in enumerate(data_loader):
        model = torch.load('ComplexYOLO_3000e.pt')
        model.cuda()
        model.eval()
        inference_time = time.time()
        output = model(rgb_map.float().cuda())  # torch.Size([1, 60, 16, 32])
        total_inference_time += time.time() - inference_time

        # for all images in a batch
        for image_idx in range(output.size(0)):
            # Printing ground truth
            image_targets = targets[image_idx]
            ground_truth = copy.deepcopy(image_targets)
            image_targets[:, 1] *= cnf.BEV_WIDTH # x 
            image_targets[:, 2] *= cnf.BEV_HEIGHT # y
            image_targets[:, 3] *= cnf.BEV_WIDTH * ((bc["maxY"]-bc["minY"]) / (bc["maxX"]-bc["minX"])) / (cnf.BEV_WIDTH / cnf.BEV_HEIGHT)  # 5/3
            image_targets[:, 4] *= cnf.BEV_HEIGHT *  (cnf.BEV_WIDTH / cnf.BEV_HEIGHT) / ((bc["maxY"]-bc["minY"]) / (bc["maxX"]-bc["minX"]))  # 3/5
            image_targets[:, 5] = torch.atan2(image_targets[:, 5], image_targets[:, 6])

            img_bev = rgb_map[image_idx] * 255
            img_bev = img_bev.permute(1, 2, 0).numpy().astype(np.uint8)
            img_bev = cv2.resize(img_bev, (cnf.BEV_WIDTH, cnf.BEV_HEIGHT)) # TODO: Resize but maintain aspect ratio

            for c, x, y, w, l, yaw in image_targets[:, 0:6].numpy(): # targets = [cl, y1, x1, w1, l1, math.sin(float(yaw)), math.cos(float(yaw))]
                # Draw rotated box
                drawRotatedBox(img_bev, x, y, w, l, yaw, cnf.colors[int(c)])
       
            # Printing predictions
            image_boxes = get_region_boxes(output[image_idx], conf_thresh, num_classes, utils.anchors, num_anchors) # Convert the boxes from the output with anchors to acutal boxes
            pred_boxes = copy.deepcopy(image_boxes)
            for i in range(len(image_boxes)):
                image_boxes[i][1] *= cnf.BEV_WIDTH / 32.0 # 32 cell = 1024 pixels
                image_boxes[i][2] *= cnf.BEV_HEIGHT / 16.0  # 16 cell = 512 pixels
                image_boxes[i][3] *= cnf.BEV_WIDTH * ((bc["maxY"]-bc["minY"]) / (bc["maxX"]-bc["minX"])) / (cnf.BEV_WIDTH / cnf.BEV_HEIGHT)  / 32.0  # 32 cell = 1024 pixels
                image_boxes[i][4] *= cnf.BEV_HEIGHT * (cnf.BEV_WIDTH / cnf.BEV_HEIGHT) / ((bc["maxY"]-bc["minY"]) / (bc["maxX"]-bc["minX"])) / 16.0  # 16 cell = 512 pixels
                image_boxes[i][5] = np.arctan2(image_boxes[i][5], image_boxes[i][6])
                image_boxes[i][6] = image_boxes[i][7]
                drawRotatedBox(img_bev,
                               image_boxes[i][1],
                               image_boxes[i][2],
                               image_boxes[i][3],
                               image_boxes[i][4],
                               image_boxes[i][5],
                               cnf.colors[int(image_boxes[i][0])],
                               1,
                               (165, 0, 255))

            img_bev = cv2.rotate(img_bev, cv2.ROTATE_180)
            if args.show_results:
                cv2.imshow('single_sample', img_bev)
                print(f"\nShowing image {batch_idx}\n")
            
            if args.save_results:
                cv2.imwrite(f"results/{batch_idx}.png", img_bev)

            # Converting the pred_boxes and ground_truth to metric space
            pred_boxes[:, 1] /= 32.0
            pred_boxes[:, 2] /= 16.0
            pred_boxes[:, 3] /= 32.0
            pred_boxes[:, 4] /= 16.0
            
            metric_gt = inverse_yolo_targets(np.array(ground_truth)[:, :7])
            metric_pred = inverse_yolo_targets(np.array(pred_boxes)[:, :7])

            # TODO: boxes and targets should be saved to csv file
            pred_class_counts += np.bincount(pred_boxes[:, 0].astype(int))
            gt_class_counts += np.bincount(ground_truth[:, 0].numpy().astype(int))
          
            # # Printing in metric space
            # img_metric = np.zeros((250, 50, 3), dtype=np.uint8)
            # img_metric = cv2.resize(img_metric, (250, 50))
            # for c, x, y, w, l, yaw in metric_gt[:, 0:6]: # targets = [cl, y1, x1, w1, l1, math.sin(float(yaw)), math.cos(float(yaw))]
            #     # Draw rotated box
            #     drawRotatedBox(img_metric, x, y+25, w, l, yaw, cnf.colors[int(c)])
            
            # for c, x, y, w, l, yaw in metric_pred[:, 0:6]: # targets = [cl, y1, x1, w1, l1, math.sin(float(yaw)), math.cos(float(yaw))]
            #     # Draw rotated box
            #     drawRotatedBox(img_metric, x, y+25, w, l, yaw, cnf.colors[int(c)], 1 ,(165, 0, 255))
                
            # img_metric = cv2.resize(img_metric, (2000, 500))
            # img_metric = cv2.rotate(img_metric, cv2.ROTATE_180)
            # cv2.imshow('img_metric', img_metric)
     

            tp, fp, fn = evaluate_image(metric_pred, metric_gt, iou_thresholds=iou_thresholds) # TODO: Compare with BEV format
            true_positives[image_idx] += tp
            false_positives[image_idx] += fp
            false_negatives[image_idx] += fn
      
            tp_bev, fp_bev, fn_bev = evaluate_image(image_boxes[:, 0:6], image_targets[:, 0:6], iou_thresholds=iou_thresholds)
            true_positives_bev[image_idx] += tp_bev
            false_positives_bev[image_idx] += fp_bev
            false_negatives_bev[image_idx] += fn_bev

        if args.show_results:
            key = cv2.waitKey(0) & 0xFF  # Ensure the result is an 8-bit integer
            if key == 27:  # Check if 'Esc' key is pressed
                cv2.destroyAllWindows() 
                break
            elif key == 110:
                show_next_image = True  # Set the flag to False to avoid showing the same image again
                continue  # Skip the rest of the loop and go to the next iteration

    precision, recall, f1 = precision_recall_f1(np.sum(true_positives, axis=0), np.sum(false_positives, axis=0), np.sum(false_negatives, axis=0))
    precision_bev, recall_bev, f1_bev = precision_recall_f1(np.sum(true_positives_bev, axis=0), np.sum(false_positives_bev, axis=0), np.sum(false_negatives_bev, axis=0))
    print(f"gt_class_counts: {gt_class_counts}")
    print(f"tp+fn: {true_positives[0, :, 0] + false_negatives[0, :, 0]}")
    print(f"SUMS: gt_class_counts: {np.sum(gt_class_counts)} tp+fn: {np.sum(true_positives[0, :, 0] + false_negatives[0, :, 0])}")

    print(f"pred_class_counts: {pred_class_counts}")
    print(f"tp+fp: {true_positives[0, :, 0] + false_positives[0, :, 0]}")

    print(f"Upper limit recall: {pred_class_counts / gt_class_counts +1e-16}")

    print(f"bev tp+fp: {true_positives_bev[0, :, 0] + false_positives_bev[0, :, 0]}")
    print(f"bev tp+fn: {true_positives_bev[0, :, 0] + false_negatives_bev[0, :, 0]}")

    print(f"true_positives: {true_positives}")
    print(f"false_positives: {false_positives}")
    print(f"false_negatives: {false_negatives}")

    print(f"true_positives_bev: {true_positives_bev}")
    print(f"false_positives_bev: {false_positives_bev}")
    print(f"false_negatives_bev: {false_negatives_bev}")

    print(f"Metric: precision:\n {precision} \nrecall:\n {recall} \nf1:\n {f1}")
    print(f"BEV: precision:\n {precision_bev} \nrecall:\n {recall_bev} \nf1:\n {f1_bev}")
    print(f"Average inference time: {total_inference_time / len(data_loader)}")