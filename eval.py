from __future__ import division
import json
import os
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
from tqdm import tqdm
from region_loss import RegionLoss

from zod import ZOD_Dataset, inverse_yolo_targets
import config as cnf
from kitti_bev_utils import *
import utils
import eval_utils
import argparse
from codecarbon import EmissionsTracker

def parse_train_configs():
    parser = argparse.ArgumentParser(description='The Implementation of Complex YOLOv2')
    parser.add_argument("--show_results", action="store_true", help="Show results")
    parser.add_argument("--save_results", action="store_true", help="Save results")
    parser.add_argument("--conf_thresh", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--nms_thresh", type=float, default=0.5, help="Non-maximum suppression threshold")
    parser.add_argument("--experiment_name", type=str, default="default", help="Name of the experiment")
    parser.add_argument("--model_path", type=str, default="ComplexYOLO_20e_zod_anchor.pt", help="Path to the model")
    parser.add_argument("--data_path", type=str, default="./zod", help="Path to the data")
    parser.add_argument("--eval_subset", type=float, default=1.0, help="Percentage of the data to evaluate")
    parser.add_argument("--eval_num_samples", type=int, default=0, help="Number of samples to evaluate")
    parser.add_argument("--data_split", type=str, default="test", help="Data split (train, val, trainval, test)")


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

def evaluate_image(predictions, targets, iou_thresholds=[0.1, 0.3, 0.5, 0.7, 0.9]):
    tp, fp, fn = eval_utils.get_image_statistics_rotated_bbox(predictions, targets, iou_thresholds=iou_thresholds)
    return tp, fp, fn # These have shape [num_classes, IoU_thresholds]

def precision_recall_f1(tp, fp, fn):
    precision = tp / (tp + fp + 1e-16)
    recall = tp / (tp + fn + 1e-16)
    f1 = 2 * precision * recall / (precision + recall + 1e-16)
    return precision, recall, f1

def model_eval(model, data_loader, save_results=False, experiment_name="default"):
    # define loss function
    region_loss = RegionLoss(num_classes=5, num_anchors=5)
    iou_thresholds = [i/40 for i in range(1, 41)] # 40-point interpolation
    true_positives = np.zeros((len(cnf.class_list), len(iou_thresholds)))
    false_positives = np.zeros((len(cnf.class_list), len(iou_thresholds)))
    false_negatives = np.zeros((len(cnf.class_list), len(iou_thresholds)))
    total_inference_time = 0
    gt_to_save = []
    preds_to_save = []
    total_loss = 0

    model.cuda()
    model.eval()

    for batch_idx, (rgb_map, targets) in tqdm(enumerate(data_loader)):
        inference_time = time.time()
        output = model(rgb_map.float().cuda())  # torch.Size([1, 60, 16, 32])
        total_loss += region_loss(output, targets).item() # TODO: Filter out [:, :, :7] z, h from the targets
        total_inference_time += time.time() - inference_time

        # for all images in a batch
        for image_idx in range(output.size(0)):
            image_targets = targets[image_idx]
            ground_truth = copy.deepcopy(image_targets)
            image_boxes = get_region_boxes(output[image_idx], conf_thresh=0.5, num_classes=len(cnf.class_list), anchors=utils.anchors, num_anchors=len(utils.anchors)) # Convert the boxes from the output with anchors to acutal boxes
            pred_boxes = copy.deepcopy(image_boxes)

            # Converting the pred_boxes and ground_truth to metric space
            pred_boxes[:, 1] /= 32.0
            pred_boxes[:, 2] /= 16.0
            pred_boxes[:, 3] /= 32.0
            pred_boxes[:, 4] /= 16.0
            
            metric_gt = inverse_yolo_targets(np.array(ground_truth)[:, :7]) # TODO: Send in :9 and ground_truth=true, retutn BEV, 3D
            metric_pred = inverse_yolo_targets(np.array(pred_boxes)[:, :7]) # TODO: Return BEV, 3D
            gt_to_save.append(metric_gt) # TODO: Add corresponding lists for 3D
            preds_to_save.append(metric_pred)
    
            tp, fp, fn = evaluate_image(metric_pred, metric_gt, iou_thresholds=iou_thresholds)
            true_positives += tp
            false_positives += fp
            false_negatives += fn

    precision, recall, f1 = precision_recall_f1(true_positives, false_positives, false_negatives)

    results_dict = {
        "total_loss": total_loss / len(data_loader),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "true_positives": true_positives.tolist(),
        "false_positives": false_positives.tolist(),
        "false_negatives": false_negatives.tolist(),
        "inference_time": total_inference_time / len(data_loader) # TODO: Add support for energy consumption
    }

    if save_results:
        os.makedirs(f"results/{experiment_name}", exist_ok=True)
        with open(f"results/{experiment_name}/results_dict.txt", "w") as f:
            f.write(json.dumps(results_dict))
        np.save(f"results/{experiment_name}/ground_truths.npy", gt_to_save)
        np.save(f"results/{experiment_name}/predictions.npy", preds_to_save)
        print("Results saved!")

    return results_dict, gt_to_save, preds_to_save

bc = cnf.boundary

if __name__ == '__main__':
    args = parse_train_configs()
    batch_size=1 
    dataset=ZOD_Dataset(root=args.data_path,set=args.data_split)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=False)

    # eval result
    iou_thresholds=[0.01, 0.1, 0.3, 0.5, 0.7, 0.9]
    conf_thresh = args.conf_thresh
    nms_thresh = args.nms_thresh # TODO: Non-maximum suppression threshold is not done yet
    num_classes = int(5)
    num_anchors = int(5)
    true_positives = np.zeros((batch_size, len(cnf.class_list), len(iou_thresholds)))
    false_positives = np.zeros((batch_size, len(cnf.class_list), len(iou_thresholds)))
    false_negatives = np.zeros((batch_size, len(cnf.class_list), len(iou_thresholds)))
    total_inference_time = 0
    total_energy_consumption = 0
    gt_to_save = []
    preds_to_save = []
    if args.eval_num_samples > 0:
        samples_to_evaluate = args.eval_num_samples
    else:
        samples_to_evaluate = int(len(data_loader) * args.eval_subset)
    print(f"Samples to evaluate: {samples_to_evaluate}")
    model = torch.load(args.model_path)
    model.cuda()
    model.eval()
    tracker = EmissionsTracker()
    for batch_idx, (rgb_map, targets) in tqdm(enumerate(data_loader)):
        if batch_idx >= samples_to_evaluate:
            break

        tracker.start_task("Inference " + str(batch_idx))
        rgb_map = makeBVFeature(rgb_map, cnf.DISCRETIZATION_X, cnf.DISCRETIZATION_Y, cnf.boundary) # TODO: Move this line when we start with new models that has it in the forward pass
        inference_time = time.time()
        output = model(rgb_map.float().cuda())  # torch.Size([1, 60, 16, 32])
        total_inference_time += time.time() - inference_time
        tracker.stop_task()
        total_energy_consumption += 0  # TODO: Add energy consumption
        # for all images in a batch
        for image_idx in range(output.size(0)):
            image_targets = targets[image_idx]
            ground_truth = copy.deepcopy(image_targets)
            image_boxes = get_region_boxes(output[image_idx], conf_thresh, num_classes, utils.anchors, num_anchors) # Convert the boxes from the output with anchors to acutal boxes
            pred_boxes = copy.deepcopy(image_boxes)
            if args.show_results:
                # Printing ground truth
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
                                (255, 255, 255))

                img_bev = cv2.rotate(img_bev, cv2.ROTATE_180)
                cv2.imshow('single_sample', img_bev)
                print(f"\nShowing image {batch_idx}\n")
                cv2.imwrite(f"results/{batch_idx}.png", img_bev)

            # Converting the pred_boxes and ground_truth to metric space
            pred_boxes[:, 1] /= 32.0
            pred_boxes[:, 2] /= 16.0
            pred_boxes[:, 3] /= 32.0
            pred_boxes[:, 4] /= 16.0
            
            metric_gt = inverse_yolo_targets(np.array(ground_truth)[:, :7]) # TODO: Send in :9 and ground_truth=true, return BEV, 3D
            
            # TODO: Get z, h from the predicted class average
            metric_pred = inverse_yolo_targets(np.array(pred_boxes)[:, :7]) # TODO: Return BEV, 3D
            gt_to_save.append(metric_gt) # TODO: Add corresponding lists for 3D
            preds_to_save.append(metric_pred)
          
            # Printing in metric space
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
     

            tp, fp, fn = evaluate_image(metric_pred, metric_gt, iou_thresholds=iou_thresholds)
            true_positives[image_idx] += tp
            false_positives[image_idx] += fp
            false_negatives[image_idx] += fn

        if args.show_results:
            key = cv2.waitKey(0) & 0xFF  # Ensure the result is an 8-bit integer
            if key == 27:  # Check if 'Esc' key is pressed
                cv2.destroyAllWindows() 
                break
            elif key == 110:
                show_next_image = True  # Set the flag to False to avoid showing the same image again
                continue  # Skip the rest of the loop and go to the next iteration

    precision, recall, f1 = precision_recall_f1(np.sum(true_positives, axis=0), np.sum(false_positives, axis=0), np.sum(false_negatives, axis=0))
    print(f"tp+fn: {true_positives[0, :, 0] + false_negatives[0, :, 0]}")
    print(f"tp+fp: {true_positives[0, :, 0] + false_positives[0, :, 0]}")

    print(f"true_positives: {true_positives}")
    print(f"false_positives: {false_positives}")
    print(f"false_negatives: {false_negatives}")


    print(f"Metric: precision:\n {precision} \nrecall:\n {recall} \nf1:\n {f1}")
    print(f"Average inference time: {total_inference_time / len(data_loader)}")

    emissions = tracker.stop()
    for task_name, task in tracker._tasks.items():
        print(f"Task: {task_name}")
        print(f"\tEnergy : {1000 * task.emissions_data.energy_consumed} Wh")


    results_dict = {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "true_positives": true_positives.tolist(),
        "false_positives": false_positives.tolist(),
        "false_negatives": false_negatives.tolist(),
        "inference_time": total_inference_time / len(data_loader),
        "energy_consumption": total_energy_consumption / len(data_loader)
    }

    if args.save_results:
        os.makedirs(f"results/{args.experiment_name}", exist_ok=True)
        with open(f"results/{args.experiment_name}/results_dict.txt", "w") as f:
            f.write(json.dumps(results_dict))
        np.save(f"results/{args.experiment_name}/ground_truths.npy", gt_to_save)
        np.save(f"results/{args.experiment_name}/predictions.npy", preds_to_save)
        print("Results saved!")