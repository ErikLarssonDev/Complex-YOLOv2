from __future__ import division
import math
import time
import sys
import tqdm

import torch
import numpy as np
from shapely.geometry import Polygon
import rotate_iou
import config as cnf

# sys.path.append('../')

import kitti_bev_utils as bev_utils
# from .rotate_iou import rotate_iou_gpu_eval


def cvt_box_2_polygon(box):
    """
    :param box: an array of shape [4, 2]
    :return: a shapely.geometry.Polygon object
    """
    # use .buffer(0) to fix a line polygon
    # more infor: https://stackoverflow.com/questions/13062334/polygon-intersection-error-in-shapely-shapely-geos-topologicalerror-the-opera
    return Polygon([(box[i, 0], box[i, 1]) for i in range(len(box))]).buffer(0)


def compute_iou_nms(idx_self, idx_other, polygons, areas):
    """Calculates IoU of the given box with the array of the given boxes.
    box: a polygon
    boxes: a vector of polygons
    Note: the areas are passed in rather than calculated here for
    efficiency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    ious = []
    box1 = polygons[idx_self]
    for idx in idx_other:
        box2 = polygons[idx]
        intersection = box1.intersection(box2).area
        iou = intersection / (areas[idx] + areas[idx_self] - intersection + 1e-12)
        ious.append(iou)

    return np.array(ious, dtype=np.float32)


def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names


def rescale_boxes(boxes, current_dim, original_shape):
    """ Rescales bounding boxes to the original shape """
    orig_h, orig_w = original_shape
    # The amount of padding that was added
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
    # Image height and width after padding is removed
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x
    # Rescale bounding boxes to dimension of original image
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h

    return boxes


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in tqdm.tqdm(unique_classes, desc="Computing AP"):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def get_image_statistics_rotated_bbox(output, targets, iou_thresholds, num_classes=5):
    """ Compute true positives, predicted scores and predicted labels per sample """
    if len(output) == 0:
        gt_per_class = np.zeros((num_classes, len(iou_thresholds)))
        for i in range(len(cnf.class_list)):
            gt_per_class[i] = np.sum(targets[:, 0] == i)
        return np.zeros((num_classes, len(iou_thresholds))), np.zeros((num_classes, len(iou_thresholds))), gt_per_class
    pred_labels = np.array(output[:,0]).astype(np.int32)
    pred_boxes = output[:, [1, 2, 4, 5, 7]] # x, y, z, w, l, h, yaw

    true_positives = np.zeros((num_classes, len(iou_thresholds)))
    false_positives = np.zeros((num_classes, len(iou_thresholds)))
    false_negatives = np.zeros((num_classes, len(iou_thresholds)))
    if len(targets) > 0:
        target_labels = np.array(targets[:, 0]).astype(np.int32)
        target_boxes = targets[:, [1, 2, 4, 5, 7]]
        overlap_matrix = np.zeros((len(pred_boxes), len(target_boxes)))

        overlap_matrix = rotate_iou.rotate_iou_gpu_eval(np.array(pred_boxes), np.array(target_boxes))
        for iou_threshold_i, iou_threshold in enumerate(iou_thresholds):
            prediction_indexes_to_match = list(range(len(pred_boxes)))
            matched_boxes = np.full(len(target_boxes), -1) # pred indices for each gt box, -1 means not matched
            while len(prediction_indexes_to_match) > 0:
                pred_i = prediction_indexes_to_match.pop(0)
                
                # Create a list with all the iou for the pred_i box and sort it
                iou_sorted_gt_indexes = np.argsort(overlap_matrix[pred_i])[::-1]

                for gt_box_idx in iou_sorted_gt_indexes:
                    candidate_match_iou = overlap_matrix[pred_i, gt_box_idx]
                    if candidate_match_iou < iou_threshold:
                        break
                    prev_match = matched_boxes[gt_box_idx]
                    if prev_match == -1:
                        matched_boxes[gt_box_idx] = pred_i
                        break
                    elif overlap_matrix[prev_match, gt_box_idx] < candidate_match_iou:
                        matched_boxes[gt_box_idx] = pred_i
                        prediction_indexes_to_match.append(prev_match)
                        break
            
            for gt_index, pred_idx in enumerate(matched_boxes):
                if pred_idx == -1:
                    false_negatives[target_labels[gt_index], iou_threshold_i] += 1
                elif pred_labels[pred_idx] == target_labels[gt_index]:
                    true_positives[target_labels[gt_index], iou_threshold_i] += 1
                else:
                    false_negatives[target_labels[gt_index], iou_threshold_i] += 1
                    false_positives[pred_labels[pred_idx], iou_threshold_i] += 1
            
            for pred_idx in range(len(pred_boxes)):
                if pred_idx not in matched_boxes:
                    false_positives[pred_labels[pred_idx], iou_threshold_i] += 1

    return true_positives, false_positives, false_negatives

def iou_rotated_single_vs_multi_boxes_cpu(single_box, multi_boxes):
    """
    :param pred_box: Numpy array
    :param target_boxes: Numpy array
    :return:
    """

    s_x, s_y, s_w, s_l, s_yaw = single_box
    s_area = s_w * s_l
    s_conners = bev_utils.get_corners(s_x, s_y, s_w, s_l, s_yaw)
    s_polygon = cvt_box_2_polygon(s_conners)

    m_x, m_y, m_w, m_l, m_yaw = multi_boxes.transpose(1, 0)
    targets_areas = m_w * m_l
    m_boxes_conners = get_corners_vectorize(m_x, m_y, m_w, m_l, m_yaw)
    m_boxes_polygons = [cvt_box_2_polygon(box_) for box_ in m_boxes_conners]

    ious = []
    for m_idx in range(multi_boxes.shape[0]):
        intersection = s_polygon.intersection(m_boxes_polygons[m_idx]).area
        iou_ = intersection / (s_area + targets_areas[m_idx] - intersection + 1e-16)
        ious.append(iou_)

    return torch.tensor(ious, dtype=torch.float)


def get_corners_vectorize(x, y, w, l, yaw):
    """bev image coordinates format - vectorization

    :param x, y, w, l, yaw: [num_boxes,]
    :return: num_boxes x (x,y) of 4 conners
    """
    bbox2 = np.zeros((x.shape[0], 4, 2), dtype=np.float32)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    # front left
    bbox2[:, 0, 0] = x - w / 2 * cos_yaw - l / 2 * sin_yaw
    bbox2[:, 0, 1] = y - w / 2 * sin_yaw + l / 2 * cos_yaw

    # rear left
    bbox2[:, 1, 0] = x - w / 2 * cos_yaw + l / 2 * sin_yaw
    bbox2[:, 1, 1] = y - w / 2 * sin_yaw - l / 2 * cos_yaw

    # rear right
    bbox2[:, 2, 0] = x + w / 2 * cos_yaw + l / 2 * sin_yaw
    bbox2[:, 2, 1] = y + w / 2 * sin_yaw - l / 2 * cos_yaw

    # front right
    bbox2[:, 3, 0] = x + w / 2 * cos_yaw - l / 2 * sin_yaw
    bbox2[:, 3, 1] = y + w / 2 * sin_yaw + l / 2 * cos_yaw

    return bbox2


def nms_cpu(boxes, confs, nms_thresh=0.5):
    """
    :param boxes: [num, 6]
    :param confs: [num, num_classes]
    :param nms_thresh:
    :param min_mode:
    :return:
    """
    # order of reduce confidence (high --> low)
    order = confs.argsort()[::-1]

    x, y, w, l, im, re = boxes.transpose(1, 0)
    yaw = np.arctan2(im, re)
    boxes_conners = get_corners_vectorize(x, y, w, l, yaw)
    boxes_polygons = [cvt_box_2_polygon(box_) for box_ in boxes_conners]  # 4 vertices of the box
    boxes_areas = w * l

    keep = []
    while order.size > 0:
        idx_self = order[0]
        idx_other = order[1:]
        keep.append(idx_self)
        over = compute_iou_nms(idx_self, idx_other, boxes_polygons, boxes_areas)
        inds = np.where(over <= nms_thresh)[0]
        order = order[inds + 1]

    return np.array(keep)


def post_processing(outputs, conf_thresh=0.95, nms_thresh=0.4):
    """
        Removes detections with lower object confidence score than 'conf_thres' and performs
        Non-Maximum Suppression to further filter detections.
        Returns detections with shape:
            (x, y, w, l, im, re, object_conf, class_score, class_pred)
    """
    if type(outputs).__name__ != 'ndarray':
        outputs = outputs.numpy()
    # outputs shape: (batch_size, 22743, 10)
    batch_size = outputs.shape[0]
    # box_array: [batch, num, 6]
    box_array = outputs[:, :, :6]

    # confs: [batch, num, num_classes]
    confs = outputs[:, :, 6:7] * outputs[:, :, 7:]
    obj_confs = outputs[:, :, 6]

    # [batch, num, num_classes] --> [batch, num]
    max_conf = np.max(confs, axis=2)
    max_id = np.argmax(confs, axis=2)

    bboxes_batch = [None for _ in range(batch_size)]

    for i in range(batch_size):
        argwhere = max_conf[i] > conf_thresh
        l_box_array = box_array[i, argwhere, :]
        l_obj_confs = obj_confs[i, argwhere, :]
        l_max_conf = max_conf[i, argwhere]
        l_max_id = max_id[i, argwhere]

        keep = nms_cpu(l_box_array, l_max_conf, nms_thresh=nms_thresh)

        if (keep.size > 0):
            l_box_array = l_box_array[keep, :]
            l_obj_confs = l_obj_confs[keep].reshape(-1, 1)
            l_max_conf = l_max_conf[keep].reshape(-1, 1)
            l_max_id = l_max_id[keep].reshape(-1, 1)
            bboxes_batch[i] = np.concatenate((l_box_array, l_obj_confs, l_max_conf, l_max_id), axis=-1)
    return bboxes_batch

def bev_box_overlap(boxes, qboxes, criterion=-1):
    riou = rotate_iou_gpu_eval(boxes, qboxes, criterion)
    return riou

def post_processing_v2(prediction, conf_thresh=0.95, nms_thresh=0.4):
    """
        Removes detections with lower object confidence score than 'conf_thres' and performs
        Non-Maximum Suppression to further filter detections.
        Returns detections with shape:
            (x, y, w, l, im, re, object_conf, class_score, class_pred)
    """
    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 6] >= conf_thresh]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Object confidence times class confidence
        score = image_pred[:, 6] * image_pred[:, 7:].max(dim=1)[0]
        # Sort by it
        image_pred = image_pred[(-score).argsort()]
        class_confs, class_preds = image_pred[:, 7:].max(dim=1, keepdim=True)
        detections = torch.cat((image_pred[:, :7].float(), class_confs.float(), class_preds.float()), dim=1)
        keep_boxes = []

        # Perform non-maximum suppression
        #keep_boxes = []
        # print(f"detections: {detections[:2, :5].numpy()}")
        # # Get yaw angle
        # detections[:, 4] = torch.atan2(detections[:, 4], detections[:, 5])
        # detections[:, 4] = 0
        # detections_2 = detections.clone()
        # overlap_part = bev_box_overlap(detections[:, :5].cpu().numpy(), detections_2[:, :5].cpu().numpy(), criterion=-1).astype(np.float64)
        # print(overlap_part.diagonal().mean())
        # # overlap_part = rotate_iou_gpu_eval(np.array([[6.0000977e+02, 8.1578074e+00,  2.0428976e+01,  5.0748917e+01, 0]]),
        # #                                np.array([[6.0000977e+02, 8.1578074e+00,  2.0428976e+01,  5.0748917e+01, 0]])).astype(np.float64)
        # print(f"overlap_part: {overlap_part}")
        # while detections.size(0):
        for detection in detections:
        #     # TODO: This step can take quite some time, check if we can avoid that
        #     large_overlap = overlap_part[0]
        #     # large_overlap = iou_rotated_single_vs_multi_boxes_cpu(detections[0, :6], detections[:, :6]) > nms_thresh #TODO: replace with gpu version from mmdet3d
        #     print(f"large_overlap: {large_overlap}")
        #     label_matches = detections[0, -1] == detections[:, -1]
        #     invalid = large_overlap & label_matches

        #     print(f"nms: {detections.size()}")
        #     # large_overlap = rotated_bbox_iou(detections[0, :6].unsqueeze(0), detections[:, :6], 1.0, False) > nms_thres # not working
        #     # Indices of boxes with lower confidence scores, large IOUs and matching labels
        #     weights = detections[invalid, 6:7]
        #     # Merge overlapping bboxes by order of confidence
        #     detections[0, :6] = (weights * detections[invalid, :6]).sum(0) / weights.sum()
              keep_boxes += [detection]
            
        #     detections = detections[~invalid]
        if len(keep_boxes) > 0:
            output[image_i] = torch.stack(keep_boxes)

    return output
