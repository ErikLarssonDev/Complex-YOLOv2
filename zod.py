from __future__ import division
import os
import os.path
import time
import torch
import numpy as np
import cv2
import math
from kitti_bev_utils import *
import config as cnf

bc = cnf.boundary

class ZOD_Dataset(torch.utils.data.Dataset):

    def __init__(self, root='/minzod_mmdet3d',set='train'):
        self.root = root
        self.data_path = os.path.join(root)
        self.lidar_path = os.path.join(self.data_path, "points/")
        self.label_path = os.path.join(self.data_path, "labels/")

        with open(os.path.join(self.data_path, 'ImageSets', '%s.txt' % set)) as f:
            self.file_list = f.read().splitlines()


    def __getitem__(self, i):
        lidar_file = self.lidar_path + '/' + self.file_list[i] + '.bin'
        label_file = self.label_path + '/' + self.file_list[i] + '.txt'
        lines = [line.rstrip() for line in open(label_file)]
        labels = [label_file_line.split(' ') for label_file_line in lines]
        target = [[*label[:7], cnf.CLASS_NAME_TO_ID[str(label[7])], *label[8:]] for label in labels]
        target = np.array(target, dtype=np.float32) 
        
        # If annotations is empty, then pick a new image
        target = self.build_yolo_target_ZOD(target)
        if len(target) == 0:
            # print(f"\nEmpty annotations for {self.file_list[i]}")
            new_index = np.random.randint(0, len(self.file_list))
            # print(f"Trying a new image with index {new_index} instead")
            return self.__getitem__(new_index)

        # load point cloud data
        a = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)

        b = removePoints(a, bc)
    
        return torch.tensor(b), torch.tensor(target)


    def __len__(self):
        return len(self.file_list)
    
    def build_yolo_target_ZOD(self, labels):
        target = []
        for i in range(labels.shape[0]):
            x, y, z, l, w, h, yaw, cl = labels[i]
            # ped and cyc labels are very small, so lets add some factor to height/width
            l = l
            w = w
            yaw = np.pi * 2 - yaw
            if (bc["minX"] < x < bc["maxX"]) and (bc["minY"] < y < bc["maxY"]):
                y1 = (y - bc["minY"]) / (bc["maxY"] - bc["minY"])  # we should put this in [0,1], so divide max_size  80 m
                x1 = (x - bc["minX"]) / (bc["maxX"] - bc["minX"])  # we should put this in [0,1], so divide max_size  40 m
                w1 = w / (bc["maxY"] - bc["minY"])
                l1 = l / (bc["maxX"] - bc["minX"])
                target.append([cl, y1, x1, w1, l1, math.sin(float(yaw)), math.cos(float(yaw)), z, h])
        return np.array(target, dtype=np.float32)
    
def inverse_yolo_targets(targets, ground_truth=False):
    z_h_priors = [(-1.12, 1.74), (-1.34, 1.34), (-1.07, 1.68), (-1.58, 0.64), (0.85, 2.34)] # The priors for z and h which are used at inference time, average of the training set
    labels = []
    for t in targets:
        if ground_truth:
            c, y, x, w, l, im, re, z, h = t
        else:
            c, y, x, w, l, im, re = t
            z, h = z_h_priors[int(c)]

        x = x * (bc["maxX"] - bc["minX"]) + bc["minX"]
        y = y * (bc["maxY"] - bc["minY"]) + bc["minY"]
        w = w * (bc["maxY"] - bc["minY"])
        l = l * (bc["maxX"] - bc["minX"])
        yaw = math.atan2(im, re)
        labels.append([c, y, x, z, w, l, h, yaw])
   
    return np.array(labels)

if __name__ == '__main__':
    dataset=ZOD_Dataset(root='./minzod_mmdet3d',set='train')
    data_loader = torch.utils.data.DataLoader(dataset, 1, shuffle=False)
    for batch_idx, (rgb_map, targets) in enumerate(data_loader):
        
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

        rgb_map = makeBVFeature(rgb_map.squeeze(), cnf.DISCRETIZATION_X, cnf.DISCRETIZATION_Y, cnf.boundary)
        img_bev = rgb_map * 255
        img_bev = img_bev.permute(1, 2, 0).numpy().astype(np.uint8)
        img_bev = cv2.resize(img_bev, (cnf.BEV_WIDTH, cnf.BEV_HEIGHT)) # TODO: Resize but maintain aspect ratio

        for c, x, y, w, l, yaw in targets[:, 0:6].numpy(): # targets = [cl, y1, x1, w1, l1, math.sin(float(yaw)), math.cos(float(yaw))]
            # Draw rotated box
            drawRotatedBox(img_bev, x, y, w, l, yaw, cnf.colors[int(c)])

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
