from __future__ import division
import os
import os.path
import torch
import numpy as np
import cv2
import math
from utils import *

# TODO: Move from this file
bc={} 
bc['minX'] = 0; bc['maxX'] = 80; bc['minY'] = -40; bc['maxY'] = 40
bc['minZ'] =-2; bc['maxZ'] = 1.25

CLASS_NAME_TO_ID = {
    'Vehicle': 0,
    'VulnerableVehicle': 1,
    'Pedestrian': 2,
    'Animal': 3,
    'PoleObject': 4,
    'TrafficBeacon': 4,
    'TrafficSign': 4,
    'TrafficSignal': 4,
    'TrafficGuide': 4,
    'DynamicBarrier': 4,
    'Unclear': 4,
}

class ZOD_Dataset(torch.utils.data.Dataset):

    def __init__(self, root='/minzod_mmdet3d',set='train',type='velodyne_train'):
        self.type = type
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
        target = [[*label[:7], CLASS_NAME_TO_ID[str(label[7])], *label[8:]] for label in labels]
        target = np.array(target, dtype=np.float32)
        target = self.build_yolo_target_ZOD(target)
        # target = get_target(label_file,calib['Tr_velo2cam'])
        #print(target)
        #print(self.file_list[i])
        
        ################################
        # load point cloud data
        a = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)

        b = removePoints(a,bc)

        data = makeBVFeature(b, bc ,40/512)   # (512, 1024, 3)

        return data, target


    def __len__(self):
        return len(self.file_list)
    
    def build_yolo_target_ZOD(self, labels):
        target = []
        for i in range(labels.shape[0]):
            x, y, z, l, w, h, yaw, cl = labels[i]
            # ped and cyc labels are very small, so lets add some factor to height/width
            l = l + 0.3
            w = w + 0.3
            yaw = np.pi * 2 - yaw
            if (bc["minX"] < x < bc["maxX"]) and (bc["minY"] < y < bc["maxY"]):
                y1 = (y - bc["minY"]) / (bc["maxY"] - bc["minY"])  # we should put this in [0,1], so divide max_size  80 m
                x1 = (x - bc["minX"]) / (bc["maxX"] - bc["minX"])  # we should put this in [0,1], so divide max_size  40 m
                w1 = w / (bc["maxY"] - bc["minY"])
                l1 = l / (bc["maxX"] - bc["minX"])
                target.append([cl, y1, x1, w1, l1, math.sin(float(yaw)), math.cos(float(yaw))])

        return np.array(target, dtype=np.float32)


