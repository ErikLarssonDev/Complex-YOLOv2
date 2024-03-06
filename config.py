import numpy as np

class_list = ['Vehicle', 'VulnerableVehicle', 'Pedestrian', 'Animal', 'StaticObject']

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

# Front side (of vehicle) Point Cloud boundary for BEV
boundary = {
    "minX": -25,
    "maxX": 25,
    "minY": 0,
    "maxY": 245,
    "minZ": -5,
    "maxZ": 3
}

BEV_WIDTH = 1024 # 3200 # 3200 # 4800 # Original: 1024 X [0, 40] 
BEV_HEIGHT = 512 # 640 # 640 # 960 # Original: 512 Y [-40, 40] 
# Cell size: 1024 / 80 = 12.8 --> 100 / 12.8 = 8 # Original: about 8cm
# Cell size: 3200 / 250 = 12.8 --> 100 / 12.8 = 8 # Original: about 8cm
# Cell size: 4800 / 250 = 19.2 --> 100 / 19.2 = 5.2 cm


CONFIG = {
       "name": "YOLO_5e_1024x512_trainval",
       "architecture": "Complex-YOLOv2-4800x960_trainval",
       "dataset": './zod',
       "epochs": 5,
       "batch_size": 1,
       "BEV_WIDTH": BEV_WIDTH,
       "BEV_HEIGHT": BEV_HEIGHT,
    }

DISCRETIZATION_X = (boundary["maxX"] - boundary["minX"]) / BEV_HEIGHT 
DISCRETIZATION_Y = (boundary["maxY"] - boundary["minY"]) / BEV_WIDTH 

# TODO: Maybe add the aspect ratio factors in the config file

colors = [(106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192), (60, 255, 60)]