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
boundary = { # TODO: Fix so that we can have rectangular boundaries and not only square
    "minX": -25,
    "maxX": 25,
    "minY": 0,
    "maxY": 250,
    "minZ": -2.73,
    "maxZ": 1.27
}
# Ratio = 5:1
# => 5/3 times too wide
# => 3/5 times too high 

# boundary = {
#     "minX": -25,
#     "maxX": 25,
#     "minY": 0,
#     "maxY": 50,
#     "minZ": -2.73,
#     "maxZ": 1.27
# }

# TODO: Explore how to increase resolution ratio = 3:1
BEV_WIDTH = 1024 # Original: 1024 X [0, 40] --> 6400 
BEV_HEIGHT = 512 # Original: 512 Y [-40, 40] --> 320
# Cell size: 3200 / (1024 x 512) = 0.006103515625 # Original: about 8cm
# Cell size: 12500 / (1500 x 1500) = 0.005555555555555556 # Original: about 8cm

# DISCRETIZATION = (boundary["maxX"] - boundary["minX"]) / BEV_HEIGHT

DISCRETIZATION_X = (boundary["maxX"] - boundary["minX"]) / BEV_HEIGHT 
DISCRETIZATION_Y = (boundary["maxY"] - boundary["minY"]) / BEV_WIDTH 

# TODO: Maybe add the aspect ratio factors in the config file

colors = [[0, 255, 255], [0, 0, 255], [255, 0, 0], [0, 255, 0], [255, 255, 255]]