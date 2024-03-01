import time

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
from tqdm import tqdm

from complexYOLO import ComplexYOLO
from kitti import KittiDataset
from zod import ZOD_Dataset
from region_loss import RegionLoss

import config as cnf

bc = cnf.boundary

batch_size=1 # TODO: Check if we can get 2 to work

# def collate_fn(batch):
#        imgs, targets = list(zip(*batch))
#        # Remove empty placeholder targets
#        # targets = [boxes for boxes in targets if boxes is not None]
#        # TODO: Do we need to add sample index to targets?
#        # Add sample index to targets
#        # for i, boxes in enumerate(targets):
#        #        boxes[:, 0] = i
#        targets = torch.cat(targets, 0)
#        imgs = torch.stack(imgs, 0)

#        print(f"targets: {targets.shape}")
#        print(f"imgs: {imgs.shape}")
#        return imgs, targets.unsqueeze(0)

# dataset
dataset=ZOD_Dataset(root='./zod',set='train')
data_loader = data.DataLoader(dataset, batch_size, shuffle=False, num_workers=1) # TODO: Why error when more than 1 worker?

model = ComplexYOLO()
# model = torch.load('ComplexYOLO_1000e.pt')
model.cuda()
# define optimizer
optimizer = optim.Adam(model.parameters())

# define loss function
region_loss = RegionLoss(num_classes=5, num_anchors=5)

for epoch in tqdm(range(1)):
       total_loss = 0
       total_metrics = {
            'nGT': 0,
            'recall': 0,
            'precision': 0,
            'nProposals': 0,
            'nCorrect': 0,
            'loss_x': 0,
            'loss_y': 0,
            'loss_w': 0,
            'loss_h': 0,
            'loss_conf': 0,
            'loss_cls': 0,
            'loss': 0
        }
       start_time_epoch = time.time()        
       for batch_idx, (rgb_map, target) in tqdm(enumerate(data_loader)):
              optimizer.zero_grad()
              # inference_time = time.time()
              output = model(rgb_map.float().cuda())
              # print(f"inference_time: {time.time() - inference_time}")

              loss, metrics = region_loss(output, target)
              if metrics['loss_cls'] == -1:
                     print(f"loss_cls -1 --> mask is only false at index: {batch_idx}")
              loss.backward()
              optimizer.step()
              total_loss += loss.item() 
              for k in total_metrics.keys():
                  total_metrics[k] += metrics[k]
       # TODO: Calculate metrics as in eval
       print("Epoch: %d, Time: %f, Loss: %f, Recall: %f, Precision %f, nGT %d, nProposals %d, nCorrect %d" %
             (epoch,
              time.time()-start_time_epoch,
              total_loss/len(data_loader),
              total_metrics["recall"]/len(data_loader),
              total_metrics["precision"]/len(data_loader),
              total_metrics["nGT"],
              total_metrics["nProposals"],
              total_metrics["nCorrect"]))
       if epoch % 10 == 0:
              torch.save(model, "ComplexYOLO_latest_zod.pt")
torch.save(model, f"ComplexYOLO_{epoch+1}e_zod.pt")

