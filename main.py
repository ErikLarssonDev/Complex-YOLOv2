import time
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
dataset=ZOD_Dataset(root='./minzod_mmdet3d',set='train')
data_loader = data.DataLoader(dataset, batch_size, shuffle=False)

model = ComplexYOLO()
model.cuda()
# define optimizer
optimizer = optim.Adam(model.parameters())

# define loss function
region_loss = RegionLoss(num_classes=5, num_anchors=5)

for epoch in tqdm(range(30000)):
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
       for batch_idx, (rgb_map, target) in enumerate(data_loader):
              optimizer.zero_grad()

              # rgb_map = rgb_map.view(rgb_map.data.size(0),rgb_map.data.size(3),rgb_map.data.size(1),rgb_map.data.size(2))
              inference_time = time.time()
              output = model(rgb_map.float().cuda())
              # print(f"inference_time: {time.time() - inference_time}")
              # print(f"output: {output.shape}")
              # print(f"target: {target.shape}")
              loss, metrics = region_loss(output, target)
              loss.backward()
              optimizer.step()
              total_loss += loss.item() 
              for k in total_metrics.keys():
                  total_metrics[k] += metrics[k]
       #        print("Epoch: %d, Batch: %d, Loss: %f, Time: %f" % (epoch, batch_idx, loss.item(), time.time()-start_time_batch))
       #        # print("Epoch: %d, Batch: %d, Time: %f" % (epoch, batch_idx, time.time()-start_time_batch))
       # print(f"metrics: {metrics["nGT"]/len(data_loader)}")
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
              torch.save(model, "ComplexYOLO_latest.pt")
torch.save(model, f"ComplexYOLO_epoch{epoch+1}.pt")

