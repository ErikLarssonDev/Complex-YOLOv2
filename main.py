import time

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np

from complexYOLO import ComplexYOLO
from kitti import KittiDataset
from region_loss import RegionLoss


batch_size=8

# dataset
dataset=KittiDataset(root='KITTI',set='train')
data_loader = data.DataLoader(dataset, batch_size, shuffle=True, pin_memory=True)

model = ComplexYOLO()
model.cuda()

# define optimizer
optimizer = optim.Adam(model.parameters())

# define loss function
region_loss = RegionLoss(num_classes=8, num_anchors=5)



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

              rgb_map = rgb_map.view(rgb_map.data.size(0),rgb_map.data.size(3),rgb_map.data.size(1),rgb_map.data.size(2))
              inference_time = time.time()
              print(f"rgb_map: {rgb_map.shape}") # (8, 3, 512, 1024)
              output = model(rgb_map.float().cuda())
              # print(f"inference_time: {time.time() - inference_time}")
              print(f"output: {output.shape}") # (8, 75, 16, 32)
              print(f"target: {target.shape}") # (8, 50, 7)
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
       # if epoch % 10 == 0:
       #        torch.save(model, "ComplexYOLO_latest")
