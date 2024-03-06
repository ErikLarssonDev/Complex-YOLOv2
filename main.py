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
from kitti_bev_utils import makeBVFeature
from zod import ZOD_Dataset
from region_loss import RegionLoss
from eval import model_eval
import wandb

import config as cnf

bc = cnf.boundary

# dataset
train_dataset=ZOD_Dataset(root=cnf.CONFIG["dataset"],set='trainval')
val_dataset=ZOD_Dataset(root=cnf.CONFIG["dataset"],set='val')
test_dataset = ZOD_Dataset(root=cnf.CONFIG["dataset"],set='test')
data_loader = data.DataLoader(train_dataset, cnf.CONFIG["batch_size"], shuffle=True, num_workers=1) # The shm memory is not enough for num_workers >=2
val_loader = data.DataLoader(val_dataset, cnf.CONFIG["batch_size"], shuffle=False, num_workers=1)
test_loader = data.DataLoader(test_dataset, cnf.CONFIG["batch_size"], shuffle=False)

model = ComplexYOLO()
# model = torch.load('ComplexYOLO_1000e.pt')
model.cuda()
# define optimizer
optimizer = optim.Adam(model.parameters())

# define loss function
region_loss = RegionLoss(num_classes=5, num_anchors=5)


wandb.init(
    # set the wandb project where this run will be logged
    project="exjobb",
    name=cnf.CONFIG["name"],
    
    # track hyperparameters and run metadata
    config=cnf.CONFIG
)

for epoch in tqdm(range(cnf.CONFIG["epochs"])):
       total_loss = 0
       total_metrics = {
        "total_loss": 0,
        "precision": [],
        "recall": [],
        "f1": [],
        "true_positives": [],
        "false_positives": [],
        "false_negatives": [],
        "inference_time": 0
    }
       start_time_epoch = time.time()        
       for batch_idx, (rgb_map, target) in tqdm(enumerate(data_loader)):
              # start_time = time.time()
              optimizer.zero_grad()
              rgb_map = makeBVFeature(rgb_map, cnf.DISCRETIZATION_X, cnf.DISCRETIZATION_Y, cnf.boundary)
              output = model(rgb_map.float().cuda())
              # print(f"inference_time: {time.time() - inference_time}")

              loss = region_loss(output, target)
              loss.backward()
              optimizer.step()
              total_loss += loss.item() 
              wandb.log({"train_loss_1_iter": loss.item()})
       
       torch.save(model, f"ComplexYOLO_latest.pt")
       print("Epoch: %d, Time: %f, Loss: %f" %
             (epoch+1,
              time.time()-start_time_epoch,
              total_loss/len(data_loader)))
       
       print("\nEvaluating the model")
       epoch_metrics, _, _ = model_eval(model, val_loader, save_results=False)
       print(f"Loss: {epoch_metrics['total_loss']}\nPrecision: {np.mean(epoch_metrics['precision'], axis=0)[19]}\nRecall: {np.mean(epoch_metrics['recall'], axis=0)[19]}\nInference time: {epoch_metrics['inference_time']}\n")
       wandb.log({"train_loss": total_loss/len(data_loader), "val_loss": epoch_metrics['total_loss'], "val_precision_05": np.mean(epoch_metrics['precision'], axis=0)[19], "val_recall_05": np.mean(epoch_metrics['recall'], axis=0)[19]})
torch.save(model, f"ComplexYOLO_{epoch+1}e_{cnf.BEV_WIDTH}x{cnf.BEV_HEIGHT}_bev.pt")
print("Evaluation on test set\n")
epoch_metrics, _, _ = model_eval(model, test_loader, save_results=True, experiment_name=cnf.CONFIG["name"])
print(f"Loss: {epoch_metrics['total_loss']}",
       f"\nPrecision: {np.mean(epoch_metrics['precision'], axis=0)[19]}",
       f"\nRecall: {np.mean(epoch_metrics['recall'], axis=0)[19]}",
       f"\nInference time: {epoch_metrics['inference_time']}",
       f"\nEnergy: {epoch_metrics['energy_consumption']}\n")