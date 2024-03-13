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
train_dataset=ZOD_Dataset(root=cnf.CONFIG["dataset"],set='train')
val_dataset=ZOD_Dataset(root=cnf.CONFIG["dataset"],set='val')
test_dataset = ZOD_Dataset(root=cnf.CONFIG["dataset"],set='test')
data_loader = data.DataLoader(train_dataset, cnf.CONFIG["batch_size"], shuffle=True, num_workers=1) # The shm memory is not enough for num_workers >=2
val_loader = data.DataLoader(val_dataset, cnf.CONFIG["batch_size"], shuffle=False, num_workers=1)
test_loader = data.DataLoader(test_dataset, cnf.CONFIG["batch_size"], shuffle=False)

model = ComplexYOLO()
if cnf.CONFIG["resume_training"]:
       model = torch.load(cnf.CONFIG["resume_checkpoint"])
       print(f"Resuming training from {cnf.CONFIG['resume_checkpoint']}")

model.cuda()
# define optimizer
optimizer = optim.Adam(model.parameters(), lr=cnf.CONFIG["learning_rate"])
if cnf.CONFIG["scheduler"]:
       scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

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
       if cnf.CONFIG["resume_training"]:
              epoch += cnf.CONFIG["start_epoch"]
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
       print(f"Loss: {epoch_metrics['total_loss']}",
              f"\nAP: {epoch_metrics['AP']}",
              f"\nAR: {epoch_metrics['AR']}",
              f"\nmAP: {epoch_metrics['mAP']}",
              f"\nmAR: {epoch_metrics['mAR']}",
              f"\nInference time: {epoch_metrics['inference_time']}",
              f"\nEnergy: {epoch_metrics['energy_consumption']}\n")

       wandb.log({"train_loss": total_loss/len(data_loader),
                     "val_loss": epoch_metrics['total_loss'],
                     "val_mAP": epoch_metrics['mAP'],
                     "val_mAR": epoch_metrics['mAR'],
                     "val_AP_Vehicle": epoch_metrics['AP'][0],
                     "val_AP_VulnerableVehicle": epoch_metrics['AP'][1],
                     "val_AP_Pedestrian": epoch_metrics['AP'][2],
                     "val_AP_Animal": epoch_metrics['AP'][3],
                     "val_AP_StaticObject": epoch_metrics['AP'][4],
                     "val_AR_Vehicle": epoch_metrics['AR'][0],
                     "val_AR_VulnerableVehicle": epoch_metrics['AR'][1],
                     "val_AR_Pedestrian": epoch_metrics['AR'][2],
                     "val_AR_Animal": epoch_metrics['AR'][3],
                     "val_AR_StaticObject": epoch_metrics['AR'][4],
                     "val_tp_05": np.sum(epoch_metrics['true_positives'], axis=0)[19],
                     "val_fp_05": np.sum(epoch_metrics['false_positives'], axis=0)[19],
                     "val_fn_05": np.sum(epoch_metrics['false_negatives'], axis=0)[19],
                     "lr": optimizer.param_groups[0]['lr'],
                  })
  
       if cnf.CONFIG["scheduler"]:
              scheduler.step()

       torch.save(model, f"{cnf.CONFIG['name']}_{epoch+1}e.pt")       
torch.save(model, f"{cnf.CONFIG['name']}.pt")
print("Evaluation on test set\n")
epoch_metrics, _, _ = model_eval(model, test_loader, save_results=True, experiment_name=cnf.CONFIG["name"])
print(f"Loss: {epoch_metrics['total_loss']}",
       f"\nAP: {epoch_metrics['AP']}",
       f"\nAR: {epoch_metrics['AR']}",
       f"\nmAP: {epoch_metrics['mAP']}",
       f"\nmAR: {epoch_metrics['mAR']}",
       f"\nInference time: {epoch_metrics['inference_time']}",
       f"\nEnergy: {epoch_metrics['energy_consumption']}\n")