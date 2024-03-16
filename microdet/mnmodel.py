import os
import time
import torch
import skimage
import sklearn.metrics
import torchvision
import wandb

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.ops import sigmoid_focal_loss

import mnds
import detection
import vision_transformer as vit
        
        
class DiceLoss(torch.nn.Module):
    def __init__(self, smoothing=1e-5, reduction='mean'):
        super(DiceLoss, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, prediction, ground_truth):
        # Conclusion, do not use one-hot encoding
        probs = torch.sigmoid(prediction)
        ground_truth = ground_truth.long()
        
        num = probs * ground_truth # numerator
        num = torch.sum(num, dim=(2,3))  # Sum over all pixels NxCxHxW --> NxC
        
        den1 = probs * probs # 1st denominator
        den1 = torch.sum(den1, dim=(2,3))
        
        den2 = ground_truth * ground_truth # 2nd denominator
        den2 = torch.sum(den2, dim=(2,3))
        
        dice_loss = 2. * (num + self.smoothing) / (den1 + den2 + self.smoothing)  # Apply smoothing for numerical stability
        
        # give different weights
        
        if self.reduction == 'mean':
            dice_loss = 1 - torch.mean(dice_loss)
        elif self.reduction == 'sum':
            dice_loss = 1 - torch.sum(dice_loss)
        else:
            raise ValueError("'Reduction method must be either 'mean' or 'sum'")
        
        return dice_loss


class MicronucleiModel():
    
    def __init__(self, data_dir, device, training_files=[], validation_files=[], edges=False, patch_size=256, scale_factor=1.0):
        self.data_dir = data_dir
        self.device = device
        self.validation_files = validation_files
        self.patch_size = patch_size
        self.threshold = 0.0
        
        if len(training_files) > 0:
            self.training_set = mnds.MicronucleiDataset(
                filelist=training_files, 
                directory=data_dir, 
                mode="random",
                edges=edges,
                transform=mnds.detection_transforms,
                scale_factor=scale_factor,
                patch_size=patch_size
            )
        
        if len(validation_files) > 0:
            self.validation_set = mnds.MicronucleiDataset(
                filelist=validation_files, 
                directory=data_dir, 
                mode="fixed",
                edges=edges,
                scale_factor=scale_factor,
                patch_size=patch_size
            )
        
    def start_model(self, batch_size, learning_rate):
        # batch_size means number of images for each batch
        self.train_dataloader = DataLoader(self.training_set, batch_size=batch_size, shuffle=True)
        self.val_dataloader = DataLoader(self.validation_set, batch_size=4, shuffle=False)
        
        self.model = detection.DetectionModel(device=self.device)
        
        # self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.loss_fn = DiceLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate) #, momentum=0.9)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=self.optimizer,
            T_max=4
        )
        
        
    def train_one_epoch(self, epoch_index, tb_writer):
        running_loss = 0.
        last_loss = 0.

        self.train_dataloader.dataset.randomize_patch_index()
        for i, data in enumerate(self.train_dataloader):
            x, y = data
            self.optimizer.zero_grad()
            p = self.model(x.to(self.device))

            # Loss function   
            Y = y.to(self.device).float()
            # Y = Y.unsqueeze(dim=1)
            
            loss = self.loss_fn(p, Y)
            # loss = 0.05 * self.loss_fn(p, Y) + 0.95 * sigmoid_focal_loss(p, Y, alpha=0.25, gamma=1, reduction='mean')
            
            # Training instructions
            loss.backward()
            
            self.optimizer.step()

            # Report results
            running_loss += loss.item()
        return running_loss / i
    
    
    def train(self, epochs, batch_size, learning_rate, output_dir):
        def save_val_img(batch_idx, epoch_idx, prediction, ground_truth, pred_path, gt_path):
            
            # prediction and ground truth are 2 color channel, save each channel individually
            # 1st channel: micronuclei, 2st channel: nuclei
            
            # Hint: try to save micronuclei first and check
            # nuclei has already been checked in previous experiments
            
            # switch prediction to binary
            prediction = prediction > self.threshold
            prediction = prediction.float()
            
            if (batch_idx % 10 == 0) and (epoch_idx==19):     
                torchvision.utils.save_image(prediction, pred_path) 
                torchvision.utils.save_image(ground_truth, gt_path)
        
        self.start_model(batch_size, learning_rate)
        
        best_vloss = 1_000_000.
        epoch_number = 0

        start = time.time()
        for epoch in range(epochs):
            # Training
            print(f'EPOCH {epoch} - ', end='')
            T = time.time()
            self.model.train(True)
            avg_loss = self.train_one_epoch(epoch_number, None)
            
            # Update Learning Rate
            self.scheduler.step()

            # Validation
            running_vloss = 0.0
            self.model.eval()
            with torch.no_grad():
                for i, vdata in enumerate(self.val_dataloader):
                    # roughly 31 batches, each batch has 4 images
                    vin, vls = vdata
                    vout = self.model(vin.to(self.device))
                    Y = vls.to(self.device).float()
                    # Y = Y.unsqueeze(dim=1)
                    
                    # save validation image
                    # filename = self.validation_files[0].split('.')[0].split('_')[-1] # shorten filename
                    # if len(vin) != 1: # when they batch contains more than 1 image, select the 2nd one
                    #     save_val_img(
                    #         batch_idx=i,
                    #         epoch_idx=epoch,
                    #         prediction=vout[1], 
                    #         ground_truth=Y[1], 
                    #         gt_path=f'{self.data_dir}{output_dir}GT_Epoch{epoch}_{i}_{filename}.png',
                    #         pred_path=f'{self.data_dir}{output_dir}Pred_Epoch{epoch}_{i}_{filename}.png'
                    #     )
                    # else:
                    #     save_val_img(
                    #         batch_idx=i,
                    #         epoch_idx=epoch,
                    #         prediction=vout[0], 
                    #         ground_truth=Y[0], 
                    #         gt_path=f'{self.data_dir}{output_dir}GT_Epoch{epoch}_{i}_{filename}.png',
                    #         pred_path=f'{self.data_dir}{output_dir}Pred_Epoch{epoch}_{i}_{filename}.png'
                    #     )
                    
                    vloss = self.loss_fn(vout, Y)
                    # vloss = 0.05 * self.loss_fn(vout, Y) + 0.95 * sigmoid_focal_loss(vout, Y, alpha=0.25, gamma=1, reduction='mean')
                    running_vloss += vloss
            avg_vloss = running_vloss / (i+1)
            C = time.time() - T
            print(f'LOSS: Training: {avg_loss} - Validation: {avg_vloss} - Time: {C:.2f} secs')

            # log metrics to wandb
            wandb.log({"training_loss":avg_loss, "validation_loss":avg_vloss})
            
            epoch_number += 1

        C = time.time() - start
        print(f"\nTrainined finished in {C:.2f} seconds")
        wandb.finish()
        
    def validate(self):
        self.model.eval()

        mn_GT = []
        mn_PRED = []
        n_GT = []
        n_PRED = []

        ### make seperate report for micronuclei and nuclei, N*C*H*W, C = 0 is micronuclei, 1 is nuclei ###
        
        with torch.no_grad():
            for i, vdata in enumerate(self.val_dataloader):
                # Get predictions
                vin, vls = vdata
                output = self.model(vin.to(self.device))
                mn_output = output[:,0,:,:] > 0 # micronuclei
                mn_pred0 = mn_output.float()
                n_output = output[:,1,:,:] > 0
                n_pred0 = n_output.float()
                # pred0 = F.softmax(output, dim=1)
                
                mn_P = torch.reshape(mn_pred0, (-1, self.patch_size, self.patch_size))
                mn_pred = mn_P.cpu().numpy()
                n_P = torch.reshape(n_pred0, (-1, self.patch_size, self.patch_size))
                n_pred = n_P.cpu().numpy()
             
                # Collect predictions and ground truth
                # Micronuclei
                mn_PRED.append(mn_pred)
                mn_GT.append(vls[:,0,:,:].cpu().numpy())
                # Nuclei
                n_PRED.append(n_pred)
                n_GT.append(vls[:,1,:,:].cpu().numpy())
        
        mn_PRED = np.concatenate(mn_PRED, axis=0).reshape((-1,))
        mn_GT = np.concatenate(mn_GT, axis=0).reshape((-1,))
        n_PRED = np.concatenate(n_PRED, axis=0).reshape((-1,))
        n_GT = np.concatenate(n_GT, axis=0).reshape((-1,))
        
        mn_report = sklearn.metrics.classification_report(mn_GT, mn_PRED > 0.5)
        print('----- Micronuclei Classification Report ------')
        print(mn_report)
        
        n_report = sklearn.metrics.classification_report(n_GT, n_PRED > 0.5)
        print('----- Nuclei Classification Report ------')
        print(n_report)
        
        mn_jaccard_score = sklearn.metrics.jaccard_score(mn_GT, mn_PRED, average='weighted')
        print(f'Micronuclei Jaccard Score: {mn_jaccard_score:.4f} \n')
        
        n_jaccard_score = sklearn.metrics.jaccard_score(n_GT, n_PRED, average='weighted')
        print(f'Nuclei Jaccard Score: {n_jaccard_score:.4f} \n')
        
        
    def save(self, outdir="models/"):
        output_file = self.data_dir + outdir + self.validation_files[0].replace('phenotype_outlines.png','pth')
        torch.save(self.model.state_dict(), output_file)

        
    def load(self, model_name, model_dir="models/"):
        model_file = self.data_dir + model_dir + model_name
        self.model = detection.DetectionModel(device=self.device)
        self.model.load_state_dict(torch.load(model_file))
        self.model.to(self.device)
        
        
    def predict(self, image, stride=1, step=16, batch_size=512):
        classes = self.model.classifier.out_channels
        probabilities = np.zeros((classes, image.shape[0]//stride, image.shape[1]//stride), dtype=np.float32)
        counts = np.zeros((image.shape[0]//stride, image.shape[1]//stride), dtype=np.float32)
        TOKENS_PER_PATCH = self.patch_size // stride
        ones = np.ones((TOKENS_PER_PATCH, TOKENS_PER_PATCH))
        batch, coords = [], []

        self.model.eval()

        def batch_predict(batch, coords):
            B = torch.cat(batch, axis=0)
            # pred0 = F.softmax(self.model(B.to(self.device))) need to be changed
            output = self.model(B.to(self.device))
            output = output > self.threshold
            pred0 = output.float()
            P = torch.reshape(pred0, (-1, classes, TOKENS_PER_PATCH, TOKENS_PER_PATCH))
            P = P.cpu().numpy()

            for c in range(len(coords)):
                y = coords[c]["a"]
                x = coords[c]["b"]
                probabilities[:,y:y+TOKENS_PER_PATCH,x:x+TOKENS_PER_PATCH] += P[c]
                counts[y:y+TOKENS_PER_PATCH,x:x+TOKENS_PER_PATCH] += ones
            coords = []


        with torch.no_grad():
            for i in tqdm(range(0,image.shape[0]-self.patch_size+1, step)):
                a = i // stride
                for j in range(0,image.shape[1]-self.patch_size+1, step):
                    b = j // stride
                    vin = mnds.patch_to_rgb(image[i:i+self.patch_size,j:j+self.patch_size])
                    batch.append(vin[None,:,:,:])
                    coords.append({"i":i, "j":j, "a":a, "b":b})

                    if len(batch) == batch_size:
                        # Get predictions
                        batch_predict(batch, coords)
                        batch, coords = [], []

            if len(batch) > 0:
                batch_predict(batch, coords)
                batch, coords = [], []

        probabilities = probabilities/counts
        return probabilities
    
