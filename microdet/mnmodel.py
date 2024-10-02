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
# from torchvision.ops import sigmoid_focal_loss

import mnds
import detection
import vision_transformer as vit
        
        
class DiceLoss(torch.nn.Module):
    def __init__(self, alpha=0.8, beta=0.2, smoothing=1e-5, reduction='mean'):
        """_summary_

        Args:
            alpha (_type_): weight for micronuclei class, default=0.8
            beta (_type_): weight for nuclei class, default=0.2
            smoothing (_type_, optional): smoothing parameter for numerical stability. Defaults to 1e-5.
            reduction (str, optional): Reduction method. Defaults to 'mean'.
        """
        super(DiceLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, prediction, ground_truth):
        # Conclusion, do not use one-hot encoding
        
        assert prediction.shape == ground_truth.shape, f'Predictions shape does not match the ground truth!'
        
        probs = torch.sigmoid(prediction)
        ground_truth = ground_truth.long()
        
        num = probs * ground_truth # numerator
        num = torch.sum(num, dim=(2,3))  # Sum over all pixels NxCxHxW --> NxC
        
        den1 = probs * probs # 1st denominator
        den1 = torch.sum(den1, dim=(2,3))
        
        den2 = ground_truth * ground_truth # 2nd denominator
        den2 = torch.sum(den2, dim=(2,3))
        
        # dice_loss = 2. * (num+ self.smoothing) / (den1 + den2 + self.smoothing)
        dice_loss_mn = 2. * (num[:,0]+ self.smoothing) / (den1[:,0] + den2[:,0] + self.smoothing)
        dice_loss_n = 2. * (num[:,1]+ self.smoothing) / (den1[:,1] + den2[:,1] + self.smoothing)
        
        if self.reduction == 'mean':
            dice_loss = 1 - (self.alpha * torch.mean(dice_loss_mn) + self.beta * torch.mean(dice_loss_n))
            # dice_loss = 1 - torch.mean(dice_loss)
        elif self.reduction == 'sum':
            dice_loss = 1 - (self.alpha * torch.sum(dice_loss_mn) + self.beta * torch.sum(dice_loss_n))
            # dice_loss = 1 - torch.sum(dice_loss)
        else:
            raise ValueError("'Reduction method must be either 'mean' or 'sum'")
        
        return dice_loss
    
class FocalLoss(torch.nn.Module):
    """_summary_
    Code are copied from torchvision.ops.sigmoid_focal_loss function
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction: str="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
            
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        else:
            raise ValueError(
                f"Invalid Value for arg 'reduction': '{self.reduction} \n Supported reduction modes: 'mean', 'sum'"
            )
        return loss
    
class CombinedFocalDiceLoss(torch.nn.Module):
    def __init__(self, focal_weight=0.95, dice_weight=0.05, alpha=0.25, gamma=2.0, reduction='mean', dice_alpha=0.8, dice_beta=0.2, smoothing=1e-5):
        super(CombinedFocalDiceLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma, reduction=reduction)
        self.dice_loss = DiceLoss(alpha=dice_alpha, beta=dice_beta, smoothing=smoothing, reduction=reduction)
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(self, inputs, targets):
        loss_focal = self.focal_loss(inputs, targets)
        loss_dice = self.dice_loss(inputs, targets)
        combined_loss = self.focal_weight * loss_focal + self.dice_weight * loss_dice
        return combined_loss
        
        

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
        else:
            self.need_validation_set = False
        
    def start_model(self, batch_size, learning_rate, loss_fn, finetune=False, weight_decay=1e-6):
        # batch_size means number of images for each batch
        self.train_dataloader = DataLoader(self.training_set, batch_size=batch_size, shuffle=True)
        
        if self.need_validation_set: # case for train the best model with all 18 images
            self.val_dataloader = DataLoader(self.validation_set, batch_size=4, shuffle=False)
        
        self.model = detection.DetectionModel(device=self.device, finetune=finetune)
        
        # self.loss_fn = torch.nn.BCEWithLogitsLoss()
        if loss_fn == 'dice':
            self.loss_fn = DiceLoss(alpha=0.8, beta=0.2, smoothing=1e-5, reduction='mean')
        elif loss_fn == 'focal':
            self.loss_fn = FocalLoss(alpha=0.25, gamma=1, reduction='mean')
        elif loss_fn == 'combined':
            # Use all default parameters, gamma = 2 so far is good
            self.loss_fn = CombinedFocalDiceLoss(focal_weight=0.95, dice_weight=0.05, alpha=0.25, gamma=2, reduction='mean', dice_alpha=0.8, dice_beta=0.2, smoothing=1e-5)
            
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay) #, momentum=0.9) # add weight decy / regularization
        
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer=self.optimizer,
        #     T_max=2,
        #     eta_min=learning_rate * 0.1
        # )
        
        
    def train_one_epoch(self, epoch_index, tb_writer):
        running_loss = 0.
        last_loss = 0.

        self.train_dataloader.dataset.randomize_patch_index()
        for i, data in enumerate(self.train_dataloader):
            x, y = data
            
            self.optimizer.zero_grad()
            p = self.model(x.to(self.device))

            # output resolution: 128
            p = torch.nn.functional.interpolate(p, (256,256))
            
            # Loss function   
            Y = y.to(self.device).float()
            # Y = Y.unsqueeze(dim=1)
            
            # decoder_params = torch.cat([x.view(-1) for x in self.model.decoder.parameters()])
            # l2_regularization = l2_penalty * torch.norm(decoder_params, 2)
            loss = self.loss_fn(p, Y)
            
            # Training instructions
            loss.backward()
            
            self.optimizer.step()

            # Report results
            running_loss += loss.item()
        return running_loss / i
    
    
    def train(self, epochs, batch_size, learning_rate, loss_fn, output_dir, finetune=False, weight_decay=1e-6):
        def save_val_img(batch_idx, epoch_idx, prediction, ground_truth, pred_path, gt_path):
            prediction = prediction > self.threshold
            prediction = prediction.float()
            
            if (batch_idx % 10 == 0) and (epoch_idx==19):     
                torchvision.utils.save_image(prediction, pred_path) 
                torchvision.utils.save_image(ground_truth, gt_path)
        
        self.start_model(batch_size=batch_size, learning_rate=learning_rate, loss_fn=loss_fn, finetune=finetune, weight_decay=weight_decay)
        
        best_vloss = 1_000_000.
        epoch_number = 0

        start = time.time()
        for epoch in range(epochs):
            # Training
            # print(f'EPOCH {epoch} - ', end='') # comment only for grid search purpose
            T = time.time()
            self.model.train(True)
            avg_loss = self.train_one_epoch(epoch_number, None)
            
            # Update Learning Rate
            # self.scheduler.step()
            # current_lr = self.optimizer.param_groups[0]['lr']
            # wandb.log({"Scheduler LR":current_lr})

            # Validation
            if self.need_validation_set:
                running_vloss = 0.0
                self.model.eval()
                with torch.no_grad():
                    for i, vdata in enumerate(self.val_dataloader):
                        vin, vls = vdata
                        vout = self.model(vin.to(self.device))
                        # output resolution: 128
                        vout = torch.nn.functional.interpolate(vout, (256,256))
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
                        running_vloss += vloss
                avg_vloss = running_vloss / (i+1)
            C = time.time() - T
            # print(f'LOSS: Training: {avg_loss} - Validation: {avg_vloss} - Time: {C:.2f} secs') # comment only for grid search purpose

            # log metrics to wandb
            if self.need_validation_set:
                wandb.log({"Train_loss":avg_loss, "Validation_loss":avg_vloss})
            else:
                wandb.log({"Train_loss":avg_loss})
            
            epoch_number += 1

        C = time.time() - start
        # print(f"\nTrainined finished in {C:.2f} seconds") # comment out for grid search
        wandb.log({"Train time":C})
        
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
                
                output = torch.nn.functional.interpolate(output, (256,256))
                
                mn_output = output[:,0,:,:] > self.threshold # micronuclei
                mn_pred0 = mn_output.float()
                n_output = output[:,1,:,:] > self.threshold
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
        
        mn_report = sklearn.metrics.classification_report(mn_GT, mn_PRED)
        print('----- Micronuclei Classification Report ------')
        print(mn_report)
        
        n_report = sklearn.metrics.classification_report(n_GT, n_PRED)
        print('----- Nuclei Classification Report ------')
        print(n_report)
        
        mn_jaccard_score = sklearn.metrics.jaccard_score(mn_GT, mn_PRED, average='weighted')
        print(f'Micronuclei Jaccard Score: {mn_jaccard_score:.4f} \n')
        
        n_jaccard_score = sklearn.metrics.jaccard_score(n_GT, n_PRED, average='weighted')
        print(f'Nuclei Jaccard Score: {n_jaccard_score:.4f} \n')
        
        
    def save(self, outdir="models/"):
        # output_file = self.data_dir + outdir + self.validation_files[0].replace('phenotype_outlines.png','pth')
        output_file = f'{self.data_dir}{outdir}best_model.pth'
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
            
            output = torch.nn.functional.interpolate(output, (256,256))
            
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
    
