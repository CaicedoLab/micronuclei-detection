import os
import time
import torch
import wandb

import numpy as np
import pandas as pd
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
# from torchvision.ops import sigmoid_focal_loss

import sys
from dinomn import mnds
from dinomn import detection as detection
        
        
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
        # Ground truth expected to be binaries, 0s and 1s
        # Conclusion, do not use one-hot encoding
        
        assert prediction.shape == ground_truth.shape, f'Predictions shape does not match the ground truth!'
        
        probs = torch.sigmoid(prediction)
        ground_truth = ground_truth.long() # convert torch.tensor 3.14 to 3, etc.
        
        num = probs * ground_truth # numerator
        num = torch.sum(num, dim=(2,3))  # Sum over all pixels NxCxHxW --> NxC
        
        den1 = probs * probs # 1st denominator
        den1 = torch.sum(den1, dim=(2,3))
        
        den2 = ground_truth * ground_truth # 2nd denominator
        den2 = torch.sum(den2, dim=(2,3))
        
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
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p = torch.sigmoid(inputs)
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
        super().__init__() # call initialization method from torch.nn.Module
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma, reduction=reduction)
        self.dice_loss = DiceLoss(alpha=dice_alpha, beta=dice_beta, smoothing=smoothing, reduction=reduction)
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(self, inputs, targets):
        loss_focal = self.focal_loss(inputs, targets)
        loss_dice = self.dice_loss(inputs, targets)
        combined_loss = self.focal_weight * loss_focal + self.dice_weight * loss_dice
        return combined_loss
        
        

class MicronucleiModel(torch.nn.Module):
    # repo_url = "yifanren/DinoMN"
    # pipeline_tag = "DinoMN-Model"
    # license = "mit"
    
    def __init__(self, device, data_dir='', training_files=[], validation_files=[], edges=False, patch_size=256, scale_factor=1.0, gaussian=False):
        super().__init__()
        
        self.data_dir = data_dir
        self.device = device
        self.validation_files = validation_files
        self.patch_size = patch_size
        self.gaussian = gaussian
        
        if len(training_files) > 0:
            self.training_set = mnds.MicronucleiDataset(
                filelist=training_files, 
                directory=data_dir, 
                mode="random",
                edges=edges,
                transform=mnds.detection_transforms,
                scale_factor=scale_factor,
                patch_size=patch_size,
                gaussian=gaussian
            )
        
        if len(validation_files) > 0:
            self.validation_set = mnds.MicronucleiDataset(
                filelist=validation_files, 
                # directory=data_dir,
                directory=os.getcwd() + '/all_data_micronuclei_no_rescale/validation',
                mode="fixed",
                edges=edges,
                scale_factor=scale_factor,
                patch_size=patch_size,
                gaussian=gaussian
            )
            self.need_validation_set = True
        else:
            self.need_validation_set = False
        
    def start_model(self, batch_size, learning_rate, loss_fn, finetune=False, weight_decay=1e-6):
        
        print(f'Number of training crops: {len(self.training_set)}')
        self.train_dataloader = DataLoader(self.training_set, batch_size=batch_size, shuffle=True)
        
        if self.need_validation_set:
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
            
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay) #, momentum=0.9)
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=self.optimizer,
            T_max=20, # max number of training epochs,
            eta_min=learning_rate * 0.1
        )
        
        
    def train_one_epoch(self):
        running_loss = 0.
        last_loss = 0.

        # For each training epoch, randomize the coordinates to reduce overfitting
        # Avoid that model sees the same regions in each training epoch
        self.train_dataloader.dataset.randomize_patch_index()
        for i, data in enumerate(self.train_dataloader):
            x, y = data
            
            self.optimizer.zero_grad()
            p = self.model(x.to(self.device))

            # output resolution: 128, interpolate to 256
            p = torch.nn.functional.interpolate(p, (self.patch_size, self.patch_size))
            
            # Loss function   
            Y = (y.to(self.device) > 0).float() # convert to binary (0 & 1)
            # Y = Y.unsqueeze(dim=1)
            
            # decoder_params = torch.cat([x.view(-1) for x in self.model.decoder.parameters()])
            # l2_regularization = l2_penalty * torch.norm(decoder_params, 2)
            loss = self.loss_fn(p, Y)
            
            # Training instructions
            loss.backward()
            
            # Update weights
            self.optimizer.step()
            
            # Report results
            running_loss += loss.item()
            # print(f'{i} iteration: running loss: {running_loss:.3f}')
        return running_loss / (i+1)
    
    
    def train(self, epochs, batch_size, learning_rate, loss_fn, finetune=False, weight_decay=1e-6):
        self.start_model(batch_size=batch_size, learning_rate=learning_rate, loss_fn=loss_fn, finetune=finetune, weight_decay=weight_decay)
        
        best_vloss = 1_000_000.

        start = time.time()
        for epoch in range(epochs):
            # Training
            # print(f'EPOCH {epoch} - ', end='') # comment only for grid search purpose
            T = time.time()
            self.model.train(True)
            avg_loss = self.train_one_epoch()
            
            # Update learning rate per epoch
            wandb.log({'current_lr':self.optimizer.param_groups[0]['lr']})
            self.scheduler.step()
            

            # Validation
            if self.need_validation_set:
                running_vloss = 0.0
                self.model.eval()
                with torch.no_grad():
                    for i, vdata in enumerate(self.val_dataloader):
                        vin, vls = vdata
                        vout = self.model(vin.to(self.device))
                        # output resolution: 128
                        vout = torch.nn.functional.interpolate(vout, (self.patch_size, self.patch_size))
                        Y = (vls.to(self.device) > 0).float() # convert to binary (0 & 1)
                        
                        vloss = self.loss_fn(vout, Y)
                        running_vloss += vloss
                avg_vloss = running_vloss / (i+1)
            C = time.time() - T
            # print(f'LOSS: Training: {avg_loss} - Validation: {avg_vloss} - Time: {C:.2f} secs') # comment only for grid search purpose

            # log metrics to wandb
            if self.need_validation_set:
                wandb.log({"Train_loss":avg_loss, "Validation_loss":avg_vloss}, commit=False) # avoid logging more steps than num of epochs
            else:
                wandb.log({"Train_loss":avg_loss})

        C = time.time() - start
        # print(f"\nTrainined finished in {C:.2f} seconds") # comment out for grid search
        wandb.log({"Train time":C})

        
    def save(self, outdir="models/", model_name='model'):
        # if self.need_validation_set: # LOO
        #     output_file = self.data_dir + outdir + self.validation_files[0].replace('phenotype_outlines.png','pth')
        # # elif self.gaussian:
        # #     output_file = f'{self.data_dir}{outdir}best_model_gaussian.pth'
        # else: # train with all images
        output_file = f'{self.data_dir}{outdir}{model_name}.pth'
        torch.save(self.model.state_dict(), output_file)

        
    def load(self, model_path):
        # model_file = model_dir + model_name
        self.model = detection.DetectionModel(device=self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        
    
    def _generate_fixed_coord(self, im, step, patch_size):
        C,H,W = im.shape
        X = list(range(0,W-patch_size+1, step))
        Y = list(range(0,H-patch_size+1, step))
        patches_per_image = len(X) * len(Y)
        X,Y = np.meshgrid(X,Y, indexing='ij')
        X = X.reshape((patches_per_image,))
        Y = Y.reshape((patches_per_image,))
        coord = np.stack((Y,X)).T
        
        return coord
        
        
    def predict(self, image, stride=1, step=16, batch_size=512):
        classes = self.model.classifier.out_channels
        probabilities = np.zeros((classes, image.shape[0]//stride, image.shape[1]//stride), dtype=np.float32)
        counts = np.zeros((image.shape[0]//stride, image.shape[1]//stride), dtype=np.float32)
        TOKENS_PER_PATCH = self.patch_size // stride # 256, same as self.patch_size
        ones = np.ones((TOKENS_PER_PATCH, TOKENS_PER_PATCH))
        batch, coords = [], []

        self.model.eval() # finetune not allowed, use pre-trained weights!

        def batch_predict(batch, coords):
            B = torch.cat(batch, axis=0)
            # pred0 = F.softmax(self.model(B.to(self.device))) need to be changed
            output = self.model(B.to(self.device))
            
            output = torch.nn.functional.interpolate(output, (self.patch_size,self.patch_size))
            
            # the output is not probability here (weights instead), last layer of detection model is binary classification, so we transform output to binary values.
            # classify pixel-wisely to 0 or 1, and scan all over the input image, and then average them, it is probability at the end.
            output = output > 0.0
            
            pred0 = output.float()
            P = torch.reshape(pred0, (-1, classes, TOKENS_PER_PATCH, TOKENS_PER_PATCH))
            # P.detach().cpu().numpy() might be better
            P = P.cpu().numpy()
            # print(f'P shape in batch_predict(): {P.shape}')
            
            for c in range(len(coords)):
                y = coords[c]["a"]
                x = coords[c]["b"]
                probabilities[:,y:y+TOKENS_PER_PATCH,x:x+TOKENS_PER_PATCH] += P[c]
                counts[y:y+TOKENS_PER_PATCH,x:x+TOKENS_PER_PATCH] += ones
            coords = []
        
        with torch.no_grad(): # turn off gradients for inference
            image = mnds.patch_to_rgb(image)
            C = self._generate_fixed_coord(im=image, step=step, patch_size=self.patch_size)
            
            for i in range(len(C)): # loop through coordinates
                y,x = C[i,:]
                coords.append({'a':y, 'b':x})
                
                crop = image[:,y:y+self.patch_size,x:x+self.patch_size]
                batch.append(crop[None,:,:,:])
                
                if len(batch) == batch_size:
                        # Get predictions
                        batch_predict(batch, coords)
                        batch, coords = [], []
            
            # remaining batches that is lower than specified batch size
            if len(batch) > 0:
                batch_predict(batch, coords)
                batch, coords = [], []

        probabilities = probabilities/counts
        return probabilities
    