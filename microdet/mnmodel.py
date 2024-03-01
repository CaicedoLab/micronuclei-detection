import os
import time
import torch
import skimage
import sklearn.metrics
from torchvision.ops import sigmoid_focal_loss

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

import mnds
import detection
import vision_transformer as vit

class MicronucleiModel():
    
    def __init__(self, data_dir, device, training_files=[], validation_files=[], edges=False, patch_size=256, scale_factor=1.0):
        self.data_dir = data_dir
        self.device = device
        self.validation_files = validation_files
        
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
        self.train_dataloader = DataLoader(self.training_set, batch_size=batch_size, shuffle=True)
        self.val_dataloader = DataLoader(self.validation_set, batch_size=4, shuffle=False)
        
        self.model = detection.DetectionModel(device=self.device)
        
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate) #, momentum=0.9)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=self.optimizer,
            T_max=4,
            verbose=True
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
            Y = torch.reshape(y, (-1, 32*32)).to(self.device)
            loss = self.loss_fn(p, Y)
            # loss = sigmoid_focal_loss(p, Y, alpha=0.5, gamma=1, reduction='mean') + self.loss_fn(p, Y)
            # print(f'Loss shape: {loss.shape}')
            
            # Training instructions
            loss.backward()
            
            self.optimizer.step()
            
            # Report results
            running_loss += loss.item()
        return running_loss / i
    
    
    def train(self, epochs, batch_size, learning_rate):
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
                    vin, vls = vdata
                    vout = self.model(vin.to(self.device))
                    Y = torch.reshape(vls, (-1, 32*32)).to(self.device)
                    vloss = self.loss_fn(vout, Y)
                    # vloss = sigmoid_focal_loss(vout, Y, alpha=0.5, gamma=1, reduction='mean') + self.loss_fn(vout, Y)
                    running_vloss += vloss
            avg_vloss = running_vloss / (i+1)
            C = time.time() - T
            print(f'LOSS: Training: {avg_loss} - Validation: {avg_vloss} - Time: {C:.2f} secs')

            epoch_number += 1

        C = time.time() - start
        print(f"\nTrainined finished in {C:.2f} seconds")
        
    def validate(self):
        self.model.eval()

        GT = []
        PRED = []

        with torch.no_grad():
            for i, vdata in enumerate(self.val_dataloader):
                # Get predictions
                vin, vls = vdata
                pred0 = F.softmax(self.model(vin.to(self.device)), dim=1)
                P = torch.reshape(pred0, (-1, 32, 32))
                pred = P.cpu().numpy()

                # Collect predictions and ground truth
                PRED.append(pred)
                GT.append(vls.cpu().numpy())

        PRED = np.concatenate(PRED, axis=0).reshape((-1,))
        GT = np.concatenate(GT, axis=0).reshape((-1,))
        
        # Precision-recall curve
        display = sklearn.metrics.PrecisionRecallDisplay.from_predictions(
            GT, PRED, name="Detector" #, plot_chance_level=True
        )
        _ = display.ax_.set_title("Precision-Recall curve")
        
        # Classification report
        report = sklearn.metrics.classification_report(GT, PRED > 0.5)
        print(report)
        
        
    def save(self, outdir="models/"):
        output_file = self.data_dir + outdir + self.validation_files[0].replace('phenotype_outlines.png','pth')
        torch.save(self.model.state_dict(), output_file)

        
    def load(self, model_name, model_dir="models/"):
        model_file = self.data_dir + model_dir + model_name
        self.model = detection.DetectionModel(device=self.device)
        self.model.load_state_dict(torch.load(model_file))
        self.model.to(self.device)
        
        
    def predict(self, image, stride=8, patch_size=256, step=16, batch_size=512):
        probabilities = np.zeros((image.shape[0]//stride, image.shape[1]//stride), dtype=np.float32)
        counts = np.zeros((image.shape[0]//stride, image.shape[1]//stride), dtype=np.float32)
        TOKENS_PER_PATCH = patch_size // stride
        ones = np.ones((TOKENS_PER_PATCH, TOKENS_PER_PATCH))
        batch, coords = [], []

        self.model.eval()

        def batch_predict(batch, coords):
            B = torch.cat(batch, axis=0)
            pred0 = F.softmax(self.model(B.to(self.device)))
            P = torch.reshape(pred0, (-1, TOKENS_PER_PATCH, TOKENS_PER_PATCH))
            P = P.cpu().numpy()

            for c in range(len(coords)):
                y = coords[c]["a"]
                x = coords[c]["b"]
                probabilities[y:y+TOKENS_PER_PATCH,x:x+TOKENS_PER_PATCH] += P[c]
                counts[y:y+TOKENS_PER_PATCH,x:x+TOKENS_PER_PATCH] += ones
            coords = []


        with torch.no_grad():
            for i in tqdm(range(0,image.shape[0]-patch_size+1, step)):
                a = i // stride
                for j in range(0,image.shape[1]-patch_size+1, step):
                    b = j // stride
                    vin = mnds.patch_to_rgb(image[i:i+patch_size,j:j+patch_size])
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
    
