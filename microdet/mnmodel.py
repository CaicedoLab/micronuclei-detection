import os
import time
import torch
import skimage
import sklearn.metrics

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import mnds
import extractor
import detection
import vision_transformer as vit

class MicronucleiModel:
    
    def __init__(self, data_dir, training_files, validation_files, device):
        self.data_dir = data_dir
        self.device = device
        self.validation_files = validation_files
        
        self.training_set = mnds.MicronucleiDataset(
            filelist=training_files, 
            directory=data_dir, 
            mode="random", 
            transform=mnds.detection_transforms
        )
        
        self.validation_set = mnds.MicronucleiDataset(
            filelist=validation_files, 
            directory=data_dir, 
            mode="fixed"
        )
        
    def start_model(self, batch_size, learning_rate):
        self.train_dataloader = DataLoader(self.training_set, batch_size=batch_size, shuffle=True)
        self.val_dataloader = DataLoader(self.validation_set, batch_size=4, shuffle=False)
        
        self.model = detection.DetectionModel(device=self.device)
        
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        
        
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

            # Validation
            running_vloss = 0.0
            self.model.eval()
            with torch.no_grad():
                for i, vdata in enumerate(self.val_dataloader):
                    vin, vls = vdata
                    vout = self.model(vin.to(self.device))
                    Y = torch.reshape(vls, (-1, 32*32)).to(self.device)
                    vloss = self.loss_fn(vout, Y)
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
                pred0 = self.model(vin.to(self.device))
                P = torch.reshape(pred0, (-1, 32, 32))
                pred = P.cpu().numpy()

                # Collect predictions and ground truth
                PRED.append(pred)
                GT.append(vls.cpu().numpy())

        PRED = np.concatenate(PRED, axis=0).reshape((-1,))
        GT = np.concatenate(GT, axis=0).reshape((-1,))
        
        # Precision-recall curve
        display = sklearn.metrics.PrecisionRecallDisplay.from_predictions(
            GT, PRED, name="Detector", plot_chance_level=True
        )
        _ = display.ax_.set_title("Precision-Recall curve")
        
        # Classification report
        report = sklearn.metrics.classification_report(GT, PRED > 0.5)
        print(report)
        
        
    def save(self):
        output_file = self.data_dir + "models/" + self.validation_files[0].replace('phenotype_outlines.png','pth')
        torch.save(self.model, output_file)
        