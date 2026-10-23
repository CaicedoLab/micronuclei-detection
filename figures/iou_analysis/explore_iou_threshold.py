import os
import time
import skimage.morphology
import torch
import skimage
import pandas as pd
from tqdm import tqdm
import argparse
import numpy as np


def measures_at(threshold, IOU):
    matches = IOU > threshold
    
    true_positives = np.sum(matches, axis=1) == 1   # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Extra objects
    false_negatives = np.sum(matches, axis=1) == 0  # Missed objects
    
    assert np.all(np.less_equal(true_positives, 1))
    assert np.all(np.less_equal(false_positives, 1))
    assert np.all(np.less_equal(false_negatives, 1))
    
    TP, FP, FN = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
    
    f1 = 2*TP / (2*TP + FP + FN + 1e-9)

    prec = TP / (TP + FP)

    rec = TP / (TP + FN)

    return f1, prec, rec, TP, FP, FN


def parse_args():
    parser = argparse.ArgumentParser(
        description="Explore IoU Threshold",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Shows default values in help message
    )
    
    
    parser.add_argument('--folder_path', type=str, help='IoU (method) matrices path')
    parser.add_argument('--meta_path', type=str, help='Meta dataframe path')
    parser.add_argument('--iou_threshold', type=float, help='IoU matrices threhold')
    return parser.parse_args()


def main():
    args = parse_args()
    
    meta = pd.read_csv(args.meta_path)
    meta = meta[meta.split == 'test']
    folders = sorted(d for d in os.listdir(args.folder_path) 
                     if os.path.isdir(os.path.join(args.folder_path, d))) # list folders
    results = []
    
    for folder in tqdm(folders): 
        print(folder)
        IoU_files = os.listdir(os.path.join(args.folder_path, folder))
        IoU_files = sorted([file for file in IoU_files if file.endswith('.npy')])
        
        for file in tqdm(IoU_files):
            IoU = np.load(os.path.join(args.folder_path, folder, file))
            if IoU.size == 0:
                iou_score = 0
            else:
                # iou_score = np.mean(IoU.max(axis=0)) # Maximum column-wise
                iou_score = IoU.max(axis=0).sum() # average on 1238 objects
            
            imid = file.split('.npy')[0]
        
            if IoU.size == 0:
                precision = 0.0
                recall = 0.0
                f1 = 0.0
            else:
                f1, precision, recall, TP, FP, FN = measures_at(args.iou_threshold, IoU)
            
            filename = imid + '.tif'
            # print(filename)
            dataset = meta[meta.filenames == filename].datasets.item()
            # print(dataset)
            item = {'imid':imid, 'methods':folder, 'datasets':dataset, 'iou_threshold':args.iou_threshold, 
                    'iou_score':round(iou_score, 4), 'precision':round(precision, 4), 'recall':round(recall, 4), 'f1':round(f1, 4)}
            results.append(item)

    pd.DataFrame(results, columns=['imid', 'methods', 'datasets', 'iou_threshold', 'iou_score', 'precision', 'recall', 'f1']).to_csv(
        os.path.join(args.folder_path, f'macro_results_{args.iou_threshold}.csv'), index=False
    )
    
            
if __name__ == '__main__':
    main()