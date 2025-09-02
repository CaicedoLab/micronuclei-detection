# mn-DINO
Detecting micronuclei in images using DINOv2 backbone

Refer to [tutorial notebook](tutorials/) for real examples


# Install package
```bash
pip install mn-dino
```

# Load the model
```python
import torch
from mn-dino import mnmodel
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(repo_id="yifanren/mn-DINO", filename="latest.pth")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = mnmodel.MicronucleiModel(device=device)
model.load(model_path)
```

#  Make predictions
```python
import skimage
import numpy as np

STEP = 64 # recommended value
PREDICTION_BATCH = 4
THRESHOLD = 0.5

im = skimage.io.imread(your_image_path)
im = np.array((im - np.min(im))/(np.max(im) - np.min(im)), dtype="float32") # normalize image
probabilities = model.predict(im, stride=1, step=STEP, batch_size=PREDICTION_BATCH)

mn_predictions = probabilities[0,:,:] > THRESHOLD
nuclei_predictions = probabilities[1,:,:] > THRESHOLD
```

# Evaluation
```python
import skimage
from mn-dino import evaluation

mn_gt = skimage.io.imread(your_annotated_image_path) # make sure the annotations are masks
evaluation.segmentation_report(imid='My_Image', predictions=mn_predictions, gt=mn_gt, intersection_ratio=0.1)
```

# Train your own specialist model
- Expected file extension of training images and nuclei masks is `.tif`, the corresponding training masks is `.png`. Following values are tunable if retraining on non-micronucleus subcellular datasets.
- Combined loss = 0.8 * subcellular loss + 0.2 * nuclei loss.

```python
device = f"cuda:{gpu}" if torch.cuda.is_available() else 'cpu'
model = mnmodel.MicronucleiModel(
    device=device,
    data_dir=DIRECTORY,
    patch_size=256,
    scale_factor=1.0,
    gaussian=True
)

model.train(epochs=20, 
            batch_size=4, 
            learning_rate=1e-5, 
            loss_fn='combined',
            finetune=True,
            weight_decay=1e-6,
            wandb_mode=False
)

model.save(outdir=OUTPUT_DIR, model_name=MODEL_NAME)
```

# Reproducing mn-dino Training Experiment
```bash
git pull git@github.com:CaicedoLab/micronuclei-detection.git
cd micronuclei-detection
```

Training
```python
python3 training_model.py --path 'path to micronuclei dataset' --gpu 0 --epochs 20 --loss_fn 'combined' --lr 1e-6 --scale 1.0 --finetune --gaussian
```

Prediction
```python
python3 prediction.py --path 'path to micronuclei dataset' --gpu 0 --step 64 --batch_size 4 --prob_threshold 0.5 --iou_threshold 0.1 --scale 1
```

Add `--wandb_mode` if user wants to show loss on Weights and Biases
