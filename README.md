# Micronuclei Detection
Detecting micronuclei in images using Transformer Networks


# Install package
```bash
pip install dinomn
```

# Load the model
```python
import torch
from dinomn import mnmodel
from dinomn import evaluation
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(repo_id="yifanren/DinoMN", filename="DinoMN.pth")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = mnmodel.MicronucleiModel(device=device)
model.load(model_path)
```

#  Make predictions
```python
import skimage
import numpy

STEP = 64 # recommended value
PREDICTION_BATCH = 4
THRESHOLD = 0.5

im = skimage.io.imread(your_image_path)
im = np.array((im - np.min(im))/(np.max(im) - np.min(im)), dtype="float32") # normalize image
probabilities = model.predict(im, stride=1, step=STEP, batch_size=PREDICTION_BATCH)

mn_predictions = probabilities[0,:,:] > THRESHOLD
nuclei_predictions = probabilities[1,:,:] > THRESHOLD
```