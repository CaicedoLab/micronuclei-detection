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

# Evaluation
```python
import skimage
from dinomn import evaluation

mn_gt = skimage.io.imread(your_annotated_image_path) # make sure the annotations are masks
evaluation.segmentation_report(imid='My_Image', predictions=mn_predictions, gt=mn_gt, intersection_ratio=0.1)
```

# Train your own specialist model
- Expected training images extension: `.phenotype.tif`, nuclei mask extension: `.nuclei-clean.tif`, ground truth mask: `phenotype_outlines.png`.

```python
files = os.listdir(DIRECTORY)
annot_files = [x for x in filelist if x.endswith('.phenotype_outlines.png')]

validation_files = os.listdir(CURRENT_PATH + '/all_data_micronuclei_no_rescale/validation')
validation_files = [x for x in validation_files if x.endswith('.phenotype_outlines.png')]

device = f"cuda:{gpu}" if torch.cuda.is_available() else 'cpu'

model = mnmodel.MicronucleiModel(
    device=device,
    data_dir=DIRECTORY,
    training_files=training_files, 
    validation_files=validation_filelist,
    patch_size=256,
    gaussian=True
)

model.save(outdir=OUTPUT_DIR, model_name=MODEL_NAME)
```