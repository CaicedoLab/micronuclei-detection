#!/bin/bash
tar -xvf microdet.tar.gz
# tar -xvf dataset_v2.tar.gz
rm microdet.tar.gz
# rm dataset_v2.tar.gz
cd microdet


# usage: python3 training_model image_index number_of_gpus
python3 training_model.py 0 0