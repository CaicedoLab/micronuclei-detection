#!/bin/bash
tar -xvf microdet.tar.gz
rm microdet.tar.gz
cd microdet


# usage: python3 training_model image_index number_of_gpus
python3 training_model.py 0 0