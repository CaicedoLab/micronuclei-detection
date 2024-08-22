#!/bin/bash
# Best experiment with 18 images
tar -xvf microdet.tar.gz
rm microdet.tar.gz
cd microdet


# To train with 18 images
# usage: python3 training_model.py gpu_index
python3 training_model.py 0 > train_output.txt
# for i in $(seq 0 9); do
#     python3 training_model.py $i 0
# done > train_output.txt