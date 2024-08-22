#!/bin/bash
# Best experiment with 18 images
tar -xvf microdet.tar.gz
rm microdet.tar.gz
cd microdet


# usage: python3 prediction.py gpu
python3 prediction.py 0 > prediction_output.txt
# for i in $(seq 0 9); do
#     python3 prediction.py $i 0
# done > prediction_output.txt