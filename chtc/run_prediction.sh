#!/bin/bash
tar -xvf microdet.tar.gz
rm microdet.tar.gz
cd microdet


# usage: python3 prediction.py imidx gpu
for i in $(seq 0 9); do
    python3 prediction.py $i 0
done > prediction_output.txt