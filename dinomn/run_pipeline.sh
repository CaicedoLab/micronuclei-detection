#!/bin/bash

gpu=3

python3 training_model.py $gpu
python3 prediction.py $gpu > predictions.txt