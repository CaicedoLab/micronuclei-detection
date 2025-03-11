#!/bin/bash

gpu=3

python3 training_model.py $gpu
python3 prediction.py $gpu > DinoMN_oversample_experiment.txt # prediction_output_model_v3_no_rescale.txt
# for i in {0..5}
# do
#     python3 leave_one_out.py $i $gpu
# done