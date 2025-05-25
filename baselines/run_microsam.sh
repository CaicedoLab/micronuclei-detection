#!/bin/bash

lst=("pilot_screen" "hela_rpe1" "mnfinder_train" "BBBC039")

# Loop over the list
for specialist in "${lst[@]}"
do
    python3 microsam_prediction.py "$specialist"
done