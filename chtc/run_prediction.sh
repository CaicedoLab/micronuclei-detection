#!/bin/bash
tar -xvf dinomn.tar.gz
rm dinomn.tar.gz
cd dinomn

gpu=0

usage() {
    echo "Usage: $0 [-i step_size]"
    exit 1
}

# Parse command line options.
while getopts ":i:" opt; do
    case ${opt} in
        i )
            step=$OPTARG
            ;;
    esac
done
shift $((OPTIND -1))

# usage: python3 prediction.py gpu imidx
mkdir predictions
output_file="prediction_step$step.txt"
python3 prediction.py $step $gpu > "$output_file"
cp "$output_file" predictions/