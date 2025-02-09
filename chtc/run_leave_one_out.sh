#!/bin/bash
# Best Experiment with LOO strategy in 18 images
tar -xvf microdet.tar.gz
rm microdet.tar.gz
cd microdet

gpu=0

usage() {
    echo "Usage: $0 [-i imidx]"
    exit 1
}

# Parse command line options.
while getopts ":i:" opt; do
    case ${opt} in
        i )
            microscope=$OPTARG
            ;;
    esac
done
shift $((OPTIND -1))

mkdir predictions
output_file="prediction_$microscope.txt"
python3 leave_one_out.py $microscope $gpu > $output_file
cp $output_file predictions/