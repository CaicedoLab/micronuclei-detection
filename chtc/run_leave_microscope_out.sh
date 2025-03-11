#!/bin/bash
tar -xvf microdet.tar.gz
rm microdet.tar.gz
cd microdet

gpu=0

usage() {
    echo "Usage: $0 [-i microscope]"
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

mkdir oversampled_predictions
output_file="prediction_$microscope.txt"
python3 leave_microscope_out.py $microscope $gpu > $output_file
cp $output_file oversampled_predictions/