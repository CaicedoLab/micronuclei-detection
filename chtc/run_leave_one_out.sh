#!/bin/bash
tar -xvf dinomn.tar.gz
rm dinomn.tar.gz
cd dinomn

gpu=0

usage() {
    echo "Usage: $0 [-i subset]"
    exit 1
}

# Parse command line options.
while getopts ":i:" opt; do
    case ${opt} in
        i )
            subset=$OPTARG
            ;;
    esac
done
shift $((OPTIND -1))

mkdir predictions
output_file="prediction_$subset.txt"
python3 leave_one_out.py $subset $gpu > $output_file
cp $output_file predictions/