#!/bin/bash
tar -xvf dinomn.tar.gz
rm dinomn.tar.gz
cd dinomn

gpu=0

usage() {
    echo "Usage: $0 [-i cell_line]"
    exit 1
}

# Parse command line options.
while getopts ":i:" opt; do
    case ${opt} in
        i )
            cell_line=$OPTARG
            ;;
    esac
done
shift $((OPTIND -1))

mkdir predictions
output_file="prediction_$cell_line.txt"
python3 leave_one_out.py $cell_line $gpu > $output_file
cp $output_file predictions/