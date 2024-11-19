#!/bin/bash
# Best model trained on all 18 images, predict on new data
tar -xvf microdet.tar.gz
rm microdet.tar.gz
cd microdet

gpu=0

usage() {
    echo "Usage: $0 [-i image_id]"
    exit 1
}

# Parse command line options.
while getopts ":i:" opt; do
    case ${opt} in
        i )
            imid=$OPTARG
            ;;
    esac
done
shift $((OPTIND -1))

# usage: python3 prediction.py gpu imidx
# mkdir predictions
# output_file="prediction_output$imid.txt"
python3 prediction.py $imid $gpu # > "$output_file"
# cp "$output_file" predictions/