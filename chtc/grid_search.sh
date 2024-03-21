#!/bin/bash
tar -xvf microdet.tar.gz
rm microdet.tar.gz
cd microdet

experiment_id=""
loss_fn=""
learning_rate=""
batch_size=""
finetune=""

# Function to display usage
usage() {
    echo "Usage: $0 [-i experiment_id] [-l loss_fn] [-r learning_rate] [-b batch_size] [-f finetune]"
    exit 1
}

# Parse command line options.
while getopts ":i:l:r:b:f:" opt; do
    case ${opt} in
        i )
            experiment_id=$OPTARG
            ;;
        l )
            loss_fn=$OPTARG
            ;;
        r )
            learning_rate=$OPTARG
            ;;
        b )
            batch_size=$OPTARG
            ;;
        f )
            finetune=$OPTARG
            ;;
        \? )
            echo "Invalid Option: -$OPTARG" 1>&2
            usage
            ;;
        : )
            echo "Invalid option: $OPTARG requires an argument" 1>&2
            usage
            ;;
    esac
done
shift $((OPTIND -1))

output_file="experiment${experiment_id}_${loss_fn}_${learning_rate}_${batch_size}_${finetune}.txt"
for imidx in $(seq 0 9); do
    python3 grid_search.py $imidx "$loss_fn" "$learning_rate" "$batch_size" "$finetune" 0
done > "$output_file"

cp "$output_file" config_output/