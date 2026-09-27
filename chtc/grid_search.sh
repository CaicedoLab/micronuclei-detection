#!/bin/bash
# tar -xvf microdet.tar.gz
# rm microdet.tar.gz
# cd microdet
export WANDB_API_KEY="$(cat ./.wandb_key)"
cd micronuclei-detection
mkdir config_output

experiment_id=""
learning_rate=""
batch_size=""

# Function to display usage
usage() {
    echo "Usage: $0 [-i experiment_id] [-r learning_rate] [-b batch_size]"
    exit 1
}

# Parse command line options.
while getopts ":i:l:r:b:" opt; do
    case ${opt} in
        i )
            experiment_id=$OPTARG
            ;;
        r )
            learning_rate=$OPTARG
            ;;
        b )
            batch_size=$OPTARG
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

output_file="experiment${experiment_id}_${learning_rate}_${batch_size}.txt"
# for imidx in $(seq 0 9); do

export PYTHONPATH="$PWD:$PYTHONPATH"
export USER="${USER:-yren86}"
export LOGNAME="$USER"
export HOME="$PWD"
export TORCHINDUCTOR_CACHE_DIR="$PWD/.cache/torchinductor"
python3 chtc/grid_search.py \
    --id "$experiment_id" \
    --path /scratch/yren86/annotated_mn_datasets \
    --gpu 0 \
    --model_type dino \
    --epochs 20 \
    --batch_size "$batch_size" \
    --loss_fn "combined" \
    --lr "$learning_rate" \
    --weight_decay 1e-6 \
    --scale 1.0 \
    --gaussian \
    --test_set \
    --step 32 \
    --prob_threshold 0.5 \
    --iou_threshold 0.1 \
    --wandb_mode > "$output_file"

cp "$output_file" config_output/