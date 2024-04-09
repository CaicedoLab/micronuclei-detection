tar -xvf microdet.tar.gz
rm microdet.tar.gz
cd microdet


usage() {
    echo "Usage: $0 [-t tile_id] [-i image_id]"
    exit 1
}

# Parse command line options.
while getopts ":t:i:" opt; do
    case ${opt} in
        t )
            tile_id=$OPTARG
            ;;
        i )
            imidx=$OPTARG
            ;;
    esac
done
shift $((OPTIND -1))

output_file="best_experiment_$tile_id.txt"
python3 run_best_experiment.py $imidx > "$output_file"
cp "$output_file" best_experiment/