#!/bin/bash
cd /scr/yren/dataset_v3_all

shopt -s nullglob # useful when such file does not exist
# change groundtruth naming
for file in *.phenotype_phalloidin_outlines*; do
    mv "$file" "${file/phenotype_phalloidin_outlines/phenotype_outlines}"
done

# change input file naming
for file in *.phenotype_phalloidin.tif; do
    mv "$file" "${file/phenotype_phalloidin/phenotype}"
done
shopt -u nullglob