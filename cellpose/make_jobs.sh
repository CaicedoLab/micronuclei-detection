#!/bin/bash
declare -a file=('20X_c0-DAPI_A1_Tile-10' '20X_c0-DAPI_A1_Tile-110' '20X_c0-DAPI_A1_Tile-130' '20X_c0-DAPI_A1_Tile-150'
'20X_c0-DAPI_A1_Tile-170' '20X_c0-DAPI_A1_Tile-190' '20X_c0-DAPI_A1_Tile-30' '20X_c0-DAPI_A1_Tile-50' '20X_c0-DAPI_A1_Tile-70' '20X_c0-DAPI_A1_Tile-90'
'C2-20X_c0-DAPI-GFP_A1_Tile-17' 'C2-20X_c0-DAPI-GFP_A1_Tile-5' 'C2-20X_c0-DAPI-GFP_A2_Tile-5' 'C2-20X_c0-DAPI-GFP_A3_Tile-6' 'C2-20X_c0-DAPI-GFP_B1_Tile-8'
'C2-20X_c0-DAPI-GFP_B2_Tile-20' 'C2-20X_c0-DAPI-GFP_B3_Tile-10' 'C2-20X_c0-DAPI-GFP_B3_Tile-16')

for i in "${file[@]}"
do
    echo "Making job plate $i"
    mkdir -p jobs/job_$i
    cd "jobs/job_$i"
    
    cp ../../chtc-container.sh chtc-container.sh
    cp ../../run_train.sh .
    cp ../../modeltrainleaveoneout.py .


    sed -i "s/image_name/$i/g" modeltrainleaveoneout.py

    condor_submit chtc-container.sh

    cd -
done

