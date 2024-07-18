#!/bin/bash
declare -a number=("1" "2" "3" "4" "5" "6" "7")
declare -a id=("1" "2" "3" "4" "5" "6" "7" "8")





for i in "${number[@]}"
do

	for j in "${id[@]}"
	do
		echo "Making job plate $i $j"
		mkdir -p jobs/job_$i$j
		cd "jobs/job_$i$j"
		
		cp ../../template_container.sh template_container.sh
		cp ../../prediction_batch.py .
		cp ../../run_batch.sh .


		sed -i "s/number/$i/g" prediction_batch.py
		sed -i "s/number/$i/g" run_batch.sh

		sed -i "s/idea/$j/g" prediction_batch.py
		sed -i "s/idea/$j/g" run_batch.sh

		condor_submit template_container.sh

		cd -
	done
done
