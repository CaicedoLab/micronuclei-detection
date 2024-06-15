# My very first CHTC job
#
# print a 'hello' message to the job's terminal output:
echo "Hello CHTC from Job $1 running on `whoami`@`hostname`"

mkdir plate_number_images_idea
mkdir output
export HOME=`pwd`?


unzip /staging/groups/caicedo_group/micronuclei/screen-plates-divided/plate_number_images_idea.zip -d plate_number_images_idea


python3 prediction_batch.py
ls


zip -r plate_number_maps_idea.zip output
mv plate_number_maps_idea.zip /staging/groups/caicedo_group/micronuclei/prob-maps
rm -rf output
rm -rf plate_number_images_idea
[vagrawal22@ap2001 Caicedo]$ cat make_jobs.sh 
#!/bin/bash
declare -a number=("1" "2" "3" "4" "5" "7")
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
