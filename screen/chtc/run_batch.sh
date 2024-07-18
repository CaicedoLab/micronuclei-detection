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
