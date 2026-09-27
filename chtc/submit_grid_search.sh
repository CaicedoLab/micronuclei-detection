# Submit file
universe = docker
# link to self-built docker image
docker_image = yasuo0810/mndino_chtc:v4.1

# set the executable to run
executable = ./grid_search.sh
arguments = -i $(Process) -r $(learning_rate) -b $(batch_size)
log = log/log$(Process).log
error = error/job$(Process).err
output = output/job$(Process).out

transfer_input_files = /home/yren86/micronuclei-detection, /home/yren86/.wandb_key
transfer_output_files = micronuclei-detection/config_output

should_transfer_files = YES
when_to_transfer_output = ON_EXIT

request_cpus = 8
request_gpus = 1 

+WantGPULab = true
+GPUJobLength = "medium"

request_memory = 300GB
request_disk = 50GB

requirements = ( Machine == "jcaicedogpu0000.chtc.wisc.edu" || Machine == "jcaicedogpu0001.chtc.wisc.edu" || Machine == "jcaicedogpu0002.chtc.wisc.edu" || Machine == "jcaicedogpu0003.chtc.wisc.edu" || Machine == "jcaicedogpu0004.chtc.wisc.edu" )

queue learning_rate,batch_size from configuration.txt