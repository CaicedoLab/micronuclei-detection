docker_image = vidit2003/microdet-chtc:latest
log = Screen$(Cluster).log
#
# Specify your executable (single binary or a script that runs several
#  commands), arguments, and a files for HTCondor to store standard
#  output (or "screen output").
#  $(Process) will be a integer number for each job, starting with "0"
#  and increasing for the relevant number of jobs.
executable = run_batch.sh
arguments = $(Process)
output = Screen$(Cluster)_$(Process).out
error = Screen$(Cluster)_$(Process).err
#
# Specify that HTCondor should transfer files to and from the
#  computer where each job runs. The last of these lines *would* be
#  used if there were any other files needed for the executable to use.
should_transfer_files = YES
when_to_transfer_output = ON_EXIT_OR_EVICT
transfer_input_files = prediction_batch.py, /home/vagrawal22/Caicedo/best_model.pth, /home/vagrawal22/Caicedo/mnds.py, /home/vagrawal22/Caicedo/evaluation.py, /home/vagrawal22/Caicedo/mnmodel.py, run_batch.sh, /home/vagrawal22/Caicedo/detection.py, /home/vagrawal22/Caicedo/vision_transformer.py, /home/vagrawal22/Caicedo/utils
# Tell HTCondor what amount of compute resources
#  each job will need on the computer where it runs.

# GPU Stuff
+WantGPULab = true
+GPUJobLength = "long"

request_gpus = 1
#require_gpus = (Capability > 7.9)
requirements = (Machine == "jcaicedogpu0000.chtc.wisc.edu" || Machine == "jcaicedogpu0001.chtc.wisc.edu")
request_cpus = 8
request_memory = 32GB
request_disk = 80GB
queue 1
