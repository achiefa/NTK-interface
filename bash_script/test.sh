#!/bin/bash

PATH_TO_FIT_FOLDER=$2
PATH_TO_SCRIPT="/Users/s2569857/Codes/NTK-interface/build/run"
PATH_TO_FIT_FOLDER=$2
OUT_PATH="${2}/out"
TRAIN="${PATH_TO_SCRIPT}/train"

# This is the concurrency limit
MAX_POOL_SIZE=10

# Jobs will be loaded from this file
JOB_LIST=job-list.txt

# Output file
OUTPUT=output.txt

# This is used within the program. Do not change.
CURRENT_POOL_SIZE=0


# This is a just a function to print the output as a log with timestamp
_log() {
        echo " $(date +'[%F %T]') - $1"
}

# This is the custom function to process each job read from the file
process_job() {
  # customize your job function as required
  # in our example, we just "ping" each hostname read from the file
  nohup ${TRAIN} -i $1 $2 > "${OUT_PATH}/output_"${1%".txt"} &
}



# ------ This is the main program code --------

# Starting the timer
T1=$(date +%s)

mkdir -p ${OUT_PATH}
# Reading the $JOB_LIST file, line by line
for ((i = 1; i <= $1; i++))
do
  
  # This is the blocking loop where it makes the program to wait if the job pool is full
  while [ $CURRENT_POOL_SIZE -ge $MAX_POOL_SIZE ]; do
    _log "Pool is full. waiting for jobs to exit..."
    sleep 10
    
    # The above "echo" and "sleep" is for demo purposes only.
    # In a real usecase, remove those two and keep only the following line
    # It will drastically increase the performance of the script
    CURRENT_POOL_SIZE=$(jobs | wc -l)
  done
  
  
  # This is a custom function to process each job read from the file
  # It calls the custom function with each line read by the $JOB_LIST and send it to background for processing
  _log "Starting job $line"  
  process_job $i $2
  
  # When a new job is created, the program updates the $CURRENT_POOL_SIZE variable before next iteration
  CURRENT_POOL_SIZE=$(jobs | wc -l)
  _log "Current pool size = $CURRENT_POOL_SIZE"
  
  
done # this is where we feed the $JOB_LIST file for the read operation

# wait for all background jobs (forks) to exit before exiting the parent process
wait

# Ending the timer
T2=$(date +%s)

_log "All jobs completed in $((T2-T1)) seconds. Parent process exiting."
_log "Final output is written in $OUTPUT"
exit 0