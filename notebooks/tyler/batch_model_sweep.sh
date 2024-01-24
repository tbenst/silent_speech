#!/bin/bash

# Check for the number of submissions argument
if [ $# -eq 0 ]
then
    echo "No arguments supplied. Defaulting to 4 submissions."
    num_submissions=4
else
    num_submissions=$1
fi

# Define the script to be submitted
script_path="~/nks/submit-owners.sh"

# Function to submit a job N times
submit_job () {
    for ((i=1; i<=num_submissions; i++))
    do
        ssbatch $@
    done
}

# Submit jobs with different options, N times each
submit_job $script_path --precision 32
submit_job $script_path --precision 32 --no-dtw
submit_job $script_path --precision 32 --no-dtw --no-supTcon
submit_job $script_path --precision 32 --no-dtw --no-crossCon
submit_job $script_path --precision 32 --no-dtw --no-crossCon --no-supTcon
submit_job $script_path --precision 32 --no-dtw --no-crossCon --no-supTcon --audio-lambda 0.0 --max-len 256000

echo "All jobs submitted."
