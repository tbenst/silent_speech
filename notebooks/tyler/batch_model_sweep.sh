#!/bin/bash

# Check for the number of submissions argument
if [ $# -eq 0 ]
then
    echo "No arguments supplied. Defaulting to 5 submissions."
    num_submissions=5
else
    num_submissions=$1
fi

# Get the directory of the script itself
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Define the script to be submitted
script_path="$script_dir/submit-owners.sh"

# Function to submit a job N times
submit_job () {
    for ((i=1; i<=num_submissions; i++))
    do
        sbatch $@
    done
}

# Submit jobs with different options, N times each
submit_job $script_path --precision 32
submit_job $script_path --precision 32 --no-dtw
submit_job $script_path --precision 32 --no-dtw --no-supTcon
submit_job $script_path --precision 32 --no-dtw --no-crossCon --grad-accum 4
submit_job $script_path --precision 32 --no-crossCon --grad-accum 4
submit_job $script_path --precision 32 --no-dtw --no-crossCon --no-supTcon
submit_job $script_path --precision 32 --no-dtw --no-crossCon --no-supTcon --audio-lambda 0.0 --grad-accum 4
submit_job $script_path --precision 32 --no-dtw --no-crossCon --no-supTcon --emg-lambda 0.0

echo "All jobs submitted."
