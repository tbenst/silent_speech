#!/bin/bash

# Check if at least one number is provided
if [ $# -eq 0 ]
then
    echo "No numbers supplied. Please provide a list of numbers."
    exit 1
fi

# Get the directory of the script itself
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Define the path to the SLURM submission script
submit_script="$script_dir/submit_beam_search.sh"

# Function to submit a job for each number in the list
submit_jobs () {
    for number in "$@"
    do
        sbatch "$submit_script" --run-id "GAD-$number"
    done
}

# Call the function with all the passed numbers
submit_jobs "$@"

echo "All jobs submitted."
