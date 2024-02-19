#!/bin/bash
#SBATCH -p henderj
#SBATCH --time=7-00:00:00  # Run for 7 days
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --constraint=GPU_MEM:80GB
#SBATCH --signal=B:SIGUSR1@90

# Change to the specified directory
cd $GROUP_HOME/tyler/jobs

# Create a file for logging activity
touch "$(date -Iseconds).log"

# avoid inactive
end=$((SECONDS+604800))  # 7 days in seconds
while [ $SECONDS -lt $end ]; do
    # Append the current date and time to a log file
    echo "$(date)" >> stay_alive.log
    sleep 360  # Sleep for 6 min
done
