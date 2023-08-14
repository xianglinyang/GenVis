#!/bin/bash

# Define Hyperparameters
declare -a iterations=("10" "100" "200")
declare -a ratios=("0.0008" "0.001" "0.002" "0.003" "0.05" "0.1" "0.5" "1.0")
content_path="/home/xianglin/projects/DVI_data/resnet18_cifar10"
script="/home/xianglin/projects/git_space/GenVis/single_dvi_main.py"

# Limit number of parallel jobs
MAX_JOBS=5
COUNT=0

# Loop through the iterations
for iteration in "${iterations[@]}"; do
    # Loop through the array
    for ratio in "${ratios[@]}"; do
        # Run each Python script in the background, passing the string arguments
        echo "Running with I:${iteration},ratio:${ratio}"
        python "${python_script}" "-c" "${content_path}" "-i" "${iteration}" "-r" "${ratio}"> "${script}.log" 2>&1 &

        # Increment the counter
        ((COUNT++))

        # If the number of current jobs is equal to MAX_JOBS, wait for any job to finish
        if [ $COUNT -eq $MAX_JOBS ]; then
            wait -n
            # Decrement the counter
            ((COUNT--))
        fi
    done
done

# Wait for all remaining background jobs to finish
wait

# Print completion message
echo "All Python scripts have been executed."
