#!/bin/bash

# Define the grid sweep values
# Number of most influential tags to track
TAG_COUNTS=(10 25 50)

# Number of top words from reviews to include in Bag of Words (0 = disabled)
WORD_COUNTS=(0 50 100)

echo "Starting Grid Sweep"
echo "Tags, Words, AUC" > sweep_results.csv

# Loop through combinations
for t in "${TAG_COUNTS[@]}"; do
    for w in "${WORD_COUNTS[@]}"; do
        echo "Launching job: Tags=$t, Words=$w"
        
        # Run in background (&) and pipe output to a specific log file
        (python grid_search_model.py --top_n_tags $t --top_n_words $w | \
        grep "RESULT:" | \
        awk -F "RESULT: " '{print $2}' >> sweep_results.csv) &
        
    done
done

wait

echo "Sweep Complete. Results saved to sweep_results.csv"
cat sweep_results.csv