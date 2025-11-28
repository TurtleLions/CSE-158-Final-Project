#!/bin/bash

# Define Grid
TAG_COUNTS=(10 25 50 100)
WORD_COUNTS=(0 50 100 250)
C_VALUES=(0.1 1.0 10.0)

# Limit concurrent jobs
MAX_JOBS=10

echo "Starting Grid Sweep..."
echo "Tags,Words,C,AUC" > sweep_results.csv

# Ensure Python doesn't hog threads
export OMP_NUM_THREADS=1

for t in "${TAG_COUNTS[@]}"; do
    for w in "${WORD_COUNTS[@]}"; do
        for c in "${C_VALUES[@]}"; do
            
            while [ $(jobs -r | wc -l) -ge $MAX_JOBS ]; do
                sleep 1
            done

            echo "Launching: Tags=$t, Words=$w, C=$c"
            
            (
                python grid_search_model.py --top_n_tags $t --top_n_words $w --reg_C $c | \
                grep "DATA_ROW:" | \
                sed 's/DATA_ROW://' >> sweep_results.csv
            ) &

        done
    done
done

# Wait for remaining jobs
wait

echo "Sweep Complete. Check sweep_results.csv"
cat sweep_results.csv