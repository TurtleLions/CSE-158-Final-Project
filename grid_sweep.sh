#!/bin/bash

TAG_COUNTS=(10 25 50 100)

WORD_COUNTS=(0 50 100 250)

C_VALUES=(0.1 1.0 10.0)

MAX_JOBS=10

echo "Starting Grid Sweep with max $MAX_JOBS concurrent jobs..."

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Initialize results file
if [ ! -f sweep_results.csv ]; then
    echo "Tags, Words, C, AUC" > sweep_results.csv
fi

# Generate the list of commands to run
COMMANDS_FILE="commands_list.txt"
> "$COMMANDS_FILE"

for t in "${TAG_COUNTS[@]}"; do
    for w in "${WORD_COUNTS[@]}"; do
        for c in "${C_VALUES[@]}"; do
            CMD="python grid_search_model.py --top_n_tags $t --top_n_words $w --reg_C $c | grep 'RESULT:' | awk -F 'RESULT: ' '{print \$2}' >> sweep_results.csv"
            echo "$CMD" >> "$COMMANDS_FILE"
        done
    done
done

TOTAL_JOBS=$(wc -l < "$COMMANDS_FILE")
echo "Generated $TOTAL_JOBS jobs."
echo "Running jobs... (Output order in CSV will be random)"

cat "$COMMANDS_FILE" | xargs -P "$MAX_JOBS" -n 1 -I {} sh -c "{}"

echo "Sweep Complete. Check sweep_results.csv for data."
rm "$COMMANDS_FILE"