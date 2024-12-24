#!/bin/bash

for mu in $(seq 1 1 9); do
    edgefile="src/graphs/artificials/mu-0.${mu}.edgelist"

    if [ -f "$edgefile" ]; then
        for i in {1..10}; do
            cargo run --release "$edgefile"                 # get the elapsed_time,num_nodes,num_edges,modularity,
            python3 src/utils/comparisons.py "$edgefile"    # get the nmi value between leiden and louvain
        done    # both are written in the .csv file as : elapsed_time,num_nodes,num_edges,modularity,nmi_louvain,nmi_leiden
    else
        echo "Edgefile $edgefile not found!"
    fi
done
