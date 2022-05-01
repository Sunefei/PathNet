#!/bin/bash

dataset=$1
sub_dataset=${2:-''}
hops=${3:-1}

weight_decay_lst=(.001 .01 .1)

for weight_decay in "${weight_decay_lst[@]}"; do
	if [ "$dataset" = "snap-patents" ] || [ "$dataset" = "arxiv-year" ]; then
        python main.py --dataset $dataset \
        --method sgc  --sub_dataset ${sub_dataset:-''} --hops $hops \
        --display_step 25 --runs 5  --directed
	else
		python main.py --dataset $dataset \
        --method sgc  --sub_dataset ${sub_dataset:-''} --hops $hops \
        --display_step 25 --runs 5 
	fi
done
