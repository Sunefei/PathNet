#!/bin/bash

dataset=$1
sub_dataset=${2:-''}

hidden_channels_lst=(16 32 64 128 256)
lr_lst=(0.01 0.05 0.002)
alpha_lst=(0.1 0.2 0.5 0.9)

for hidden_channels in "${hidden_channels_lst[@]}"; do
    for lr in "${lr_lst[@]}"; do
        for alpha in "${alpha_lst[@]}"; do
			if [ "$dataset" = "snap-patents" ] || [ "$dataset" = "arxiv-year" ]; then
            	python main.py --dataset $dataset --sub_dataset ${sub_dataset:-''} --method gprgnn --lr $lr --hidden_channels $hidden_channels --gpr_alpha $alpha --display_step 25 --runs 5 --directed
			else
            	python main.py --dataset $dataset --sub_dataset ${sub_dataset:-''} --method gprgnn --lr $lr --hidden_channels $hidden_channels --gpr_alpha $alpha --display_step 25 --runs 5
			fi
        done
    done
done
