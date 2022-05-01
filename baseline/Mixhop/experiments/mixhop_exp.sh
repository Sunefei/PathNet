#!/bin/bash

dataset=$1
sub_dataset=${2:-''}

hidden_channels_lst=(8 16 32)

num_layers_lst=(2 3)


for hidden_channels in "${hidden_channels_lst[@]}"; do
    for num_layers in "${num_layers_lst[@]}"; do
		if [ "$dataset" = "snap-patents" ] || [ "$dataset" = "arxiv-year" ]; then
			python main.py --dataset $dataset --sub_dataset ${sub_dataset:-''} \
				--method mixhop --num_layers $num_layers --hidden_channels $hidden_channels \
				--display_step 25 --runs 5 --hops 2 --directed
		else
			python main.py --dataset $dataset --sub_dataset ${sub_dataset:-''} \
				--method mixhop --num_layers $num_layers --hidden_channels $hidden_channels \
				--display_step 25 --runs 5 --hops 2
		fi
    done
done
