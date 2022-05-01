#!/bin/bash

dataset=$1
sub_dataset=${2:-''}

hidden_channels_lst=()

if [ "$dataset" = "snap-patents" ] || [ "$dataset" = "pokec" ]; then 
    hidden_channels_lst=(4 8 16 32)
else 
    hidden_channels_lst=(4 8 16 32 64)
fi 

lr_lst=(0.1 0.01 0.001)
jk_type_lst=('max' 'cat')

for jk_type in "${jk_type_lst[@]}"; do
for hidden_channels in "${hidden_channels_lst[@]}"; do
    for lr in "${lr_lst[@]}"; do
        if [ "$dataset" = "snap-patents" ] || [ "$dataset" = "arxiv-year" ]; then
            python main.py --dataset $dataset --sub_dataset ${sub_dataset:-''} \
            --method gcnjk --num_layers 2 --hidden_channels $hidden_channels \
            --lr $lr  --display_step 25 --runs 5 --jk_type $jk_type  --directed
        else
            python main.py --dataset $dataset --sub_dataset ${sub_dataset:-''} \
            --method gcnjk --num_layers 2 --hidden_channels $hidden_channels \
            --lr $lr  --display_step 25 --runs 5  --jk_type $jk_type
        fi
done            
done
done
