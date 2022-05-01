#!/bin/bash

dataset=$1
sub_dataset=${2:-''}

hidden_channels_lst=()
gat_heads_lst=()

if [ "$dataset" = "snap-patents" ] || [ "$dataset" = "pokec" ]; then 
    gat_heads_lst=(2 4)
    hidden_channels_lst=(4 8 12)
else 
    gat_heads_lst=(2 4 8)
    hidden_channels_lst=(4 8 12 32)
fi 

lr_lst=(0.1 0.01 0.001)
jk_type_lst=('max' 'cat')


for jk_type in "${jk_type_lst[@]}"; do
for hidden_channels in "${hidden_channels_lst[@]}"; do
    for gat_heads in "${gat_heads_lst[@]}"; do
            for lr in "${lr_lst[@]}"; do

        if [ "$dataset" = "snap-patents" ] || [ "$dataset" = "arxiv-year" ]; then
                        python main.py --dataset $dataset --sub_dataset ${sub_dataset:-''} \
                        --method gatjk --num_layers 2 --hidden_channels $hidden_channels \
                        --lr $lr --gat_heads $gat_heads --directed --display_step 25 --runs 5 --jk_type $jk_type
        else
            python main.py --dataset $dataset --sub_dataset ${sub_dataset:-''} \
                        --method gatjk --num_layers 2 --hidden_channels $hidden_channels \
                        --lr $lr --gat_heads $gat_heads --display_step 25 --runs 5  --jk_type $jk_type
        fi
done            
done
done
done
