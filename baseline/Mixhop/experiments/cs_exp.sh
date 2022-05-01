#!/bin/bash

dataset=$1
sub_dataset=${2:-''}
num_layers=${3:-3}
hidden_channels=${4:-64}

python main.py --dataset $dataset --sub_dataset ${sub_dataset:-''} --method cs --num_layers $num_layers --hidden_channels $hidden_channels --display_step 25 --runs 5

if [ "$dataset" = "snap-patents" ] || [ "$dataset" = "arxiv-year" ]; then
    python cs_tune_hparams.py --dataset $dataset --sub_dataset ${sub_dataset:-''} --trials 500 --directed
    python cs_tune_hparams.py --dataset $dataset --sub_dataset ${sub_dataset:-''} --trials 500 --cs_fixed --directed
    python cs_tune_hparams.py --dataset $dataset --sub_dataset ${sub_dataset:-''} --trials 500 --hops 2 --directed
    python cs_tune_hparams.py --dataset $dataset --sub_dataset ${sub_dataset:-''} --trials 500 --cs_fixed --hops 2 --directed
else
    python cs_tune_hparams.py --dataset $dataset --sub_dataset ${sub_dataset:-''} --trials 500
    python cs_tune_hparams.py --dataset $dataset --sub_dataset ${sub_dataset:-''} --trials 500 --cs_fixed
    python cs_tune_hparams.py --dataset $dataset --sub_dataset ${sub_dataset:-''} --trials 500 --hops 2
    python cs_tune_hparams.py --dataset $dataset --sub_dataset ${sub_dataset:-''} --trials 500 --cs_fixed --hops 2
fi
