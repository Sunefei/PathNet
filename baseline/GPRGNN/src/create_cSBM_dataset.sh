#! /bin/sh
#
# create_cSBM_dataset.sh

python cSBM_dataset.py --phi 0.5 \
    --name cSBM_phi_0.5 \
    --root ../data/ \
    --num_nodes 800 \
    --num_features 1000 \
    --avg_degree 5 \
    --epsilon 3.25 
