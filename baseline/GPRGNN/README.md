# GPRGNN

This is the source code for our ICLR2021 paper: [Adaptive Universal Generalized PageRank Graph Neural Network](https://openreview.net/forum?id=n6jl7fLxrP).


<p align="center">
  <img src="https://github.com/jianhao2016/GPRGNN/blob/master/figs/workflow.png" width="600">
</p>

Hidden state feature extraction is performed by a neural networks using individual node features propagated via GPR. Note that both the GPR weights <img src="https://render.githubusercontent.com/render/math?math=\gamma_k"> and parameter set <img src="https://render.githubusercontent.com/render/math?math=\{\theta\}"> of the neural network are learned simultaneously in an end-to-end fashion (as indicated in red).


The learnt GPR weights of the GPR-GNN on real world datasets. Cora is homophilic while Texas is heterophilic (Here, H stands for the level of homophily defined in the main text, Equation (1)). An interesting trend may be observed: For the heterophilic case the weights alternate from positive to negative with dampening amplitudes. The shaded region corresponds to a 95% confidence interval.


<p align="center">
  <img src="https://github.com/jianhao2016/GPRGNN/blob/master/figs/Different_gamma_upated_H.png" width="600">
</p>

# Requirement:
```
pytorch
pytorch-geometric
numpy
```

# Run experiment with Cora:

go to folder `src`
```
python train_model.py --RPMAX 2 \
        --net GPRGNN \
        --train_rate 0.025 \
        --val_rate 0.025 \
        --dataset cora 
```

# Create cSBM dataset:
go to folder `src`
```
source create_cSBM_dataset.sh
```
 The total size of cSBM datasets we used is over 1GB hence they are not included in this repository, but we do have a sample of the dataset in `data/cSBM_demo`. We reccommend you to regenerate these datasets using the format of above script, start its name with `'cSBM_data'` and change the parameter <img src="https://render.githubusercontent.com/render/math?math=\phi"> to what we choose in section A.10 in Appendix of our paper.

# Repreduce results in Table 2:

To reproduce the results in Table 2 of [our paper](https://openreview.net/forum?id=n6jl7fLxrP) you need to first perform hyperparameter tuning. 
For details of optimization of all models, please refer to section A.9 in Appendix of our paper. Here are the settings for GPRGNN and APPNP:

We choose random walk path lengths with K = 10 and use a 2-layer (MLP) with 64 hidden units for the NN component. For the GPR weights, we use different initializations including PPR with <img src="https://render.githubusercontent.com/render/math?math=\alpha\in\{0.1, 0.2, 0.5, 0.9\}">, <img src="https://render.githubusercontent.com/render/math?math=\gamma_k=\delta_{0k}"> or <img src="https://render.githubusercontent.com/render/math?math=\delta_{Kk}"> and the default random initialization in pytorch. Similarly, for APPNP we search the optimal <img src="https://render.githubusercontent.com/render/math?math=\alpha\in\{0.1, 0.2, 0.5, 0.9\}">. For other hyperparameter tuning, we optimize the learning rate over {0.002, 0.01, 0.05} and weight decay {0.0, 0.0005} for all models. 

<!-- <img src="https://render.githubusercontent.com/render/math?math=\alpha\in\{0.1, 0.2, 0.5, 0.9\}">
<img src="https://render.githubusercontent.com/render/math?math=\gamma_k=\delta_{0k}">
<img src="https://render.githubusercontent.com/render/math?math=\delta_{Kk}"> -->


Here is a list of hyperparameters for your reference:

- For cora and citeseer, choosing different alpha doesn't make big difference. So you can choose alpha = 0.1.
- For pubmed, we choose lr = 0.05, alpha = 0.2, wd = 0.0005 and add dprate = 0.5 (dropout for GPR part).
- For computers, we choose lr = 0.05, alpha = 0.5 and wd = 0.
- For Photo, we choose lr = 0.01, alpha = 0.5 and wd = 0.
- For chameleon, we choose lr = 0.05, alpha = 1, wd = 0 and dprate = 0.7.
- For Actor, we choose lr = 0.01, alpha = 0.9, wd = 0.
- For squirrel, we choose lr = 0.05, alpha = 0, wd = 0, dprate = 0.7.
- For Texas, we choose lr = 0.05, alpha = 1, wd = 0.0005.
- For Cornell, we choose lr = 0.05, alpha = 0.9, wd = 0.0005.

# Citation
Please cite our paper if you use this code in your own work:
```latex
@inproceedings{
chien2021adaptive,
title={Adaptive Universal Generalized PageRank Graph Neural Network},
author={Eli Chien and Jianhao Peng and Pan Li and Olgica Milenkovic},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=n6jl7fLxrP}
}
```

Feel free to email us(jianhao2@illinois.edu, ichien3@illinois.edu) if you have any further questions. 



