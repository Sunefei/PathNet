#!/bin/bash

CUDA=$1 

pip install torch==1.7.1+$CUDA -f https://download.pytorch.org/whl/torch_stable.html

pip install scipy==1.6.0
pip install ogb==1.2.4
pip install flake8==3.8.4
pip install Babel==2.9.0
pip install flask==1.1.2


pip install --no-index torch-scatter==2.0.5 -f https://pytorch-geometric.com/whl/torch-1.7.0+$CUDA.html
pip install --no-index torch-sparse==0.6.8 -f https://pytorch-geometric.com/whl/torch-1.7.0+$CUDA.html
pip install --no-index torch-cluster==1.5.8 -f https://pytorch-geometric.com/whl/torch-1.7.0+$CUDA.html
pip install --no-index torch-spline-conv==1.2.0 -f https://pytorch-geometric.com/whl/torch-1.7.0+$CUDA.html
pip install torch-geometric==1.6.3

