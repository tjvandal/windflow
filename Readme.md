# WindFlow: Dense feature tracking of atmospheric winds with deep optical flow

## Setup


## Download data

G5NR data used in the study is a total of 7.5 TBs containing specific humidity (QV), u-direction (U), and v-direction (V). 
https://gmao.gsfc.nasa.gov/global_mesoscale/7km-G5NR/data_access/

```
python data/download_g5nr.py data/G5NR
```

## Make training samples

```
mpirun -np 4 python windflow/datasets/g5nr/g5nr.py data/G5NR data/G5NR_patches/
```

## Train optical flow model

```
python train.py --model_path models/raft-size_512/ --dataset g5nr --data_path data/G5NR_patches --model_name raft --batch_size 2 --loss L1 --max_iterations 500000 --lr 0.00001
```

## Perform Inference


## Acknowledgements 


https://github.com/celynw/flownet2-pytorch/
