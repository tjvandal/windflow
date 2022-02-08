# WindFlow: Dense feature tracking of atmospheric winds with deep optical flow

(Manuscript in submission)

## Setup

```
conda env create -f environment.yml
pip install spatial-correlation-sampler
```

## Download data

G5NR data used in the study is a total of 7.5 TBs containing specific humidity (QV), u-direction (U), and v-direction (V). <br>
https://gmao.gsfc.nasa.gov/global_mesoscale/7km-G5NR/data_access/ <br>
https://portal.nccs.nasa.gov/datashare/gmao_obsteam/osse_for_wisc/

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

## Perform Inference on G5NR Test Set

On a single V100 GPU, inference applied to 1 files takes ~75 seconds.

```
cd pipelines/g5nr/
CUDA_VISIBLE_DEVICES=0 python pipelines/g5nr/g5nr_to_flows.py --model_name raft
```

## Transfer to GOES-16 ABI

### GOES-16/17 data can be accessed via AWS public cloud 

Data access found here: https://registry.opendata.aws/noaa-goes/ 

```
python pipelines/geo/geo_flows_to_zarr.py --checkpoint models/raft-size_512/checkpoint.pth.tar
```

## Acknowledgements 

External packages and flownet code was used from: https://github.com/celynw/flownet2-pytorch/

