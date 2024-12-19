# Amortized Bayesian Experimental Design for Decision-Making

This repository contains the code for the paper *Amortized Bayesian Experimental Design for Decision-Making* (Huang et al., NeurIPS 2024).
The full paper can be found at [arXiv](https://arxiv.org/abs/0000). Our implementations is built based on the [TNP-pytorch](https://github.com/tung-nd/TNP-pytorch) library. 



## Installation

```
git clone https://github.com/huangdaolang/amortized-decision-aware-bed.git

cd amortized-decision-aware-bed

conda create --name tndp python=3.9

conda activate tndp

pip install -r requirements.txt
```

## Usage

We use [Hydra](https://hydra.cc/) to manage the configurations. See `configs` for all configurations and defaults.

To run toy experiments, you can run `train_toy.py`.

To run decision-aware active learning experiments, you can run `train_tal.py`. 

To run top-k HPO experiments, please use `train_topk.py`. An example case to run on Ranger dataset is as follows:
```
python train_topk.py nn=tndp_topk dataset="hpo/ranger"
```
For HPOB dataset, you can download it from `https://github.com/releaunifreiburg/HPO-B`, and put the downloaded data in `data/HPOB`.


# Citation
If you find this work useful, please cite our paper:
```
@inproceedings{
huang2024amortized,
title={Amortized Bayesian Experimental Design for Decision-Making},
author={Huang, Daolang and Guo, Yujia and Acerbi, Luigi and Kaski, Samuel},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
}
```
