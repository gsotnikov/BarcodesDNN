# Topological obstructions in neural networks’ learning

This repository contains implemetation of the outlined papaer.

1) Train N models: `python find_minimas.py --type=1 --dataset=SVHN --N=8 --device=cuda`

`--type` - Type 1 (minimas obtained with small constant learning rate), Type 2 (minimas obtained with scheduling of a learning rate)

`--dataset` - Supposed options, which were used in the paper: [SVHN, CIFAR10]

`--N` - Number of independent models to train

`--device` - CUDA/CPU

2) When step `1)` is executed, path between trained minimas can be found by execution of a `ComputeBarcode.ipynb`.

3) To plot figures from the paper, please use `Report.ipynb`.