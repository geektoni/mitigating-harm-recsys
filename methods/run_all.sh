#!/bin/bash

python train_rec.py --model "sigformer" --epochs 100 --runs 10 --gpu 1
python train_rec.py --model "siren" --epochs 100 --runs 10 --gpu 1 --batch-size 4096
python train_rec.py --model "ncf" --epochs 100 --runs 10 --gpu 1 --batch-size 8192
python train_rec.py --model "ncfharm" --epochs 100 --runs 10 --gpu 1 --batch-size 8192
python train_rec.py --model "lightgcl" --epochs 100 --runs 10 --gpu 1 --batch-size 4096
python train_rec.py --model "gformer" --epochs 100 --runs 10 --gpu 1 --batch-size 4096