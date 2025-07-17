#!/bin/bash

python ranker_kuairand.py --runs 10 --score-model lightgcl --score-type harm --method hybrid --cores 8 --epoch 100 
python ranker_kuairand.py --runs 10 --score-model lightgcl --score-type harm --method hybrid --cores 8 --beta 0.00 --epoch 100 
python ranker_kuairand.py --runs 10 --score-model lightgcl --score-type harm --method hybrid --cores 8 --beta 0.5 --epoch 100 
python ranker_kuairand.py --runs 10 --score-model lightgcl --score-type harm --method hybrid --cores 8 --beta 1 --epoch 100 
