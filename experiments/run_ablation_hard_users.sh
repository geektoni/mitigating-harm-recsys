#!/bin/bash

python ranker_kuairand.py --runs 10 --score-model lightgcl --score-type harm --method hybrid --cores 8 --beta 0.0 --epoch 100 --users hard
python ranker_kuairand.py --runs 10 --score-model lightgcl --score-type harm --method remove --cores 8 --beta 0.0 --epoch 100 --users hard
python ranker_kuairand.py --runs 10 --score-model lightgcl --score-type harm --method hybrid --cores 8 --beta 0.0 --epoch 100 --users easy
python ranker_kuairand.py --runs 10 --score-model lightgcl --score-type harm --method remove --cores 8 --beta 0.0 --epoch 100 --users easy