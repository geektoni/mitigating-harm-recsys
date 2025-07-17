#!/bin/bash
python ranker_kuairand.py --runs 10 --score-model sigformer --score-type harm --method hybrid --cores 8 --beta 0.0 --epoch 100 
python ranker_kuairand.py --runs 10 --score-model gformer --score-type harm --method hybrid --cores 8 --beta 0.0 --epoch 100 
python ranker_kuairand.py --runs 10 --score-model siren --score-type harm --method hybrid --cores 8 --beta 0.0 --epoch 100 
python ranker_kuairand.py --runs 10 --score-model lightgcl --score-type harm --method hybrid --cores 8 --beta 0.0 --epoch 100 

python ranker_kuairand.py --runs 10 --score-model sigformer --score-type harm --method remove --cores 8 --beta 0.0 --epoch 100 
python ranker_kuairand.py --runs 10 --score-model gformer --score-type harm --method remove --cores 8 --beta 0.0 --epoch 100 
python ranker_kuairand.py --runs 10 --score-model siren --score-type harm --method remove --cores 8 --beta 0.0 --epoch 100 
python ranker_kuairand.py --runs 10 --score-model lightgcl --score-type harm --method remove --cores 8 --beta 0.0 --epoch 100 