# You Don’t Bring Me Flowers: Mitigating Unwanted Recommendations Through Conformal Risk Control

This repository contains all the data and scripts to replicate the experiments and figures for our paper, accepted to RecSys 2025.

🏆 This paper was [awarded](https://recsys.acm.org/best-papers/) the **Best Full Paper Award at RecSys 2025**.

---

## Table of Contents

- [Install](#install)
- [Replicate our results](#replicate-our-results)
  - [Train the rankers](#train-the-rankers)
  - [Run again all the evaluations](#run-again-all-the-evaluations)
- [Data Analysis and Preprocessing](#data-analysis-and-preprocessing)
- [Cite us](#cite-us)
- [Authors](#authors)

---

## Install

```bash
conda create --name harm-recsys python=3.10
conda activate harm-recsys
pip install -r requirement.txt
````

## Replicate our results

We directly provide the results of the various experiments and ablations (Section 7) in the directory `results`. You can use the following scripts to re-generate the plots within the paper.

```bash
conda activate harm-recsys

python analytics/plot_figure_3_and_4.py results/01_results_remove_replace/
python analytics/plot_figure_5.py results/02_ablation_beta/
python analytics/plot_figure_6.py results/03_results_hard_easy_users/
```

If you want to replicate in full our experiments, then please do the following:

### Train the rankers

The filtered dataset are already available in `KuaiRand-Harm/training`. All the trained rankers will be saved in `methods/kuairand/results`, while the data splitting will be saved in `methods/kuairand/training` for reproducibility.

You can adjust the batch size depending on the size of your GPU (see `methods/run_all.sh`).

```bash
conda activate harm-recsys
cd methods
python train_rec.py --model "ncf" --epochs 100 --runs 10 --gpu 1 --batch-size 8192
python train_rec.py --model "sigformer" --epochs 100 --runs 10 --gpu 1
python train_rec.py --model "siren" --epochs 100 --runs 10 --gpu 1 --batch-size 2048
python train_rec.py --model "gformer" --epochs 100 --runs 10 --gpu 1 --batch-size 256
python train_rec.py --model "lightgcl" --epochs 100 --runs 10 --gpu 1 --batch-size 2014
```

### Run again all the evaluations

To run all the evaluations with the same configuration we used within our experiments, please use the following scripts:

```bash
conda activate harm-recsys
export PYTHONPATH=.
bash experiments/run_experiment_replace_remove.sh
bash experiments/run_ablation_beta.sh
bash experiments/run_ablation_hard_users.sh
```

## Data Analysis and Preprocessing

The notebook used for the data analysis (Section 3) can be found in `data_analysis`.
Before running it, please follow these instructions to obtain the full Kuaishou dataset:

```bash
wget https://zenodo.org/records/10439422/files/KuaiRand-27K.tar.gz
tar -xzvf KuaiRand-27K.tar.gz
mkdir KuaiRand-Harm/data
mv KuaiRand-27K/data/*.csv KuaiRand-Harm/data
```

Make sure to have enough space on your local machine.
The data analysis notebook is also quite RAM intensive.

You can find the notebook used to create the experimental data in
`data_analysis/generate_dataset_for_training.ipynb`.
You do **not** need to run it again, since the experimental data are already available in the repository.

## Cite us

If you reuse this code, please cite our paper:

```bibtex
@inproceedings{de2025you,
  title={You Don't Bring Me Flowers: Mitigating Unwanted Recommendations Through Conformal Risk Control},
  author={De Toni, Giovanni and Purificato, Erasmo and Gomez, Emilia and Passerini, Andrea and Lepri, Bruno and Consonni, Cristian},
  booktitle={Proceedings of the Nineteenth ACM Conference on Recommender Systems},
  pages={492--502},
  year={2025}
}
```

## Authors

- Giovanni De Toni, Fondazione Bruno Kessler (FBK), Trento, Italy, [gdetoni@fbk.eu](mailto:gdetoni@fbk.eu)
- Erasmo Purificato, European Commission, Joint Research Centre (JRC), Ispra, Italy, [erasmo.purificato@acm.org](mailto:erasmo.purificato@acm.org)
- Emilia Gomez, European Commission, Joint Research Centre (JRC), Seville, Spain, [emilia.gomezgutierrez@ec.europa.eu](mailto:emilia.gomezgutierrez@ec.europa.eu)
- Andrea Passerini, University of Trento, Trento, Italy, [andrea.passerini@unitn.it](mailto:andrea.passerini@unitn.it)
- Bruno Lepri, Fondazione Bruno Kessler (FBK), Trento, Italy, [lepri@fbk.eu](mailto:lepri@fbk.eu)
- Cristian Consonni, European Commission, Joint Research Centre (JRC), Ispra, Italy, [cristian.consonni@acm.org](mailto:cristian.consonni@acm.org)
