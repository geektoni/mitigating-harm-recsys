from LightGCL.model import LightGCL
from LightGCL.dataloader import LightGCLDataLoader
from SiReN.dataloader import SiReNDataset
from SiReN.model import SiReN
from SIGformer.dataloader import MyDataset
from SIGformer.model import SIGformer
from GFormer.model import GFormer, GTLayer
from GFormer.dataloader import GFormerDataLoader

from NCF.model import RankerNN
from NCF.dataloader import KuaiHarmDataset, train_model, obtain_prediction

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from datasets import load_movielens_1m

import os
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split

import sys
import zlib
import pickle

from argparse import ArgumentParser

# Fix seeds to ensure reproducibility
random.seed(2025)
np.random.seed(2025)
torch.manual_seed(2025)

# Only for movielens
BASE_HARM = {
    0.3:0.35,
    0.2:0.5,
    0.1:0.7,
    0.05:0.99
}

# Extract only top users
def extract_only_top_users(data, flag_treshold=2.5):
    user_flag_ratio = data.groupby('user_id').agg(
    total_videos=('video_id', 'count'),
    flagged_videos=('is_hate', 'sum')
    )

    # Calculate the flagged ratio
    user_flag_ratio['flag_ratio'] = (user_flag_ratio['flagged_videos'] / user_flag_ratio['total_videos'])*100

    # Get the top 50 users by decreasing flagged ratio
    top_50_users = user_flag_ratio.sort_values('flag_ratio', ascending=False).head(100).reset_index()
    top_50_users = user_flag_ratio[user_flag_ratio.flag_ratio > flag_treshold].reset_index()

    print(f"Total Users for W_% > {flag_treshold}: ", len(top_50_users.user_id.unique()))

    only_top_50_users = data[data.user_id.isin(top_50_users.user_id.values)].copy()
    only_top_50_users['user_id'], mapping_user_id = pd.factorize(only_top_50_users['user_id'])
    only_top_50_users['video_id'], mapping_video_id = pd.factorize(only_top_50_users['video_id'])

    return only_top_50_users

parser = ArgumentParser()
parser.add_argument("--model", type=str, choices=[  "ncf",
                                                     "ncfharm",
                                                    "lightgcl",
                                                    "sigformer",
                                                    "gformer",
                                                    "siren"], default="ncf")
parser.add_argument("--base-harm", type=float, choices=[0.3, 0.2, 0.1, 0.05], default=0.3,
                    help="Base harmfulness (it works only for the movielens example)")
parser.add_argument("--dataset", type=str, choices=["kuairand", "movielens"], default="kuairand", help="Dataset we want to train our models on.")
parser.add_argument("--batch-size", type=int, default=1024, help="Size of the batch size")
parser.add_argument("--epochs", type=int, default=10, help="How many training epochs do we consider")
parser.add_argument("--runs", type=int, default=1, help="How many runs do we need to generate the data for")
parser.add_argument("--gpu", type=int, default=0, help="Index of the GPU we intent to use.")
parser.add_argument("--data-path", type=str, default="../KuaiRand-Harm/training/single_and_repeated_interactions_is_click.csv.gzip")
parser.add_argument("--build-only-dataset", default=False, action="store_true", help="Generate only the dataset split")
parser.add_argument("--filter-hate", default=False, action="store_true", help="Filter the data by hate")

# Parse the arguments
args = parser.parse_args()

# Model we want to test
model_name = args.model

# How many training epochs do we have
EPOCHS= args.epochs

# Get the device we will be training on (e.g., CPU or GPU)
CPU_ID = args.gpu
device = torch.device(f'cuda:{CPU_ID:d}' if torch.cuda.is_available() else 'cpu')

# Read the original data & convert it in the suitable format
if args.dataset == "kuairand":
    original_data = pd.read_csv(args.data_path, compression="gzip")
    if args.filter_hate:
        parsed_data = extract_only_top_users(original_data)[["user_id", "video_id", "is_hate", "fraction_play_time"]]
    else:
        parsed_data = original_data[["user_id", "video_id", "is_hate", "fraction_play_time", "is_hate_y", "fraction_play_time_y"]]
    print("[*] Finished parsing the data")
else:
    parsed_data = load_movielens_1m(
        alpha=BASE_HARM.get(args.base_harm)
        )[["user_id", "video_id", "is_hate", "fraction_play_time", "is_hate_y", "fraction_play_time_y"]]

for run_id in range(args.runs):

    # File where we are going to store the results
    run_train_path = f"./{args.dataset}/training/train_{run_id}_{args.filter_hate}_{args.base_harm}.txt"
    run_temp_path = f"./{args.dataset}/training/test_calibration_{run_id}_{args.filter_hate}_{args.base_harm}.txt"

    if os.path.exists(run_train_path) and os.path.exists(run_temp_path) and not args.build_only_dataset:
        print(f"Data for run {run_id} already exists. Skipping split.")
        train_data, temp_data = train_test_split(parsed_data, test_size=0.3, stratify=parsed_data["is_hate"], random_state=run_id)
    else:
        train_data, temp_data = train_test_split(parsed_data, test_size=0.3, stratify=parsed_data["is_hate"], random_state=run_id)

        print("Training data: ", len(train_data))
        print("Test+Calibration data: ", len(temp_data))

        # Save these data to disk
        train_data.to_csv(run_train_path, index=None, sep=' ', header=None)
        temp_data.to_csv(run_temp_path, index=None, sep=' ', header=None)
    
    # Skip the rest if we need only the training dataset
    if args.build_only_dataset:
        continue

    # Build also the test loader (and run the eval over all data for consistency)
    test_loader = DataLoader(
            KuaiHarmDataset(temp_data, None),
            batch_size=args.batch_size, shuffle=False)
    
    if model_name == "ncf" or model_name == "ncfharm":
        
        # Read and avoid splitting the various stuff
        train_data = pd.read_table(run_train_path, header=None, sep=' ',
                                  names=["user_id", "video_id", "is_hate", "fraction_play_time", "NONE", "NONE2"])[["user_id", "video_id", "is_hate", "fraction_play_time"]]
        train_dataset = KuaiHarmDataset(train_data, None)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True) #8192

        # Train the ranker
        if model_name == "ncf":
            model = RankerNN(parsed_data.user_id.max(), parsed_data.video_id.max(),
                            num_genres=1,
                            max_output_value=train_data.fraction_play_time.max() if args.dataset == "kuairand" else 5,
                            min_output_value=0 if args.dataset == "kuairand" else 1)
        else:
            model = RankerNN(parsed_data.user_id.max(), parsed_data.video_id.max(),
                            num_genres=1,
                            max_output_value=1,
                            min_output_value=0)
        criterion = nn.MSELoss() if model_name == "ncf" else nn.BCEWithLogitsLoss(
            pos_weight = torch.FloatTensor([(train_data.is_hate==0.).sum()/train_data.is_hate.sum()])
        )
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        train_model(model, train_loader,
                    criterion, optimizer, epochs=EPOCHS, verbose=True,
                    device=device,
                    validation_loader=test_loader,
                    use_feedback=model_name == "ncfharm",
                    validation_filename=f"./{args.dataset}/results/test_{run_id}_{model_name}_{args.filter_hate}_{args.base_harm}")

    elif model_name == "lightgcl":

        dataset = LightGCLDataLoader(
            run_train_path,
            run_temp_path,
            device=device,
            batch_size=args.batch_size # 1024
        )

        model = LightGCL(
            dataset.adj_norm.shape[0],
            dataset.adj_norm.shape[1],
            64,
            dataset.u_mul_s,
            dataset.v_mul_s,
            dataset.svd_u.T,
            dataset.svd_v.T,
            dataset.train_csr,
            dataset.adj_norm,
            l=2,
            temp=0.2,
            lambda_1=0.2,
            lambda_2=1e-7,
            dropout=0.0,
            batch_user=args.batch_size,
            device=device)
        model.to(device)
        
        # Rename since we need the dataloader
        dataset = dataset.train_loader

    elif model_name == "sigformer":

        # Build the dataset for SIGformer
        dataset = MyDataset(
            #"../ignore/raw/SIGformer/data/KuaiRec/train.txt",
            #"../ignore/raw/SIGformer/data/KuaiRec/test.txt",
            #"../ignore/raw/SIGformer/data/KuaiRec/test.txt",
            run_train_path,
            run_temp_path,
            run_temp_path,
            device,
            eigs_dim=8
        )

        # Build the model
        model = SIGformer(
            dataset,
            dataset.num_users,
            dataset.num_items,
            hidden_dim=8,
            n_layers=3,
            learning_rate=1e-2,
            device=device
        )

        test_loader = DataLoader(
            KuaiHarmDataset(temp_data, None),
            batch_size=100000, shuffle=False)

    elif model_name == "gformer":

        dataset = GFormerDataLoader(
            #"../ignore/raw/SIGformer/data/KuaiRec/train.txt",
            #"../ignore/raw/SIGformer/data/KuaiRec/test.txt",
            #"../ignore/raw/SIGformer/data/KuaiRec/test.txt",
            run_train_path,
            run_temp_path,
            device=device,
            batch_size=args.batch_size #256
        )

        gtLayer = GTLayer(device=device).to(device)
        model = GFormer(gtLayer,
                        dataset.num_users,
                        dataset.num_items,
                        dataset=dataset,
                        device=device).to(device)
        
        # ret_train = pd.read_table(
        #     "../ignore/raw/SIGformer/data/KuaiRec/test.txt", header=None, sep=' ',
        #     names=["user_id", "video_id", "is_hate"]
        # )
        # ret_train["fraction_play_time"] = 0
        # test_loader = DataLoader(
        #     KuaiHarmDataset(ret_train, None),
        #     batch_size=1024, shuffle=False)

    else:

        dataset = SiReNDataset(
            run_train_path,
            run_temp_path,
            offset=1, K=40,
            device=device
        )
        model = SiReN(
            dataset.train_data, dataset.num_users, dataset.num_items,
            device=device, batch_size=args.batch_size #2048
        )
        test_user_ids = torch.tensor(dataset.test_data[0].unique())

    # Train the model on the training data
    # (only if we consider the sign-aware scores)
    if model_name != "ncf" and model_name != "ncfharm":
        for epoch in range(1, EPOCHS+1):
            train_loss = model.train_func(dataset, epoch)
            if epoch % 1 == 0: # Print diagnostic info 
                print(f'[*] Epoch {epoch:d}, Train_loss = {train_loss:f}')

            # Every 10 epochs save to disk the results
            # This way, we can evaluate how the model improves by increasing the
            # number of iterations. 
            if epoch % 10 == 0:
                pred_ratings, _, _, user_ids_safe, item_ids_safe, _ = obtain_prediction(
                        model, test_loader, device=device, verbose=True
                )

                ratings = {}
                for pred, user_id, item_id in zip(pred_ratings, user_ids_safe, item_ids_safe):
                    if user_id not in ratings:
                        ratings[user_id] = {}
                    if item_id not in ratings[user_id]:
                        ratings[user_id][item_id] = []
                    ratings[user_id][item_id] = pred
                    
                ram_size = sys.getsizeof(ratings)
                compressed_data = zlib.compress(pickle.dumps(ratings))
                with open(f"./{args.dataset}/results/test_{run_id}_{model_name}_{args.filter_hate}_{args.base_harm}_{epoch}.zlib.pickle", 'wb') as f:
                    f.write(compressed_data)

                disk_size = len(compressed_data)

                print(f"RAM usage: ~{ram_size / 1024:.2f} KB")
                print(f"Estimated disk size: ~{disk_size / 1024:.2f} KB")
