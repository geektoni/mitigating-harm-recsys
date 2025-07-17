import concurrent.futures
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from calibration import calibrate_gamma
from datasets import load_movielens_100k
from utils import MovieLensDataset, RankerNN, train_model, process_user, process_user_only_model, obtain_prediction

from score_functions import SimilarityScoreMethod, HarmScoreMethod, NaiveScoreMethod

from tqdm import tqdm
from argparse import ArgumentParser

import pandas as pd

# Main Execution
if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--score-type", type=str, choices=["subtraction", "weights", "similarity"], default="subtraction")
    parser.add_argument("--method", type=str, choices=["replace", "remove"], default="remove")
    args = parser.parse_args()

    # Fix for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    for alpha in [0.35, 0.95, 0.99]:

        df = load_movielens_100k(alpha=alpha)
        num_users, num_items, MAX_SCORE = df['user_id'].max(), df['item_id'].max(), df['rating'].max()

        # Set which will contain all results
        full_evaluation_results = []

        for run_id in range(args.runs):
        
            train_data, temp_data = train_test_split(df, test_size=0.3, stratify=df["rating"], random_state=run_id)
            safe_data, temp_data = train_test_split(temp_data, test_size=0.66664, stratify=temp_data["rating"], random_state=run_id)
            test_data, calibration_data = train_test_split(temp_data, test_size=0.5, stratify=temp_data["rating"], random_state=run_id)
            del temp_data

            # print("[*] Training size: ", len(train_data))
            # print("[*] Safe size: ", len(safe_data))
            # print("[*] Test size: ", len(test_data))
            # print("[*] Calibration size: ", len(calibration_data))

            # The safe data are videos the user has seen previously
            # such that their binary_feedback is zero (meaning the users did not flag them).
            safe_data = train_data.copy()
            safe_data = safe_data[safe_data.binary_feedback == 0]

            train_dataset = MovieLensDataset(train_data)
            calibration_dataset = MovieLensDataset(calibration_data)
            safe_dataset = MovieLensDataset(safe_data)
            
            train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
            
            # Train the ranker
            model = RankerNN(num_users, num_items)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            train_model(model, train_loader, criterion, optimizer, epochs=args.epochs, use_gpu=False, verbose=False)
            
            # Given the training data, we compute the "global" score (e.g., the fraction of times a video was deemed harmful)
            tmp = train_data.groupby("item_id")["binary_feedback"].mean().reset_index()
            global_scores_items  = dict(zip(tmp["item_id"], tmp["binary_feedback"]))

            # Evaluate for all potential alphas
            for k in [20]:

                # Consider only users with a number of items greater than k
                # both for safe, calibration and testing
                filtered_data = test_data[(test_data.groupby("user_id")["user_id"].transform("count") > k)]
                filtered_calibration_data = calibration_data[(calibration_data.groupby("user_id")["user_id"].transform("count") > k)]
                filtered_safe_data = safe_data[(safe_data.groupby("user_id")["user_id"].transform("count") > k)]

                # Filter the safe, test and calibration data to have only users which are present in all three.
                # It ensures the calibration guarantees (otherwise we have distribution shift)
                # Step 2: Keep only users present in all three datasets
                user_ids_available_in_all = set(filtered_data.user_id.unique()) & \
                                            set(filtered_calibration_data.user_id.unique()) & \
                                            set(filtered_safe_data.user_id.unique())

                filtered_data = filtered_data[filtered_data.user_id.isin(user_ids_available_in_all)]
                filtered_calibration_data = filtered_calibration_data[filtered_calibration_data.user_id.isin(user_ids_available_in_all)]
                filtered_safe_data = filtered_safe_data[filtered_safe_data.user_id.isin(user_ids_available_in_all)]

                # Step 3: Compute item counts per user in filtered_data and filtered_calibration_data
                user_item_counts_data = filtered_data.groupby("user_id")["user_id"].count()
                user_item_counts_calibration = filtered_calibration_data.groupby("user_id")["user_id"].count()

                # Step 4: Compute item counts in filtered_safe_data
                user_item_counts_safe = filtered_safe_data.groupby("user_id")["user_id"].count()

                # Step 5: Keep only users where filtered_safe_data has at least as many items as in the other datasets
                valid_users = user_item_counts_safe[
                    (user_item_counts_safe >= user_item_counts_data) & 
                    (user_item_counts_safe >= user_item_counts_calibration)
                ].index

                # Step 6: Apply this user filtering
                filtered_data = filtered_data[filtered_data.user_id.isin(valid_users)]
                filtered_calibration_data = filtered_calibration_data[filtered_calibration_data.user_id.isin(valid_users)]
                filtered_safe_data = filtered_safe_data[filtered_safe_data.user_id.isin(valid_users)]

                # For the user both in calibration and testing, we extract their marked videos in the training test
                feedback_data_from_training = train_data[train_data.user_id.isin(user_ids_available_in_all)]

                # Build the test and data loaders
                calibration_loader = DataLoader(
                    MovieLensDataset(filtered_calibration_data),
                    batch_size=128, shuffle=False
                )
                test_loader = DataLoader(
                    MovieLensDataset(filtered_data),
                    batch_size=128, shuffle=False)
                safe_loader = DataLoader(
                    MovieLensDataset(filtered_safe_data),
                    batch_size=128, shuffle=False
                )
                _, _, _, user_ids_calibrations, item_ids_calibration, _ = obtain_prediction(
                    model, calibration_loader
                )

                # Obtain evaluations for the safe items for each user
                pred_ratings_safe, true_ratings_safe, _, user_ids_safe, item_ids_safe, _ = obtain_prediction(
                    model, safe_loader
                )

                # Compute the harmfulness of the model on the full data for top-K
                pred_ratings, true_ratings, feedbacks, user_ids, item_ids, _ = obtain_prediction(
                    model, test_loader
                )

                # Split IDs for each CPU
                num_workers = min(args.cores, os.cpu_count())
                user_id_chunks = np.array_split(np.unique(user_ids), 
                                                num_workers
                                                )
                
                results_model_filtered = []
                with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                    futures = list(
                        executor.submit(
                            process_user_only_model, chunk,
                            user_ids,
                            item_ids,
                            pred_ratings,
                            true_ratings,
                            feedbacks,
                            k) for chunk in user_id_chunks)

                    for future in futures:
                        results_model_filtered.extend(future.result())
                
                size_of_recommendation, loss, ndcg_no_gamma, recall_no_gamma = zip(*results_model_filtered)
                size_of_recommendation_model = np.mean(size_of_recommendation)
                loss_model = np.mean(loss)
                ndcg_model = np.mean(ndcg_no_gamma)
                recall_model = np.mean(recall_no_gamma)
                del size_of_recommendation, loss, ndcg_no_gamma, recall_no_gamma # Remove variables to avoid using them again

                # Append for each alpha the model evaluation
                full_evaluation_results.append(
                    [run_id, alpha, loss_model]
                )

        df = pd.DataFrame(
            full_evaluation_results,
            columns= ["run_id", "alpha", "harm"]
        )
        print(alpha, np.mean(df.harm.values))
