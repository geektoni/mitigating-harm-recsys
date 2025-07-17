import concurrent.futures
from copy import deepcopy
import torch
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split

from src.calibration import calibrate_gamma_precomputed
from src.utils import KuaiHarmDataset, process_user, process_user_only_model
from src.utils import obtain_prediction_from_precomputed
from src.utils import synthetic_classifier_predictions_random

from src.score_functions import HarmScoreMethod, NaiveScoreMethod

from tqdm import tqdm
from argparse import ArgumentParser

import pandas as pd
import pickle
import zlib

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def extract_only_top_users(data, user_type="hard"):
    
    # Aggregate user information
    user_flag_ratio = data.groupby('user_id').agg(
        total_videos=('video_id', 'count'),
        flagged_videos=('is_hate', 'sum')
    )

    # Calculate the flagged ratio
    user_flag_ratio['flag_ratio'] = (user_flag_ratio['flagged_videos'] / user_flag_ratio['total_videos'])*100

    # Get either "hard" users (1st quantile) or "easy" users (3rd quantile)
    if user_type == "hard":
        flag_treshold = np.quantile(1-user_flag_ratio.flag_ratio.values, 0.25)
        hard_users = user_flag_ratio[1-user_flag_ratio.flag_ratio <= flag_treshold].reset_index()
    elif user_type == "easy":
        flag_treshold = np.quantile(user_flag_ratio.flag_ratio.values, 0.75)
        hard_users = user_flag_ratio[user_flag_ratio.flag_ratio <= flag_treshold].reset_index()
    
    print(f"Total Users for H > {1-flag_treshold}: ", len(hard_users.user_id.unique()))

    return hard_users.user_id.unique()

# Main Execution
if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--epoch", type=int, default=10, help="Epoch we want to consider")
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--accuracy-of-ranker", type=float, default=-1, help="How accurate the ranker should be (only synthetic case)")
    parser.add_argument("--beta", type=float, default=-1.0, help="Filter threshold for property 1 (negative numbers means no threshold)")
    parser.add_argument("--dataset", type=str, choices=["kuairand", "movielens"], default="kuairand", help="Dataset we want to train our models on.")
    parser.add_argument("--base-harm", type=float, choices=[0.3, 0.2, 0.1, 0.05], default=0.3,
                    help="Base harmfulness (it works only for the movielens example)")
    parser.add_argument("--score-model", type=str, default="sigformer", choices=["ncf", "ncfharm",
                                                    "lightgcl",
                                                    "sigformer",
                                                    "gformer",
                                                    "siren"])
    parser.add_argument("--score-type", type=str, choices=["naive", "harm", "globalharm"], default="subtraction")
    parser.add_argument("--method", type=str, choices=["replace", "remove", "hybrid"], default="remove")
    parser.add_argument("--use-single-stage-ranker", default=False, action="store_true",
                        help="Whether to use a single stage ranker which uses the same scores both for risk-control and ranking")
    parser.add_argument("--users", default="all", type=str, choices=["all", "hard", "easy"], help="Pick the users more likely to report videos")
    args = parser.parse_args()

    # Fix for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Collaborative Filter, no further features
    num_genres = 1

    # Set which will contain all results
    full_evaluation_results = []

    for run_id in range(args.runs):
        
        # Read the training and test/calibration data
        train_data = pd.read_table(f"./methods/{args.dataset}/training/train_{run_id}_False_{args.base_harm}.txt",
                                   header=None,
                                   sep=' ',
                                   names=["user_id", "video_id", "is_hate", "fraction_play_time", "is_hate_y", "fraction_play_time_y"])
        temp_data = pd.read_table(f"./methods/{args.dataset}/training/test_calibration_{run_id}_False_{args.base_harm}.txt", header=None, sep=' ',
                                  names=["user_id", "video_id", "is_hate", "fraction_play_time", "NONE", "NONE2"])
        
        # Consider only a certain split of users (e.g., hard users)
        # and filter the data accordingly.
        # We keep training/calibration fixed, but we just change the
        # test data.
        hard_users_idx = None
        if args.users != "all":
            print(f"[**] Filtering for {args.users} users.")
            hard_users_idx = extract_only_top_users(train_data, args.users)

        with open(f"./methods/{args.dataset}/results/test_{run_id}_{args.score_model}_False_{args.base_harm}_{args.epoch}.zlib.pickle", 'rb') as f:
            compressed_data = f.read()
            predicted_harmfulness_scores = pickle.loads(zlib.decompress(compressed_data))
        
        # Load NCF data if we are using a two-stage recommender
        if not args.use_single_stage_ranker:
            with open(f"./methods/{args.dataset}/results/test_{run_id}_ncf_False_{args.base_harm}_{args.epoch}.zlib.pickle", 'rb') as f:
                compressed_data = f.read()
                predicted_watch_time = pickle.loads(zlib.decompress(compressed_data))
        else:
            # Use the same scores as for the harmfulness
            predicted_watch_time = deepcopy(predicted_harmfulness_scores)

        num_users = max(
            train_data["user_id"].max(),
            temp_data["user_id"].max()
        )+1
        num_items = max(
            train_data["video_id"].max(),
            temp_data["video_id"].max()
        )+1
        MAX_SCORE = train_data["fraction_play_time"].max()

        test_data, calibration_data = train_test_split(temp_data, test_size=0.3, stratify=temp_data["is_hate"], random_state=run_id)
        del temp_data

        # Get max watch time
        max_watch_time_train = train_data.fraction_play_time.max()
        min_watch_time_train = train_data.fraction_play_time.min()
        
        # Rescale to not have negative values
        if args.score_model == "lightgcl":
            values_scores_ranker = []
            for user_id in predicted_harmfulness_scores.keys():
                for item_id in predicted_harmfulness_scores.get(user_id).keys():
                    values_scores_ranker.append(
                        predicted_harmfulness_scores.get(user_id)[item_id]
                    )            
            min_val = np.min(values_scores_ranker)

            for user_id in predicted_harmfulness_scores.keys():
                for item_id in predicted_harmfulness_scores.get(user_id).keys():
                    predicted_harmfulness_scores.get(user_id)[item_id] -= min_val

        # We consider repeated videos as coming from the training data
        # We drop those videos for which they are not seen twice.
        test_repeated_videos = train_data.dropna()
        test_repeated_videos = test_repeated_videos[(test_repeated_videos.is_hate == 0) & (test_repeated_videos.fraction_play_time > args.beta)]
        
        print("[*] REPEATED VIDEOS AVAILABLE: ", len(test_repeated_videos))
        print("[*] Num. harmful items on 2nd View: ", test_repeated_videos.is_hate_y.sum())

        print(f"Unique users: {num_users}")
        print(f"Max items: {num_items}")
        print(f"Max watch time: {MAX_SCORE}")

        print("[*] Training size: ", len(train_data))
        print("[*] Test size: ", len(test_data))
        print("[*] Calibration size: ", len(calibration_data))
        print("[*] Number of harmful video in test: ", test_data.is_hate.sum()) 
        
        # Given the training data, we compute the "global" score (e.g., the fraction of times a video was deemed harmful)
        tmp = train_data.groupby("video_id")["is_hate"].mean().reset_index()
        global_scores_items  = dict(zip(tmp["video_id"], tmp["is_hate"]))

        # Evaluate for all potential alphas
        for k in tqdm([20], desc=f"Run {run_id}"):

            # Consider only users with a number of items greater than k
            # both for safe, calibration and testing
            filtered_data = test_data[(test_data.groupby("user_id")["user_id"].transform("count") >= k)]
            filtered_calibration_data = calibration_data[(calibration_data.groupby("user_id")["user_id"].transform("count") >= k)]
            filtered_safe_data = test_repeated_videos

            # Obtain harm evaluations for each user in the test set and calibration set
            harmfulness_rating_test, _, ground_truth_test_harmfulness, _, _, _ = obtain_prediction_from_precomputed(
                filtered_data, precomputed_scores=predicted_harmfulness_scores
            )
            harmfulness_rating_test_calibration, _, ground_truth_calibration_harmfulness, user_ids_calibrations, item_ids_calibration, _ = obtain_prediction_from_precomputed(
                filtered_calibration_data, precomputed_scores=predicted_harmfulness_scores
            )

            del predicted_harmfulness_scores

            # Get the calibration scores from the ranker
            # These are needed to pick the best gamma for NAIVE
            naive_scores_calibration, _, _, _, _, _ = obtain_prediction_from_precomputed(
                filtered_calibration_data, precomputed_scores=predicted_watch_time
            )

            # Here, the pred ratings are the one we get from the prediction set
            pred_ratings_safe, _, _, user_ids_safe, item_ids_safe, _, feedback_second_time_watching, true_ratings_safe = obtain_prediction_from_precomputed(
                filtered_safe_data, precomputed_scores=predicted_watch_time, get_second_view=True
            )

            # Compute the harmfulness of the model on the full data for top-K
            pred_ratings, true_ratings, feedbacks, user_ids, item_ids, _ =  obtain_prediction_from_precomputed(
                filtered_data, precomputed_scores=predicted_watch_time
            )

            del predicted_watch_time

            # Re-gen the scores if we specify manually the ranker accuracy
            if args.accuracy_of_ranker != -1:
                harmfulness_rating_test = synthetic_classifier_predictions_random(
                    1-ground_truth_test_harmfulness, args.accuracy_of_ranker, 10000
                )
                harmfulness_rating_test_calibration = synthetic_classifier_predictions_random(
                    1-ground_truth_calibration_harmfulness, args.accuracy_of_ranker, 10000
                )

            # Create the naive and harmful aware scorers
            naive_scorer = NaiveScoreMethod(
                safe_user_ids=user_ids_safe,
                safe_items_scores=pred_ratings_safe,
                safe_items_scores_ground_truth=true_ratings_safe
            )
            harm_scorer = HarmScoreMethod(
                user_ids_safe,
                pred_ratings_safe,
                true_ratings_safe,
                user_ids,
                harmfulness_rating_test,
                user_ids_calibrations,
                harmfulness_rating_test_calibration,
            )

            # Generate the global harmfulness scores for both calibration and testing
            global_harmfulness_scores_calibration = np.array(
                [global_scores_items.get(item_id, 0) for item_id in item_ids_calibration]
            )
            global_harmfulness_scores_test = np.array(
                [global_scores_items.get(item_id, 0) for item_id in item_ids]
            )

            global_harm_scorer = HarmScoreMethod(
                user_ids_safe,
                pred_ratings_safe,
                true_ratings_safe,
                user_ids,
                1-global_harmfulness_scores_test,
                user_ids_calibrations,
                1-global_harmfulness_scores_calibration
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

            # We pick as base harm the current model
            base_harm = loss_model
            print("[**] Base harmfulness: ", base_harm)
            if args.dataset == "movielens":
                base_harm = args.base_harm

            # Find the best gamma based on the score
            if args.score_type == "globalharm":
                calibration_gammas_results = calibrate_gamma_precomputed(
                                            user_ids_calibrations,
                                            item_ids_calibration,
                                            global_harmfulness_scores_calibration,
                                            ground_truth_calibration_harmfulness,
                                            alpha=None, k=k,
                                            scoring_method=global_harm_scorer,
                                            max_score=1,
                                            cores=args.cores,
                                            method=args.method,
                                            device=DEVICE,
                                            num_gammas=100)
            elif args.score_type == "harm":
                calibration_gammas_results = calibrate_gamma_precomputed(
                                            user_ids_calibrations,
                                            item_ids_calibration,
                                            harmfulness_rating_test_calibration,
                                            ground_truth_calibration_harmfulness,
                                            alpha=None, k=k,
                                            scoring_method=harm_scorer,
                                            min_score=harmfulness_rating_test_calibration.min(),
                                            max_score=harmfulness_rating_test_calibration.max(),
                                            cores=args.cores,
                                            method=args.method,
                                            device=DEVICE,
                                            num_gammas=100)
            elif args.score_type == "naive":
                calibration_gammas_results = calibrate_gamma_precomputed(
                                        user_ids_calibrations,
                                        item_ids_calibration,
                                        naive_scores_calibration,
                                        ground_truth_calibration_harmfulness,
                                        alpha=None, k=k,
                                        scoring_method=naive_scorer,
                                        max_score=naive_scores_calibration.max(), # The max score is taken from the calibration
                                        cores=args.cores,
                                        method=args.method,
                                        device=DEVICE,
                                        num_gammas=100)

            # Filter the test data to consider only potentially "hard" users
            # Basically, given the current risk control, what happens to the
            # harder users. Here, we just need to change the user_id_chucks for
            # the evaluation.
            if hard_users_idx is not None:
                # Split IDs for each CPU, and we consider only those
                # hard users currently within the test set
                num_workers = min(args.cores, os.cpu_count())
                user_id_chunks = np.array_split(filtered_data[filtered_data.user_id.isin(hard_users_idx)].user_id.unique(), 
                                                num_workers
                                                )

            for alpha in tqdm(np.linspace(0.0, base_harm, num=25)[::-1], desc="Run alphas"):

                # Find the best alpha
                potential_gammas = []
                for _, (gamma, score) in enumerate(calibration_gammas_results):
                    if score <= alpha:
                        potential_gammas.append(gamma)
                        gamma_for_this_alpha = gamma
                gamma_for_this_alpha = min(potential_gammas) if len(potential_gammas) > 0 else calibration_gammas_results[-1][0]

                # Find the best gamma based on the score
                conformal_risk_score_list = []
                if args.score_type == "globalharm":
                    conformal_risk_score_list.append(
                        ("Global Harm", global_harm_scorer, gamma_for_this_alpha)
                    )
                elif args.score_type == "harm":
                    conformal_risk_score_list.append(
                        ("Harm", harm_scorer, gamma_for_this_alpha)
                    )
                elif args.score_type == "naive":
                    conformal_risk_score_list.append(
                        ("Naive" if not args.use_single_stage_ranker else f"Naive ({args.score_model.capitalize()})",
                         naive_scorer,
                         gamma_for_this_alpha)
                    )

                # Iterate over all possible scoring strategies
                for scorer_name, scorer, gamma_value in conformal_risk_score_list:

                    # Append the baseline value, if we have alpha == 0.
                    # We need it to compute the average improvement over the baseline
                    if alpha == 0.0:
                        full_evaluation_results.append(
                            [run_id, args.epoch, args.base_harm, args.beta, scorer_name, args.method, 0, -1, k, ndcg_model, loss_model, size_of_recommendation_model, 0, recall_model, 0, "Conformal"]
                        )

                    #Send each data to a CPU for processing
                    results = []
                    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                        futures = list(
                            executor.submit(
                                process_user,
                                chunk,
                                user_ids,
                                item_ids,
                                pred_ratings,
                                true_ratings, #if args.score_model == "ncf" else 1-feedbacks,
                                feedbacks,
                                k,
                                scorer,
                                args.score_type,
                                gamma_value,
                                method=args.method,
                                second_time_feedback=feedback_second_time_watching,
                                safe_user_ids=user_ids_safe) for chunk in user_id_chunks)

                        for future in futures:
                            results.extend(future.result())

                    # Get all the results and average them
                    size_of_recommendation, loss, ndcg_with_gamma, num_items_replaced, recall_with_gamma, item_used_for_this_user = zip(*results)
                    loss = np.mean(loss)
                    ndcg_with_gamma = np.mean(ndcg_with_gamma)
                    size_of_recommendation = np.mean(size_of_recommendation)
                    num_items_replaced = np.mean(num_items_replaced)
                    recall_with_gamma = np.mean(recall_with_gamma)
                    item_used_for_this_user = np.mean(item_used_for_this_user)

                    # Append the results to the final arrays
                    full_evaluation_results.append(
                        [run_id, args.epoch, args.base_harm, args.beta, scorer_name, args.method, gamma_value, alpha, k, ndcg_with_gamma, loss, size_of_recommendation, num_items_replaced, recall_with_gamma, item_used_for_this_user, "Conformal"]
                    ) 
            
        # Free some memory before running again
        del train_data

    df = pd.DataFrame(
        full_evaluation_results,
        columns= ["run_id", "epoch", "base_harm", "beta", "conformal_score", "conformal_method", "gamma", "alpha", "k", "nDCG @ k", "H(S,X)", "|S|", "random_items", "Recall @ k", "items_exhaustes", "Method"]
    )
    df.to_csv(f"results_{args.dataset}_{args.method}_{args.score_model}_{args.score_type}_{args.runs}_{args.beta}_{args.base_harm}_{args.epoch}_{args.use_single_stage_ranker}_{args.users}.csv", index=None)
