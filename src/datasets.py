import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from tqdm import tqdm

def load_kuairand_data(path_training_data="../KuaiRand-Harm/training/single_interactions_user_features_27k.csv",
                       video_training_data="../KuaiRand-Harm/training/video_features_basic_27k.csv",
                       repeated_training_data="../KuaiRand-Harm/training/repeated_interactions_user_features_27k.csv"):

    # Read the videos, and perform some preprocessing
    training_data = pd.read_csv(path_training_data)
    video_data = pd.read_csv(video_training_data)
    repeated_videos = pd.read_csv(repeated_training_data)

    return training_data, video_data, repeated_videos

def load_movielens_1m(path="ml-1m/ratings.dat", alpha=0.999, concentration=25):
    
    df = pd.read_table(path, header=None, names=["user_id", "item_id", "rating", "timestamp"],
                       sep='::',
                       engine="python")
    df.drop(columns=['timestamp'], inplace=True)
    
    betas = np.random.beta(alpha*concentration, (1-alpha)*concentration, size=len(df))
    
    probabilities = np.exp(-df.rating * betas)
    binary_feedback = np.random.binomial(1, p=probabilities)
    df.loc[:, "binary_feedback"] = binary_feedback
    
    return df 

    for k, (user, user_df) in tqdm(enumerate(df.groupby("user_id")), total=len(df.user_id.unique())):
        beta = betas[k]

        ratings_for_this_user = user_df.rating.values
        prob = np.exp(-ratings_for_this_user * beta)
        binary_feedback = np.random.binomial(1, p=prob)
        
        # Assign to the correct user
        df.loc[df["user_id"] == user, "binary_feedback"] = binary_feedback
    
    df.to_csv("ratings_ml25m_harmfulness.csv")

    return df


# Load the dataset
def load_movielens_100k(path="ml-100k/u.data", item_path="ml-100k/u.item", weighting=1, alpha=0.999, concentration=25):
    df = pd.read_csv(path, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    df.drop(columns=['timestamp'], inplace=True)
    
    items = pd.read_csv(item_path, sep='|', encoding='latin-1', header=None, usecols=[0] + list(range(5, 24)), names=['item_id'] + [f'genre_{i}' for i in range(19)])
    df = df.merge(items, on='item_id', how='left')

    # Compute item similarity based on genre
    item_features = items.drop(columns=['item_id']).values
    similarity_matrix = cosine_similarity(item_features)
    item_sim = dict(zip(items['item_id'], similarity_matrix))
    
    def get_flagging_prob(r, beta, sim_score, weighting=1):
        base_prob = np.exp(-r * beta)
        return min(1, weighting*base_prob + (1-weighting) * sim_score)  # Ensure probability stays in [0,1]
    
    flagged_items = {}
    
    for user, user_df in df.groupby("user_id"):
        flagged_items[user] = {}
        beta = np.random.beta(alpha*concentration, (1-alpha)*concentration)
        
        for rating in user_df["rating"].unique():
            sample_item = user_df[user_df["rating"] == rating].sample(1)["item_id"].values[0]
            flagged_items[user][sample_item] = True
        
        binary_feedback = []
        for _, row in user_df.iterrows():
            item_id = row["item_id"]
            rating = row["rating"]

            if item_id in flagged_items.get(user):
                does_the_user_flag = True
            else:
                # We set other_id-1 since it is an array and not a dictionary
                sim_score = max([item_sim[item_id][other_id-1] for other_id in flagged_items[user]])
        
                prob = get_flagging_prob(rating, beta, sim_score, weighting)
                does_the_user_flag = np.random.binomial(1, p=prob)
        
                if does_the_user_flag:
                    flagged_items[user][item_id] = True

            binary_feedback.append(does_the_user_flag)
            
        # Assign to the correct user
        df.loc[df["user_id"] == user, "binary_feedback"] = binary_feedback
    
    return df