import numpy as np
import pandas as pd

def load_movielens_1m(path="../movielens/ml-1m/ratings.dat", alpha=0.999, concentration=25):
    
    df = pd.read_table(path, header=None, names=["user_id", "video_id", "fraction_play_time", "timestamp"],
                       sep='::',
                       engine="python")
    df.drop(columns=['timestamp'], inplace=True)
    
    # Create a mapping of user_id to their corresponding beta value
    unique_users = df['user_id'].unique()
    user_betas = np.random.beta(alpha * concentration, (1 - alpha) * concentration, size=len(unique_users))
    beta_map = dict(zip(unique_users, user_betas))
    betas = np.array([beta_map.get(usrid) for usrid in df.user_id.values])
    
    probabilities = np.exp(-df.fraction_play_time * betas)
    binary_feedback = np.random.binomial(1, p=probabilities)
    df.loc[:, "is_hate"] = binary_feedback

    # We copy them from the original. Here, since we do not have the double time
    # we just assume it is not harmful anymore. 
    df.loc[:, "is_hate_y"] = 0
    df.loc[:, "fraction_play_time_y"] = df.loc[:, "fraction_play_time"]
    
    return df 