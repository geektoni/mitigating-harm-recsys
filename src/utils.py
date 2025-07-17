import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

from src.calibration import harmfulness_loss, recall, adaptive_threshold
from src.score_functions import NaiveScoreMethod

from tqdm import tqdm

# Create a PyTorch dataset class
class KuaiHarmDataset(Dataset):
    def __init__(self, interaction_data, video_data):

        # Merge video information and interaction data
        # columsn_from_interaction_data = interaction_data.columns.tolist()
        # columsn_from_interaction_data.pop(columsn_from_interaction_data.index("video_id"))
        # interaction_and_video_informations = interaction_data.merge(video_data, on="video_id", how="inner")
        # interaction_and_video_informations.drop(columns=columsn_from_interaction_data, inplace=True)

        self.users = torch.tensor(interaction_data["user_id"].values, dtype=torch.long)
        self.items = torch.tensor(interaction_data["video_id"].values, dtype=torch.long)
        self.watch_time = torch.tensor(interaction_data["fraction_play_time"].values, dtype=torch.float32)
        self.harm = torch.tensor(interaction_data["is_hate"].values, dtype=torch.int32)

        #self.video_information = torch.tensor(interaction_and_video_informations.iloc[:, 2:].values, dtype=torch.float32)
    
    def __len__(self):
        return len(self.harm)
    
    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], torch.tensor([0]), self.watch_time[idx], self.harm[idx]
        #return self.users[idx], self.items[idx], self.video_information[idx], self.watch_time[idx], self.harm[idx]

# Create a PyTorch dataset class
class MovieLensDataset(Dataset):
    def __init__(self, data):
        self.users = torch.tensor(data['user_id'].values, dtype=torch.long)
        self.items = torch.tensor(data['item_id'].values, dtype=torch.long)
        #self.genres = torch.tensor(data.iloc[:, 3:-1].values, dtype=torch.float32)
        self.ratings = torch.tensor(data['rating'].values, dtype=torch.float32)
        self.binary_feedback = torch.tensor(data['binary_feedback'].values, dtype=torch.int32)
    
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], torch.tensor([0]), self.ratings[idx], self.binary_feedback[idx]

# Define the neural network ranker
class RankerNN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32, num_genres=1,
                 max_output_value=5, min_output_value=1):
        super(RankerNN, self).__init__()
        self.user_embedding = nn.Embedding(num_users + 1, embedding_dim)
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim)
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 2 + num_genres, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.max_output_value = max_output_value
        self.min_output_value = min_output_value

        self.apply_sigmod_in_forward = True
    
    def disable_sigmoid_in_forward(self, disable=True):
        self.apply_sigmod_in_forward = not disable

    def forward(self, user, item, genre):
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)
        x = torch.cat([user_emb, item_emb, genre], dim=1)
        if self.apply_sigmod_in_forward:
            output = torch.sigmoid(self.fc(x).squeeze())*(self.max_output_value-self.min_output_value) + self.min_output_value
        else:
            output = self.fc(x).squeeze()
        return output if output.dim() > 0 else output.unsqueeze(0) 
    
    def predict(self, user, item, genre, gamma=None, device="cpu"):
        assert self.apply_sigmod_in_forward, "Predicting with sigmoid disabled!"
        user = user.to(device)
        item = item.to(device)
        genre = genre.to(device)
        scores = self.forward(user, item, genre)
        mask = np.zeros_like(scores.numpy(force=True)).tolist()
        if gamma is not None:
            mask = scores >= gamma
        return (scores, mask)

# Train the model
def train_model(model, train_loader, criterion, optimizer, epochs=5, use_feedback=False, use_gpu=True, verbose=False):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() and use_gpu else "cpu")
    model.to(device)

    criterion = criterion.to(device)
    
    model.train()
    model.disable_sigmoid_in_forward(use_feedback)
    for epoch in range(epochs):
        epoch_loss = 0
        for user, item, genre, rating, feedback in tqdm(train_loader, disable=not verbose):

            user = user.to(device)
            item = item.to(device)
            genre = genre.to(device)
            rating = rating.to(device)
            feedback = feedback.to(device)

            optimizer.zero_grad()
            prediction = model(user, item, genre).to(device)
            loss = criterion(prediction, rating) if not use_feedback else criterion(prediction, feedback.float())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if verbose:
            print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(train_loader):.4f}")
    model.disable_sigmoid_in_forward(False)

# Evaluate using nDCG
def ndcg_at_k(predictions, true_ratings, k=10, sort=True):
    sorted_indices = np.argsort(-predictions) if sort else list(range(len(predictions)))
    ideal_sorted_indices = np.argsort(-true_ratings)
    
    dcg = sum((true_ratings[sorted_indices[i]] / np.log2(i + 2)) for i in range(min(k, len(predictions))))
    idcg = sum((true_ratings[ideal_sorted_indices[i]] / np.log2(i + 2)) for i in range(min(k, len(predictions))))
    
    return dcg / idcg if idcg > 0 else 0


# Parallelize the computation over multiple cores
def process_user_only_model(user_ids_chunk, user_ids, item_ids, pred_ratings, true_ratings, feedbacks, k):
    
    results = []
    for user_id in user_ids_chunk:
    
        # Filter only the users equal to the user id
        mask_ids = user_ids == user_id
        
        pred_ratings_user = pred_ratings[mask_ids]
        true_ratings_user = true_ratings[mask_ids]
        
        sorted_indices = np.argsort(-pred_ratings_user)
        feedback_user = feedbacks[mask_ids][sorted_indices]
        
        size_of_recommendation = len(feedback_user[:k])
        loss_value = harmfulness_loss(feedback_user, k=k)
        ndcg_no_gamma_value = ndcg_at_k(np.array(pred_ratings_user), np.array(true_ratings_user), k=k)

        threshold = adaptive_threshold(true_ratings_user)
        recommended_items_ids_for_user = item_ids[mask_ids][sorted_indices]
        relevant_items_for_user = item_ids[mask_ids][true_ratings_user >= threshold]

        recall_per_user = recall(
            recommended_items_ids_for_user, 
            relevant_items_for_user,
            k=k
        )
        
        results.append((size_of_recommendation, loss_value, ndcg_no_gamma_value, recall_per_user))
    
    return results

# Parallelize the computation over multiple cores
def process_user(user_ids_chunk, user_ids, item_ids, pred_ratings, true_ratings, feedbacks,
                 k,
                 scoring_method: NaiveScoreMethod, score, gamma, method="replace",
                 second_time_feedback=None, safe_user_ids=None):
    
    # Array where we store the full results
    results = []
    for user_id in user_ids_chunk:
    
        # Filter only the users equal to the user id
        mask_ids = user_ids == user_id
        
        # Get the predicted ratins and true rating for the given user
        pred_ratings_user = pred_ratings[mask_ids]
        true_ratings_user = true_ratings[mask_ids]

        # How many safe items we need (on average) for each user
        # to ensure the conformity
        item_used_for_this_user = 0
        
        if method == "replace" or method == "hybrid":

            # We replace all items in the mask with random "trusted" items.
            # In this synthetic case, we replace them with random items picked
            # from the remaining user items, by ensuring they were not flagged.
            # This might be akin to provide "trusted" curated videos, or previously
            # seen videos the user did not flag.

            new_scores, replaced_items_mask, gt_new_scores, index_replaced_items, index_safe_items, item_used_for_this_user = scoring_method.replace_low_scores(
                pred_ratings_user,
                user_id,
                k,
                gamma,
                item_ids=item_ids[mask_ids], # we give only the ids for the current user
                score=score # type of scoring function we want
            )
            num_item_replaced = len(index_replaced_items)

            # Update the ground truth scores if we are adding previously seen
            # elements
            if num_item_replaced > 0:
                true_ratings_user[index_replaced_items] = gt_new_scores
            
            # Replace with the correct harmfulness given the feedback
            harmful_items = feedbacks[mask_ids]
            if second_time_feedback is not None:
                # Replace the harmfulness with the correct one
                if len(index_replaced_items) > 0:
                    harmful_items[index_replaced_items] = second_time_feedback[index_safe_items]
            else:
                if len(index_replaced_items) > 0:
                    harmful_items[index_replaced_items] = 0

            # We might have replaced less items than needed, so we drop
            # the others if needed by the strategy
            if method == "hybrid":

                # We keep track on which element we have replaced, removed or kept
                elements_we_have_replaced = np.zeros_like(replaced_items_mask, dtype=int)
                if num_item_replaced > 0:
                    # We mask all the items we replaced so far, and we
                    # leave the others for removal
                    replaced_items_mask[index_replaced_items] = 0
                    elements_we_have_replaced[index_replaced_items] = 1

                # We get the negative, these now are the elements we want to keep
                replaced_items_mask = ~replaced_items_mask
                
                # Get "relevant" items for this user (%75 percentile)
                # We need to do this before filtering
                if len(true_ratings_user) > 0:
                    sorted_indices_full = np.argsort(-new_scores)
                    threshold = adaptive_threshold(true_ratings_user)
                    recommended_items_ids_for_user = item_ids[mask_ids][replaced_items_mask][np.argsort(-new_scores[replaced_items_mask])]
                    relevant_items_for_user = item_ids[mask_ids][sorted_indices_full][true_ratings_user[sorted_indices_full] >= threshold]
                else:
                    recommended_items_ids_for_user = []
                    relevant_items_for_user = []
            
                # Compute the harmfulness of this recommendation
                new_scores = np.array(new_scores)[replaced_items_mask]
                sorted_indices = np.argsort(-new_scores)
                harmful_items = harmful_items[replaced_items_mask][sorted_indices]
                
                # These are the items I have replaced with the procedure.
                # It should be always less than k.
                # This is the correct number of replaced items
                num_item_replaced = np.sum(elements_we_have_replaced[replaced_items_mask][sorted_indices][:k] == 1)

            else:

                # We sort the new scores, the harmful items and the replaced items mask
                sorted_indices = np.argsort(-new_scores)
                harmful_items = harmful_items[sorted_indices]

                # Compute a mask showing those items we did replace
                items_we_really_replaced = np.zeros_like(replaced_items_mask, dtype=int)
                if len(index_replaced_items) > 0:
                    items_we_really_replaced[index_replaced_items] = 1
            
                # We get the harmful items, replaced items by index
                items_we_really_replaced = items_we_really_replaced[sorted_indices]
                
                # Compute how many random items we have in the top-k recommendations now.
                # The idea is check how much random content we are feeding the user.
                num_item_replaced = np.sum(items_we_really_replaced[:k])

                # Get "relevant" items for this user (%75 percentile)
                threshold = adaptive_threshold(true_ratings_user)
                recommended_items_ids_for_user = item_ids[mask_ids][sorted_indices]
                relevant_items_for_user = item_ids[mask_ids][true_ratings_user >= threshold]

            size_of_recommendation_filtered = len(new_scores[:k])
            loss_filtered_value = harmfulness_loss(harmful_items, k=k)
            ndcg_with_gamma_value = ndcg_at_k(
                np.array(new_scores),
                np.array(true_ratings_user),
                k=k) # We do not need to sort, since it has been done by replace_low_scores
            recall_with_gamma_value = recall(
                recommended_items_ids_for_user,
                relevant_items_for_user,
                k=k
            )

        else:

            # Filter out the scores given the function we consider
            score_masks_given_gamma = scoring_method.score_items(
                pred_ratings_user,
                gamma,
                user_id=user_id,
                item_ids=item_ids[mask_ids], # we give only the ids for the current user
                score=score # type of scoring function we want
            )

            # We sort the predicted scores and we extract the harmful items
            # based on their position (more relevant are first).
            sorted_indices = np.argsort(-pred_ratings_user)
            harmful_items = feedbacks[mask_ids][sorted_indices]
            
            size_of_recommendation_filtered = min(np.sum(score_masks_given_gamma), k)
            loss_filtered_value = harmfulness_loss(harmful_items[score_masks_given_gamma[sorted_indices]], k=k)
            ndcg_with_gamma_value = ndcg_at_k(
                np.array(pred_ratings_user[score_masks_given_gamma]),
                np.array(true_ratings_user), k=k)
            num_item_replaced = 0 # It is only used when using the "replace" method

            # Get "relevant" items for this user (%75 percentile)
            threshold = adaptive_threshold(true_ratings_user)
            relevant_items_for_user = item_ids[mask_ids][true_ratings_user >= threshold]
            recommended_items_ids_for_user = item_ids[mask_ids][sorted_indices][score_masks_given_gamma]

            recall_with_gamma_value = recall(
                recommended_items_ids_for_user,
                relevant_items_for_user,
                k=k
            )

        results.append((size_of_recommendation_filtered, loss_filtered_value, ndcg_with_gamma_value, num_item_replaced, recall_with_gamma_value, item_used_for_this_user))
    
    return results

def obtain_prediction_from_precomputed(dataset, precomputed_scores, get_second_view = False):
    
    pred_ratings = []
    user_ids = []
    masks = []
    feedbacks = []
    second_feedbacks = []
    second_rating = []
    true_ratings = []
    item_ids = []

    for (idx_row, row) in dataset.iterrows():
        
        if not get_second_view:
            prediction = precomputed_scores.get(int(row.user_id)).get(int(row.video_id))
        else:
            prediction = row.fraction_play_time # we pick the correct one, since we know how much time the user watched

        item_ids.append(row.video_id)

        user_ids.append(row.user_id)
        masks.append(0)
        feedbacks.append(row.is_hate)
        true_ratings.append(row.fraction_play_time)
        pred_ratings.append(prediction)
        if get_second_view:
            second_feedbacks.append(
                row.is_hate_y
            )
            second_rating.append(
                row.fraction_play_time_y
            )

    pred_ratings = np.array(pred_ratings)
    user_ids = np.array(user_ids)
    masks = np.array(masks)
    feedbacks = np.array(feedbacks)
    true_ratings = np.array(true_ratings)
    item_ids = np.array(item_ids)
    second_feedbacks = np.array(second_feedbacks)
    second_rating = np.array(second_rating)

    if get_second_view:
        return pred_ratings, true_ratings, feedbacks, user_ids, item_ids, masks, second_feedbacks, second_rating
    else:
        return pred_ratings, true_ratings, feedbacks, user_ids, item_ids, masks

def obtain_prediction(model, test_loader, gamma=None, device="cpu"):

    pred_ratings = []
    user_ids = []
    masks = []
    feedbacks = []
    true_ratings = []
    item_ids = []

    model.eval()
    with torch.no_grad():
        for user, item, genre, rating, feedback in test_loader:

            prediction, mask = model.predict(user, item, genre, gamma=gamma, device=device)

            item_ids.extend(item.numpy())

            user_ids.extend(user.numpy())
            masks.extend(mask)
            feedbacks.extend(feedback.numpy())
            true_ratings.extend(rating.numpy())
            pred_ratings.extend(prediction.numpy(force=True))

    pred_ratings = np.array(pred_ratings)
    user_ids = np.array(user_ids)
    masks = np.array(masks)
    feedbacks = np.array(feedbacks)
    true_ratings = np.array(true_ratings)
    item_ids = np.array(item_ids)

    return pred_ratings, true_ratings, feedbacks, user_ids, item_ids, masks

def synthetic_classifier_predictions_random(y, delta, concentration=100, eps=1e-15):

    def beta_parameters_from_mean(mean, strength):
        """
        Vectorized helper: compute alpha and beta from mean and strength.
        """
        mean = np.clip(mean, 1e-10, 1 - 1e-10)  # avoid division by zero
        alpha = mean * strength
        beta = (1 - mean) * strength
        return alpha, beta
    
    y = np.asarray(y)
    alpha, beta = beta_parameters_from_mean(delta, concentration)
    
    # Sample all confidence scores at once
    confidences = np.random.beta(alpha, beta, size=y.shape)
    
    # Use confidence if label == 1, else use 1 - confidence
    probabilities = np.where(y == 1, confidences, 1 - confidences)
    
    return probabilities

def synthetic_multiclass_predictions_by_accuracy(y, target_accuracy, num_classes=5, concentration=100, eps=1e-15):
    """
    Generate synthetic multiclass predictions for true labels y (as integers 0 to num_classes-1)
    such that the expected classification accuracy is target_accuracy.
    
    For each sample:
      - With probability target_accuracy, we generate a probability vector intended to be correct
        (i.e. peaked on the true class).
      - With probability (1 - target_accuracy), we generate one intended to be wrong 
        (i.e. peaked on one of the incorrect classes, chosen uniformly at random).
    
    We add randomness by sampling from a Dirichlet distribution.
    The 'concentration' parameter controls the peakedness:
      - A high concentration (e.g., 100) makes the Dirichlet sample nearly deterministic.
      - A lower concentration yields more spread-out probability vectors.
    
    Parameters:
      - y: array-like of shape (n_samples,). True labels as integers from 0 to num_classes-1.
      - target_accuracy: desired accuracy (e.g., 0.8 for 80% correct).
      - num_classes: total number of classes (default is 5).
      - concentration: concentration parameter for the Dirichlet distribution.
      - eps: small value for numerical stability.
    
    Returns:
      - predictions: array of shape (n_samples, num_classes) containing probability vectors.
    """
    y = np.asarray(y)
    n_samples = y.shape[0]
    predictions = np.empty(n_samples)
    
    for i in range(n_samples):
        # Decide whether to simulate a "correct" or "incorrect" prediction.
        if np.random.rand() < target_accuracy:
            # Correct prediction: we want the true class to be the highest.
            # Build a Dirichlet parameter vector that is peaked on the true class.
            alpha = np.ones(num_classes)
            alpha[int(y[i])-1] = concentration
        else:
            # Incorrect prediction: choose a wrong class uniformly at random.
            wrong_classes = [cls for cls in range(num_classes) if cls != y[i]]
            chosen_wrong = np.random.choice(wrong_classes)
            alpha = np.ones(num_classes)
            alpha[chosen_wrong] = concentration
        
        # Sample from the Dirichlet distribution.
        sample = np.random.dirichlet(alpha)
        # Clip to avoid numerical issues.
        sample = np.clip(sample, eps, 1 - eps)
        predictions[i] = np.random.choice(num_classes, p=sample)+1 # Add 1 since scores are between 1 and 5.
        
    return predictions