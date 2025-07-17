import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import zlib
import pickle

def obtain_prediction(model, test_loader, gamma=None, device="cpu", verbose=False):

    pred_ratings = []
    user_ids = []
    masks = []
    feedbacks = []
    true_ratings = []
    item_ids = []

    model.eval()
    with torch.no_grad():
        for user, item, genre, rating, feedback in tqdm(test_loader, disable=not verbose):

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

# Create a PyTorch dataset class
class KuaiHarmDataset(Dataset):
    def __init__(self, interaction_data, video_data):

        self.users = torch.tensor(interaction_data["user_id"].values, dtype=torch.long)
        self.items = torch.tensor(interaction_data["video_id"].values, dtype=torch.long)
        self.watch_time = torch.tensor(interaction_data["fraction_play_time"].values, dtype=torch.float32)
        self.harm = torch.tensor(interaction_data["is_hate"].values, dtype=torch.int32)

    def __len__(self):
        return len(self.harm)
    
    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], torch.tensor([0]), self.watch_time[idx], self.harm[idx]

def train_model(model, train_loader, criterion,
                optimizer, epochs=5, use_feedback=False,
                use_gpu=True, verbose=False, device="cpu",
                validation_loader=None, validation_filename=None):
    
    model.to(device)

    criterion = criterion.to(device)
    
    model.train()
    model.disable_sigmoid_in_forward(use_feedback)
    for epoch in range(1, epochs+1):
        epoch_loss = 0
        for user, item, genre, rating, feedback in tqdm(train_loader, disable=not verbose):

            user = user.to(device)
            item = item.to(device)
            genre = genre.to(device)
            rating = rating.to(device)
            feedback = feedback.float().to(device)

            optimizer.zero_grad()
            prediction = model(user, item, genre).to(device)
            loss = criterion(prediction, rating) if not use_feedback else criterion(prediction, feedback)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if verbose:
            print(f"Epoch {epoch}, Loss: {epoch_loss / len(train_loader):.4f}")
        
        # Validation step if we pass the corresponding validation loader
        if validation_loader is not None:
            model.disable_sigmoid_in_forward(False)
            if epoch % 10 == 0:
                pred_ratings, _, _, user_ids_safe, item_ids_safe, _ = obtain_prediction(
                        model, validation_loader, device=device, verbose=True
                )

                ratings = {}
                for pred, user_id, item_id in zip(pred_ratings, user_ids_safe, item_ids_safe):
                    if user_id not in ratings:
                        ratings[user_id] = {}
                    if item_id not in ratings[user_id]:
                        ratings[user_id][item_id] = []
                    ratings[user_id][item_id] = pred
                
                compressed_data = zlib.compress(pickle.dumps(ratings))
                with open(validation_filename+f"_{epoch}.zlib.pickle", 'wb') as f:
                    f.write(compressed_data)
                
            model.disable_sigmoid_in_forward(use_feedback)


    model.disable_sigmoid_in_forward(False)
