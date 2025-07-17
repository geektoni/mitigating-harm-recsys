import numpy as np
import torch
import torch.nn as nn

class RankerNN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32, num_genres=19,
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