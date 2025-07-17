from abc import abstractmethod
import numpy as np

class NaiveScoreMethod:

    def __init__(self, safe_user_ids, safe_items_scores, safe_items_scores_ground_truth):
        self.safe_user_ids = safe_user_ids
        self.safe_items_scores = safe_items_scores
        self.safe_items_scores_ground_truth = safe_items_scores_ground_truth
    
    @abstractmethod
    def score_items(self, predictions, gamma, **kwargs):
        return predictions >= gamma
    
    @abstractmethod
    def replace_low_scores(self, scores, user_id, k, threshold, **kwargs):
        """
        Sort items by score in descending order and replace each item in the top-k 
        with a score below the threshold with a randomly sampled item from outside the top-k.

        Args:
            scores (np.array): Array of item scores.
            k (int): Number of top items to consider.
            threshold (float): Minimum score required to stay in the top-k.

        Returns:
            np.array: Modified array with replacements.
        """

        # Filter only those items for which we have the user id
        mask_current_id = self.safe_user_ids == user_id

        if kwargs.get("updated_scores", None) is not None:
            # This step is needed if we perform the initial thresholding
            # by using updated scores. This is done since the relevance of
            # an item and its harmfulness might be diverging objectives.
            low_score_mask = kwargs.get("updated_scores", None) < threshold
        else:
            low_score_mask = scores < threshold
        
        # Indices that needs replacements
        low_score_indices = np.arange(len(scores))[low_score_mask]  # Indices that need replacement
        num_replacements = len(low_score_indices)

        # Consider only the items with a score above the lowest one here
        # We ensure we always get an element within the top-k (in theory)
        mimal_score_current_top_k = scores[np.argsort(-scores)][:k].min()
        safe_item_ids = np.arange(len(self.safe_items_scores))[mask_current_id][self.safe_items_scores[mask_current_id] > mimal_score_current_top_k]

        # If they are not enough, then just pick a num_replacements
        if len(safe_item_ids) < num_replacements:
            safe_item_ids = np.arange(len(self.safe_items_scores))[mask_current_id]
            sorted_ids_top_k = np.argsort(-self.safe_items_scores[safe_item_ids])[:num_replacements]
            safe_item_ids = safe_item_ids[sorted_ids_top_k]

        # Total exausted items for this user
        item_used_for_this_user = 0
        if len(self.safe_items_scores[mask_current_id]) > 0:
            item_used_for_this_user = len(safe_item_ids)/len(self.safe_items_scores[mask_current_id])

        # Step 4: Select random replacements from candidate items
        # Replace only if we have enough items
        replacement_indices = np.array([]) # Empty replacement indices
        replacement_low_score_indices = np.array([])
        if num_replacements > 0 and len(safe_item_ids) > 0:
            # Replace low-score items with new items, with a new score
            replacement_indices = np.random.choice(safe_item_ids, size=min(num_replacements, len(safe_item_ids)), replace=False)
            
            # Get the indices of the elements we want to replace
            replacement_low_score_indices = np.random.choice(low_score_indices, size=min(num_replacements, len(safe_item_ids)), replace=False)

            # We replace only up to the replacement indices length
            # Therefore, if we confron low_score_mask and replacement_indices we can understand in which situation
            # we are currently
            scores[replacement_low_score_indices] = self.safe_items_scores[replacement_indices]

        # Return the new scores, the mask indicating which element we masked and
        # the total number of replacements
        return scores, low_score_mask, self.safe_items_scores_ground_truth[replacement_indices] if len(replacement_indices) > 0 else replacement_indices, replacement_low_score_indices, replacement_indices, item_used_for_this_user

class SimilarityScoreMethod(NaiveScoreMethod):

    def __init__(self, safe_user_ids, safe_items_scores, safe_items_scores_ground_truth, harmfulness_data, similarity_matrix):

        # Dataset that contains information about each user
        # about which items they flagged as harmful during
        # their previous iterations.
        self.harmfulness_data = harmfulness_data
        assert all(self.harmfulness_data.binary_feedback == 1)

        # Similarity matrix computed between each item
        self.similarity_matrix = similarity_matrix

        super().__init__(safe_user_ids, safe_items_scores, safe_items_scores_ground_truth)

    def replace_low_scores(self, scores, user_id, k, threshold, **kwargs):

        item_ids = kwargs.get("item_ids")
        score_type = kwargs.get("score")

        # Flagged items for this specific user
        flagged_items_ids = self.harmfulness_data[self.harmfulness_data.user_id == user_id]['item_id'].values

        updated_predictions = self._similarity_harmfulness_score(item_ids, 
                                                            flagged_items_ids,
                                                            scores,
                                                            self.similarity_matrix,
                                                            type=score_type)

        return super().replace_low_scores(scores, user_id, k, threshold,
                                          updated_scores=updated_predictions, **kwargs)

    def score_items(self, predictions, gamma, **kwargs):

        user_id = kwargs.get("user_id")
        item_ids = kwargs.get("item_ids")
        score_type = kwargs.get("score")

        # Flagged items for this specific user
        flagged_items_ids = self.harmfulness_data[self.harmfulness_data.user_id == user_id]['item_id'].values

        updated_predictions = self._similarity_harmfulness_score(item_ids, 
                                                            flagged_items_ids,
                                                            predictions,
                                                            self.similarity_matrix,
                                                            type=score_type)

        return super().score_items(updated_predictions, gamma)
    
    def _similarity_harmfulness_score(self,
                                      item_ids,
                                      flagged_ids,
                                      model_scores,
                                      similarity_dataset,
                                      type="subtraction"):

        # If the user has no flagged items, then we return the scores unchanged.
        if len(flagged_ids) == 0:
            return model_scores

        # Since item_features has multiple instances of the same item, here we filter
        # the dataframe and we pick only the first occurrence
        weights = np.array(
            [max(similarity_dataset[item][flagged] for flagged in flagged_ids) for item in item_ids]
        )

        # Recompute the scores with the weights
        if type == "subtraction":
            return model_scores - weights
        elif type == "similarity":
            return 1-weights
        else:
            return model_scores*(1-weights)


class HarmScoreMethod(NaiveScoreMethod):

    def __init__(self, safe_user_ids,
                 safe_items_scores,
                 safe_items_scores_ground_truth,
                 test_user_ids,
                 harmfulness_items_scores,
                 calibration_user_ids,
                 calibration_harmfulness_items_scores):

        self.test_user_ids = test_user_ids
        self.harmfulness_items_scores = harmfulness_items_scores
        self.calibration_user_ids = calibration_user_ids
        self.calibration_harmfulness_items_scores = calibration_harmfulness_items_scores

        super().__init__(safe_user_ids, safe_items_scores, safe_items_scores_ground_truth)
    
    def replace_low_scores(self, scores, user_id, k, threshold, **kwargs):

        is_calibrating = kwargs.get("is_calibrating", False)
        
        if is_calibrating:
            # Extract the harmfulness predictions of only such items
            mask_ids = self.calibration_user_ids == user_id
            updated_predictions = self.calibration_harmfulness_items_scores[mask_ids]
        else:
            # Extract the harmfulness predictions of only such items
            mask_ids = self.test_user_ids == user_id
            updated_predictions = self.harmfulness_items_scores[mask_ids]

        return super().replace_low_scores(scores, user_id, k, threshold,
                                          updated_scores=updated_predictions, **kwargs)

    def score_items(self, predictions, gamma, **kwargs):
        
        user_id = kwargs.get("user_id")
        is_calibrating = kwargs.get("is_calibrating", False)
        
        if is_calibrating:
            # Extract the harmfulness predictions of only such items
            mask_ids = self.calibration_user_ids == user_id
            updated_predictions = self.calibration_harmfulness_items_scores[mask_ids]
        else:
            # Extract the harmfulness predictions of only such items
            mask_ids = self.test_user_ids == user_id
            updated_predictions = self.harmfulness_items_scores[mask_ids]
        
        return super().score_items(updated_predictions, gamma)