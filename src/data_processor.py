from typing import Tuple

import torch
from pandas import DataFrame
from torch.nn import Parameter


class DataProcessor:
    def __init__(self, original_df: DataFrame):
        self.min_rating = min(original_df.rating.values)
        self.max_rating = max(original_df.rating.values)
        (
            self.ratings_by_user,
            self.histograms_by_users,
            self.item_to_index_rating,
        ) = self.data_process(original_df=original_df)

    def data_process(self, original_df: DataFrame) -> Tuple:
        """
        This function creates the original ratings embedding for each user and saves mapping
        from index to item place in the rating.
        In addition, it also creates the original histogram of the ratings of the user.
        :param original_df: original dataframe
        :return: Tuple of ratings_by_users, histograms_by_users, item_to_index_rating
        """
        ratings_by_users, histograms_by_users, item_to_index_rating = {}, {}, {}
        items_grouped_by_users = original_df.groupby("user_id")

        for user_id, group in items_grouped_by_users:
            ratings_as_tensor = torch.Tensor(group.rating.values)
            ratings_by_users[user_id] = Parameter(ratings_as_tensor, requires_grad=False)
            histograms_by_users[user_id] = torch.histc(
                ratings_as_tensor, bins=self.max_rating, min=self.min_rating, max=self.max_rating
            )
            item_to_index_rating[user_id] = {
                item_id: i for i, (_, _, item_id, _) in enumerate(group.itertuples())
            }

        return ratings_by_users, histograms_by_users, item_to_index_rating
