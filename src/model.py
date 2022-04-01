import torch
from torch import Tensor
from torch.nn import Embedding, Module
from torch.nn.init import xavier_normal_

from src.data_encoder import DataEncoder
from src.data_processor import DataProcessor


class HistogramMF(Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        min_rating: int,
        max_rating: int,
        data_encoder: DataEncoder,
        data_processor: DataProcessor,
        n_factors=50,
    ):
        super().__init__()
        self.user_factors = Embedding(n_users, n_factors)
        self.item_factors = Embedding(n_items, n_factors)
        self.user_factors.weight = xavier_normal_(self.user_factors.weight)
        self.item_factors.weight = xavier_normal_(self.item_factors.weight)

        self.user_biases = Embedding(n_users, 1)
        self.item_biases = Embedding(n_items, 1)

        self._min_rating = min_rating
        self._max_rating = max_rating

        self._data_encoder = data_encoder
        self._data_processor = data_processor

    def forward(self, users: Tensor, items: Tensor) -> Tensor:
        predictions = self.user_biases(users) + self.item_biases(items)
        predictions += torch.sum(
            (self.user_factors(users) * self.item_factors(items)), dim=1, keepdim=True
        )

        histograms_mass = torch.empty((len(users), 1))
        for i, (user, item, prediction) in enumerate(zip(users, items, predictions)):
            user_id = self._data_encoder.get_original_user_id(encoded_id=user.item())
            item_id = self._data_encoder.get_original_item_id(encoded_id=item.item())

            original_ratings_by_user = torch.clone(self._data_processor.ratings_by_user[user_id])
            original_song_index = self._data_processor.item_to_index_rating[user_id][item_id]

            original_ratings_by_user[original_song_index] = prediction
            predicted_round_rating = torch.round(prediction)
            predicted_rating_index = _to_index(
                min_rating=self._min_rating,
                max_rating=self._max_rating,
                rating=predicted_round_rating,
            )

            predicted_histogram = torch.histc(
                original_ratings_by_user,
                bins=self._max_rating,
                min=self._min_rating,
                max=self._max_rating,
            )
            predicted_mass = _calc_histogram_mass(predicted_histogram, predicted_rating_index)
            histograms_mass[i] = predicted_mass

        output = torch.stack((predictions, histograms_mass), dim=1)
        # make_dot(p).view()
        return output


def _to_index(min_rating: int, max_rating: int, rating: Tensor) -> int:
    min_index = max(min_rating - 1, 0)
    return int(torch.clip(rating, min=min_index, max=max_rating - 1).item())


def _calc_histogram_mass(histogram: Tensor, end: int) -> Tensor:
    area = histogram[0:end + 1]
    if len(area) == 0:
        return Tensor([0.0])

    edge_mass = 0.5 * area[len(area) - 1]
    mass = sum(area) - edge_mass
    return mass
