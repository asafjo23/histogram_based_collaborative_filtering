import torch
from pandas import Series, DataFrame
from tqdm._tqdm_notebook import tqdm_notebook

from config import DATA_DIR
from src.data_encoder import DataEncoder
from src.data_set import MFDataset


def create_dataset(data_encoder: DataEncoder) -> MFDataset:
    data_frame = data_encoder.get_encoded_dataframe()
    users_tensor = torch.LongTensor(data_frame.user_id.values)
    items_tensor = torch.LongTensor(data_frame.item_id.values)
    ratings_tensor = torch.FloatTensor(data_frame.rating.values)
    original_mass = torch.FloatTensor(data_frame.original_mass.values)
    total_mass = torch.FloatTensor(data_frame.total_mass.values)

    return MFDataset(
        users_tensor=users_tensor,
        items_tensor=items_tensor,
        ratings_tensor=ratings_tensor,
        original_mass_tensor=original_mass,
        total_mass_tensor=total_mass,
    )


def create_histogram_features(data_frame: DataFrame) -> None:
    tqdm_notebook.pandas()
    items_grouped_by_user = data_frame.groupby("user_id")
    min_rating = min(data_frame.rating.values)
    max_rating = max(data_frame.rating.values)
    data_frame["original_mass"] = data_frame.progress_apply(
        lambda row: _add_original_mass_features(
            row=row,
            group=items_grouped_by_user.get_group(row["user_id"]),
            min_rating=min_rating,
            max_rating=max_rating,
        ),
        axis=1,
    )
    data_frame["total_mass"] = data_frame.progress_apply(
        lambda row: _add_total_mass_features(
            group=items_grouped_by_user.get_group(row["user_id"]),
            min_rating=min_rating,
            max_rating=max_rating,
        ),
        axis=1,
    )
    data_frame.to_csv(f"{DATA_DIR}/BookCrossing/BX-Book-Ratings-With-Histogram_features.csv")


def _add_total_mass_features(group: Series, min_rating: int, max_rating: int) -> float:
    histogram = torch.histc(
        torch.Tensor(group.rating.values), bins=max_rating, min=min_rating, max=max_rating
    )
    total_mass = sum(histogram).item()
    return total_mass


def _add_original_mass_features(row, group: Series, min_rating: int, max_rating: int) -> float:
    histogram = torch.histc(
        torch.Tensor(group.rating.values), bins=max_rating, min=min_rating, max=max_rating
    )
    mass = _calc_histograms_tensors(histogram=histogram, end=row["rating"])
    return mass.item()


def _calc_histograms_tensors(histogram: torch.Tensor, end: int) -> torch.Tensor:
    area = histogram[0:end + 1]
    if len(area) == 0:
        return torch.Tensor([0.0])

    edge_mass = 0.5 * area[len(area) - 1]
    mass = sum(area) - edge_mass
    return mass
