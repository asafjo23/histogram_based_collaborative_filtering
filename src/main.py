import torch
import os
from pandas import read_csv
from torch.optim import SGD
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from config import DATA_DIR, MODELS_DIR
from src.loss import Loss
from src.model import HistogramMF
from src.runner import Runner
from src.create_dataset import create_dataset, create_histogram_features
from src.data_processor import DataProcessor
from src.data_encoder import DataEncoder

DF_PATH = f"{DATA_DIR}/MovieLens/ratings.csv"

if __name__ == "__main__":
    columns = ["userId", "movieId", "rating"]
    original_df = read_csv(DF_PATH, skipinitialspace=True, usecols=columns, nrows=500000)
    original_df.columns = ["user_id", "item_id", "rating"]
    original_df = original_df.astype({"user_id": "int32"})
    original_df = original_df.astype({"item_id": "int32"})
    original_df = original_df.astype({"rating": "int32"})
    create_histogram_features(data_frame=original_df)

