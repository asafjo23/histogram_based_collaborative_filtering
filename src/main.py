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

DF_PATH = f"{DATA_DIR}/BookCrossing/BX-Book-Ratings-With-Histogram_features.csv"

if __name__ == "__main__":
    columns = ["user_id", "item_id", "rating", "original_mass", "total_mass"]
    original_df = read_csv(DF_PATH, skipinitialspace=True, names=columns)
    original_df = original_df.iloc[1:, :]

    original_df = original_df.astype({"user_id": "int32"})
    original_df = original_df.astype({"item_id": str})
    original_df = original_df.astype({"rating": "int32"})
    original_df = original_df.astype({"original_mass": "float64"})
    original_df = original_df.astype({"total_mass": "float64"})

    data_encoder = DataEncoder(original_df=original_df)
    data_processor = DataProcessor(original_df=original_df)

    n_users = original_df.user_id.nunique()
    n_items = original_df.item_id.nunique()

    min_rating = min(original_df.rating.values)
    max_rating = max(original_df.rating.values)

    model = HistogramMF(
        n_users=n_users,
        n_items=n_items,
        data_encoder=data_encoder,
        data_processor=data_processor,
        min_rating=min_rating,
        max_rating=max_rating,
    )

    if os.path.exists(f"{MODELS_DIR}/book_crossing/model.pt"):
        model.load_state_dict(torch.load(f"{MODELS_DIR}/book_crossing/model.pt"))
    else:
        epochs = 10

        criterion = Loss()
        optimizer = SGD(model.parameters(), lr=5, weight_decay=1e-7)
        runner = Runner(model=model, criterion=criterion, optimizer=optimizer)

        train_set = create_dataset(data_encoder=data_encoder)
        train_load = DataLoader(train_set, batch_size=1000, shuffle=True)

        with SummaryWriter(f"runs/book_crossing/dev") as writer:
            for epoch in range(epochs):
                epoch_loss = runner.train(train_loader=train_load, epoch=epoch, writer=writer)
                print(f"epoch={epoch + 1}, loss={epoch_loss}")

        torch.save(model.state_dict(), f"{MODELS_DIR}/book_crossing/model.pt")
