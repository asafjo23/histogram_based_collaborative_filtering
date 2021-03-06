from tensorboardX import SummaryWriter
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchviz import make_dot
from tqdm import tqdm

from src.loss import Loss
from src.model import HistogramMF


class Runner:
    def __init__(self, model: HistogramMF, criterion: Loss, optimizer: Optimizer):
        self._model = model
        self._criterion = criterion
        self._optimizer = optimizer
        self._plot_computational_graph = True

    def train(self, train_loader: DataLoader, epoch: int, writer: SummaryWriter) -> float:
        self._model.train()
        total_epoch_loss = 0.0

        with tqdm(train_loader, position=0, leave=True, unit="batch") as tepoch:
            for users, items, original_ratings, original_mass, total_mass in tepoch:
                original_ratings = original_ratings.float()
                original_mass = original_mass.float()
                total_mass = total_mass.float()

                p = self._model(users=users, items=items)

                predicted_rating = p[:, :1].squeeze()
                predicted_mass = p[:, 1:].squeeze()
                mse_loss = self._criterion.mse_loss(
                    predicted_ratings=original_ratings, original_ratings=predicted_rating
                )

                histogram_loss = self._criterion.histogram_loss(
                    original_mass=original_mass,
                    predicted_mass=predicted_mass,
                    total_mass=total_mass,
                )

                # writer.add_scalar("Loss/train/mse_loss", mse_loss / len(users), epoch)

                loss = histogram_loss + mse_loss

                if self._plot_computational_graph:
                    make_dot(loss).view()

                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                total_epoch_loss += loss.item() / len(users)
                tepoch.set_postfix(train_loss=loss.item() / len(users))
                self._plot_computational_graph = False

        writer.add_scalar("Loss/train", total_epoch_loss, epoch)
        return total_epoch_loss
