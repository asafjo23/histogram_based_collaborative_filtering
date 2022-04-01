import torch
import torch.nn.functional as F


class Loss:
    def __init__(self):
        super(Loss, self).__init__()

    @staticmethod
    def mse_loss(
        original_ratings: torch.Tensor,
        predicted_ratings: torch.Tensor,
    ) -> torch.Tensor:
        return F.mse_loss(original_ratings, predicted_ratings)

    @staticmethod
    def histogram_loss(
        original_mass: torch.Tensor,
        predicted_mass: torch.Tensor,
        total_mass: torch.Tensor,
    ) -> torch.Tensor:

        histogram_loss = (
            torch.divide(torch.abs(torch.sub(original_mass, predicted_mass)), total_mass)
            .sum()
            .squeeze()
        )

        # writer.add_scalars(
        #     f"Loss/train/histogram_mass/{user_id}",
        #     {"original_mass": original_mass.item(), "predicted_mass": predicted_mass.item(),},
        #     epoch,
        # )
        #
        # writer.add_histogram(
        #     tag=f"{user_id}/predicted_histogram", values=original_histogram, global_step=epoch,
        # )

        return histogram_loss
