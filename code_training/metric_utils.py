from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.nn.functional import l1_loss, mse_loss


def concat_results(
    total: torch.Tensor,
    to_add: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if total is None:
        total = to_add
    else:
        total = torch.cat([total, to_add])
    return total


def mae_metric(outputs: Tensor, ratings: Tensor) -> float:
    return l1_loss(outputs, ratings, reduction="mean").cpu().detach().item()


def mse_metric(outputs: Tensor, ratings: Tensor) -> float:
    return mse_loss(outputs, ratings, reduction="mean").cpu().detach().item()


def get_metrics_dict(outputs: Tensor, ratings: Tensor) -> dict[str, Any]:
    return {
        "mae": mae_metric(outputs, ratings),
        "mse": mse_metric(outputs, ratings),
    }


@dataclass
class RunningTrainMetrics:
    num_examples_since_record: int
    num_examples_before_record: int
    outputs_since_record: Tensor
    ratings_since_record: Tensor
    train_examples_num: list[int]
    train_maes: list[float]

    def update_train_maes(self, output: Tensor, rating: Tensor):
        # update number examples seen
        self.num_examples_since_record += output.shape[0]

        # concat the results and ratings
        self.outputs_since_record = concat_results(self.outputs_since_record, output)
        self.ratings_since_record = concat_results(self.ratings_since_record, rating)

        # if we have seen a critical number of examples, record the errors and reset the outputs and ratings since record
        if self.num_examples_since_record >= self.num_examples_before_record:
            metrics_dict = get_metrics_dict(
                self.outputs_since_record,
                self.ratings_since_record,
            )

            # If this is the first time we are logging, then no need to add something to total examples seen
            if len(self.train_examples_num) == 0:
                self.train_examples_num.append(self.num_examples_since_record)
            else:  # otherwise total examples seen = as below
                total_examples_seen = (
                    self.train_examples_num[-1] + self.num_examples_since_record
                )
                self.train_examples_num.append(total_examples_seen)

            # Record the metrics and reset the outputs, ratings, and num examples since we recorded
            self.train_maes.append(metrics_dict["mae"])

            self.outputs_since_record = None
            self.ratings_since_record = None

            self.num_examples_since_record = 0
