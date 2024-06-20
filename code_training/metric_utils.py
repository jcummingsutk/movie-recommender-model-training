from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.nn.functional import l1_loss, mse_loss
import pandas as pd


def r_precisions_at_k(
    df: pd.DataFrame,
    relevant_rating_thresh: float,
    k: float,
) -> float:
    user_ids = df["userId"].unique()
    r_precisions = []
    nums_to_consider = []
    for user_id in user_ids:
        df_this_user: pd.DataFrame = df[df["userId"] == user_id].copy()
        df_this_user["relevant"] = (df["rating"] >= relevant_rating_thresh).astype(int)

        num_relevant_movies = df_this_user["relevant"].sum()
        num_to_consider = min(num_relevant_movies, k)

        df_this_user_top_recs = df_this_user.sort_values(
            by="prediction", ascending=False
        ).head(num_to_consider)

        df_this_user_top_recs["predicted_relevant"] = (
            df_this_user_top_recs["prediction"] >= relevant_rating_thresh
        ).astype(int)
        relevant_and_recommended = df_this_user_top_recs["relevant"].sum()
        nums_to_consider.append(num_to_consider)
        if num_to_consider > 0:
            r_precisions.append(relevant_and_recommended / num_to_consider)
    return r_precisions, nums_to_consider


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


def avg_r_precision(
    df: pd.DataFrame, model, device: str, relevant_thresh: float
) -> float:
    user_ids = df["userId"].unique()
    for user_id in user_ids:
        df_this_user = df[df["userId"] == user_id]
        df_this_user = df_this_user["rating"] >= relevant_thresh
    return


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
