from typing import Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
from .metric_utils import RunningTrainMetrics, concat_results, get_metrics_dict
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader
from .visualizations import epoch_metrics_lineplot, training_mae_numexamples_lineplot


class RecSysModel(nn.Module):
    def __init__(self, df: pd.DataFrame):
        super().__init__()

        self.user_encoder = LabelEncoder()
        self.movie_encoder = LabelEncoder()

        self.user_encoder.fit_transform(df["userId"].values)
        self.movie_encoder.fit_transform(df["movieId"].values)

        n_users = len(self.user_encoder.classes_)
        n_movies = len(self.movie_encoder.classes_)

        self.user_embed = nn.Embedding(n_users, 32)
        self.movie_embed = nn.Embedding(n_movies, 32)

        self.out = nn.Linear(64, 1)

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df["userIdEncoded"] = self.user_encoder.transform(df["userId"].values)
        df["movieIdEncoded"] = self.movie_encoder.transform(df["movieId"].values)
        return df

    def forward(self, users, movies):  # , ratings=None
        user_embeds = self.user_embed(users)
        movie_embeds = self.movie_embed(movies)
        output = torch.cat([user_embeds, movie_embeds], dim=1)

        output = self.out(output)
        return output


class MLFlowRecModel(mlflow.pyfunc.PythonModel):
    def __init__(self, rec_sys_model: RecSysModel):
        self.rec_sys_model = rec_sys_model.to("cpu")

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        data = self.rec_sys_model.preprocess_data(data)
        return data

    def predict(self, context, model_input):
        preprocessed_df = self.preprocess(model_input)
        users = torch.tensor(preprocessed_df["userIdEncoded"].values)
        movies = torch.tensor(preprocessed_df["movieIdEncoded"].values)

        return self.rec_sys_model(users, movies)


def take_train_step(
    model, train_loader, loss_func, opt, running_metrics: RunningTrainMetrics
):
    model.train()
    total_train_loss = 0
    total_outputs: torch.Tensor = None
    total_ratings: torch.Tensor = None
    for _, train_data in enumerate(train_loader):
        output = model(train_data["users"], train_data["movies"])
        rating = torch.reshape(train_data["ratings"], (output.shape[0], -1)).to(
            torch.float32
        )

        total_outputs = concat_results(total_outputs, output)
        total_ratings = concat_results(total_ratings, rating)

        batch_loss = loss_func(output, rating)  # sum of losses for this batch

        total_train_loss = total_train_loss + batch_loss.sum().item()

        opt.zero_grad()
        batch_loss.backward()
        opt.step()

        running_metrics.update_train_maes(output=output, rating=rating)

    mse_loss = total_train_loss / len(
        train_loader.dataset
    )  # avg loss over all the batches

    train_metrics_dict = get_metrics_dict(total_outputs, total_ratings)

    print(f"mae loss in train step: {train_metrics_dict['mae']}")

    if not np.isclose(mse_loss, train_metrics_dict["mse"]):
        raise ValueError("Error in calculation of the training loss")


def get_train_metrics(model, train_loader):
    total_training_outputs: torch.Tensor = None
    total_training_ratings: torch.Tensor = None
    with torch.set_grad_enabled(False):
        for _, train_data in enumerate(train_loader):
            output = model(train_data["users"], train_data["movies"])
            rating = torch.reshape(train_data["ratings"], (output.shape[0], -1)).to(
                torch.float32
            )
            total_training_outputs = concat_results(total_training_outputs, output)
            total_training_ratings = concat_results(total_training_ratings, rating)

    train_metric_dict = get_metrics_dict(total_training_outputs, total_training_ratings)

    return train_metric_dict["mae"]


def get_val_metrics(model, test_loader, sch):
    total_testing_outputs: torch.Tensor = None
    total_testing_ratings: torch.Tensor = None
    with torch.set_grad_enabled(False):
        for _, test_data in enumerate(test_loader):
            output = model(test_data["users"], test_data["movies"])
            rating = torch.reshape(test_data["ratings"], (output.shape[0], -1)).to(
                torch.float32
            )
            total_testing_outputs = concat_results(total_testing_outputs, output)
            total_testing_ratings = concat_results(total_testing_ratings, rating)

    test_metric_dict = get_metrics_dict(total_testing_outputs, total_testing_ratings)
    sch.step(test_metric_dict["mse"])
    return test_metric_dict["mae"]


def train(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    model_params: dict[str, Any],
    metric_params: dict[str, Any],
):
    opt = torch.optim.Adam(model.parameters(), model_params["lr"])
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=opt, mode="min", patience=2
    )
    loss_func = nn.MSELoss(reduction="sum")

    epochs = model_params["num_epochs"]
    num_train_examples_until_record_metrics = metric_params[
        "num_train_examples_until_record_metrics"
    ]
    running_training_metrics = RunningTrainMetrics(
        num_examples_since_record=0,
        num_examples_before_record=num_train_examples_until_record_metrics,
        outputs_since_record=None,
        ratings_since_record=None,
        train_examples_num=[],
        train_maes=[],
    )
    epoch_metrics_dict = {
        "epoch_num": [],
        "test_mae": [],
        "train_mae": [],
    }
    for epoch in range(epochs):
        print(f"epoch {epoch}")
        print(f"current learning rate: {opt.param_groups[0]['lr']}")
        take_train_step(model, train_loader, loss_func, opt, running_training_metrics)
        model.eval()
        train_mae = get_train_metrics(model, train_loader)
        test_mae = get_val_metrics(model, test_loader, sch)
        epoch_metrics_dict["epoch_num"].append(epoch)
        epoch_metrics_dict["test_mae"].append(test_mae)
        epoch_metrics_dict["train_mae"].append(train_mae)
        print(epoch_metrics_dict)
    running_train_metrics_fig = training_mae_numexamples_lineplot(
        running_metrics=running_training_metrics
    )
    mlflow.log_figure(
        figure=running_train_metrics_fig, artifact_file="training_loss.png"
    )
    plt.close(fig=running_train_metrics_fig)
    epochs_metric_df = pd.DataFrame(epoch_metrics_dict)
    epochs_metrics_fig = epoch_metrics_lineplot(epochs_metric_df)
    mlflow.log_figure(figure=epochs_metrics_fig, artifact_file="epochs_metrics.png")
    plt.close(fig=epochs_metrics_fig)

    mlflow_model = MLFlowRecModel(rec_sys_model=model)

    mlflow.pyfunc.log_model(
        python_model=mlflow_model, artifact_path="model", code_paths=["code_training"]
    )
