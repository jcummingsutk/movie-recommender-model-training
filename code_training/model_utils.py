from typing import Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader

from .metric_utils import RunningTrainMetrics, concat_results, get_metrics_dict
from .visualizations import epoch_metrics_lineplot, training_mae_numexamples_lineplot

# def avg_r_precision(
#     df: pd.DataFrame, model: nn.Module, device: str, relevant_rating_thresh: float
# ) -> float:
#     user_ids = df["userId"].unique()
#     r_precisions = []
#     model = model.to(device)
#     for user_id in user_ids:
#         df_this_user = df[df["userId"] == user_id]
#         movies = torch.tensor(df_this_user["movieIdEncoded"].values).to(device)
#         users = torch.tensor(df_this_user["userIdEncoded"].values).to(device)
#         print(user_id)
#         predictions: Tensor = model(users, movies)
#         predictions_np = predictions.detach().cpu().detach()
#         df_this_user["prediction"] = predictions_np
#         df_this_user["relevant"] = df["rating"] >= relevant_rating_thresh
#         num_relevant_movies = df_this_user["relevant"].sum()
#         df_this_user = df_this_user.sort_values(by="prediction").head(
#             num_relevant_movies
#         )
#         df_this_user["predicted_relevant"] = (
#             df_this_user["prediction"] >= relevant_rating_thresh
#         )
#         num_predicted_relevant = df_this_user["predicted_relevant"].sum()
#         if num_relevant_movies > 0:
#             r_precisions.append(
#                 float(num_predicted_relevant) / float(num_relevant_movies)
#             )

#     return np.mean(r_precisions)


class Preprocessor:
    def __init__(
        self,
        df: pd.DataFrame,
        user_id_col: str = "userId",
        movie_id_col: str = "movieId",
    ):
        self.user_id_col = user_id_col
        self.movie_id_col = movie_id_col

        self.user_encoder = LabelEncoder()
        self.movie_encoder = LabelEncoder()

        self.user_encoder.fit_transform(df[self.user_id_col].values)
        self.movie_encoder.fit_transform(df[self.movie_id_col].values)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df["userIdEncoded"] = self.user_encoder.transform(df["userId"])
        df["movieIdEncoded"] = self.movie_encoder.transform(df["movieId"])
        return df


class MLFlowPreprocessor(mlflow.pyfunc.PythonModel):
    def __init__(self, preprocessor: Preprocessor):
        self.preprocessor = preprocessor

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        data = self.rec_sys_model.preprocess_data(data)
        return data

    def predict(self, context, model_input):
        preprocessed_df = self.preprocessor.transform(model_input)

        return preprocessed_df


class RecSysModel(nn.Module):
    def __init__(self, n_users: int, n_movies: int):
        super().__init__()

        self.mlp_user_embed = nn.Embedding(n_users, 8)
        self.mlp_movie_embed = nn.Embedding(n_movies, 8)

        self.dropout0 = nn.Dropout(0.2)
        self.mlp_out0 = nn.Linear(16, 16)
        self.relu0 = nn.ReLU()
        self.batch_norm0 = nn.BatchNorm1d(16)

        self.dropout1 = nn.Dropout(0.2)
        self.mlp_out1 = nn.Linear(16, 16)
        self.relu1 = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm1d(16)

        self.dropout2 = nn.Dropout(0.2)
        self.mlp_out2 = nn.Linear(16, 4)
        self.relu2 = nn.ReLU()
        self.batch_norm2 = nn.BatchNorm1d(4)

        self.mlp_out3 = nn.Linear(4, 1)

        torch.nn.init.normal_(self.mlp_user_embed.weight.data, 0.0, 0.01)
        torch.nn.init.normal_(self.mlp_movie_embed.weight.data, 0.0, 0.01)
        torch.nn.init.normal_(self.mlp_out0.weight.data, 0.0, 0.01)
        torch.nn.init.normal_(self.mlp_out1.weight.data, 0.0, 0.01)
        torch.nn.init.normal_(self.mlp_out2.weight.data, 0.0, 0.01)
        torch.nn.init.normal_(self.mlp_out3.weight.data, 0.0, 0.01)

    def forward(self, users, movies):  # , ratings=None
        mlp_user_embeds = self.mlp_user_embed(users)
        mlp_movie_embeds = self.mlp_movie_embed(movies)

        mlp_out = torch.cat(
            [
                mlp_user_embeds,
                mlp_movie_embeds,
            ],
            dim=1,
        )

        mlp_out = self.dropout0(mlp_out)
        mlp_out = self.mlp_out0(mlp_out)
        mlp_out = self.relu0(mlp_out)
        mlp_out = self.batch_norm0(mlp_out)

        mlp_out = self.dropout1(mlp_out)
        mlp_out = self.mlp_out1(mlp_out)
        mlp_out = self.relu1(mlp_out)
        mlp_out = self.batch_norm1(mlp_out)

        mlp_out = self.dropout2(mlp_out)
        mlp_out = self.mlp_out2(mlp_out)
        mlp_out = self.relu2(mlp_out)
        mlp_out = self.batch_norm2(mlp_out)

        # mlp_out = F.dropout(mlp_out, p=0.2)
        mlp_out = self.mlp_out3(mlp_out)

        return mlp_out


class MLFlowRecModel(mlflow.pyfunc.PythonModel):
    def __init__(self, rec_sys_model: RecSysModel):
        self.rec_sys_model = rec_sys_model.to("cpu")

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        data = self.rec_sys_model.preprocess_data(data)
        return data

    def predict(self, context, model_input):
        users = torch.tensor(model_input["userIdEncoded"].values)
        movies = torch.tensor(model_input["movieIdEncoded"].values)
        return self.rec_sys_model(users, movies)


def take_train_step(
    model, train_loader, loss_func, opt, running_metrics: RunningTrainMetrics
):
    total_train_loss = 0
    total_outputs: torch.Tensor = None
    total_ratings: torch.Tensor = None
    for _, train_data in enumerate(train_loader):
        output = model(
            train_data["users"],
            train_data["movies"],
        )
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


def get_train_metrics(model, train_loader) -> dict[str, Any]:
    total_training_outputs: torch.Tensor = None
    total_training_ratings: torch.Tensor = None
    with torch.set_grad_enabled(False):
        for _, train_data in enumerate(train_loader):
            output = model(
                train_data["users"],
                train_data["movies"],
            )
            rating = torch.reshape(train_data["ratings"], (output.shape[0], -1)).to(
                torch.float32
            )
            total_training_outputs = concat_results(total_training_outputs, output)
            total_training_ratings = concat_results(total_training_ratings, rating)

    train_metric_dict = get_metrics_dict(total_training_outputs, total_training_ratings)

    return train_metric_dict


def get_val_metrics(model, test_loader) -> dict[str, Any]:
    total_testing_outputs: torch.Tensor = None
    total_testing_ratings: torch.Tensor = None
    with torch.set_grad_enabled(False):
        for _, test_data in enumerate(test_loader):
            output = model(
                test_data["users"],
                test_data["movies"],
            )
            rating = torch.reshape(test_data["ratings"], (output.shape[0], -1)).to(
                torch.float32
            )
            total_testing_outputs = concat_results(total_testing_outputs, output)
            total_testing_ratings = concat_results(total_testing_ratings, rating)

    test_metric_dict = get_metrics_dict(total_testing_outputs, total_testing_ratings)

    return test_metric_dict


def update_metrics(model, train_loader, test_loader, sch, epoch_metrics_dict, epoch):
    train_metric_dict = get_train_metrics(model, train_loader)
    test_metric_dict = get_val_metrics(model, test_loader)
    sch.step(test_metric_dict["mse"])
    # sch.step()
    epoch_metrics_dict["epoch_num"].append(epoch)
    epoch_metrics_dict["test_mae"].append(test_metric_dict["mae"])
    epoch_metrics_dict["train_mae"].append(train_metric_dict["mae"])
    epoch_metrics_dict["train_rmse"].append(np.sqrt(train_metric_dict["mse"]))
    epoch_metrics_dict["test_rmse"].append(np.sqrt(test_metric_dict["mse"]))
    for key, val in epoch_metrics_dict.items():
        print(key)
        print(val)
        print("----")


def log_model_and_metrics(
    running_training_metrics: RunningTrainMetrics,
    epoch_metrics_dict,
    model: nn.Module,
):
    model.eval()
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

    mlflow.log_metric("test_mae", epoch_metrics_dict["test_mae"][-1])
    mlflow.log_metric("train_mae", epoch_metrics_dict["train_mae"][-1])
    mlflow.log_metric("train_rmse", epoch_metrics_dict["train_rmse"][-1])
    mlflow.log_metric("test_rmse", epoch_metrics_dict["test_rmse"][-1])

    mlflow_model = MLFlowRecModel(rec_sys_model=model)

    mlflow.pyfunc.log_model(
        python_model=mlflow_model, artifact_path="model", code_paths=["code_training"]
    )


def train(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    model_params: dict[str, Any],
    metric_params: dict[str, Any],
):
    opt = torch.optim.Adam(
        model.parameters(),
        model_params["lr"],
    )
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=opt, mode="min", patience=5
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
    )  # after every n examples, I'll record the mae with this custom object
    epoch_metrics_dict = {
        "epoch_num": [],
        "test_mae": [],
        "train_mae": [],
        "train_rmse": [],
        "test_rmse": [],
    }  # To track the metrics per epoch
    for epoch in range(epochs):
        print(f"epoch {epoch}")
        print(f"current learning rate: {opt.param_groups[0]['lr']}")
        model.train()
        take_train_step(model, train_loader, loss_func, opt, running_training_metrics)
        model.eval()
        update_metrics(model, train_loader, test_loader, sch, epoch_metrics_dict, epoch)
    log_model_and_metrics(running_training_metrics, epoch_metrics_dict, model)
