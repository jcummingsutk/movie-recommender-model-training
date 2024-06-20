import argparse

import mlflow
import torch
import yaml
from azure.ai.ml import MLClient
from azure.identity import EnvironmentCredential
from code_training.config import (
    get_config_dict,
    load_azure_service_principal_environment_vars,
)
from code_training.data_utils import (
    create_dataloaders,
    get_dev_db_params,
    get_table_from_database,
)
from argparse import Namespace
from code_training.model_utils import (
    RecSysModel,
    train,
    Preprocessor,
    MLFlowPreprocessor,
)
from code_training.metric_utils import r_precisions_at_k
import numpy as np


def setup_data_and_train(
    parameters_file: str, config_file: str, config_secrets_file: str
):
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    with open(parameters_file) as fp:
        parameters = yaml.safe_load(fp)
    mlflow.log_params(parameters)
    params = get_dev_db_params(
        config_file,
        config_secrets_file,
    )
    df = get_table_from_database(**params, table="ratings")
    df_test = get_table_from_database(**params, table="ratings_test")
    df_train = get_table_from_database(**params, table="ratings_train")

    preprocessor = Preprocessor(df)
    df = preprocessor.transform(df)
    df_train = preprocessor.transform(df_train)
    df_test = preprocessor.transform(df_test)

    model = RecSysModel(
        len(preprocessor.user_encoder.classes_),
        len(preprocessor.movie_encoder.classes_),
    ).to(device)

    print(model)

    train_loader, test_loader = create_dataloaders(
        df_train, df_test, device, parameters["data"]["batch_size"]
    )

    relevant_threshold = parameters["metrics"]["relevant_thresh"]
    top_k = parameters["metrics"]["top_k"]
    train(
        model,
        train_loader,
        test_loader,
        model_params=parameters["model"],
        metric_params=parameters["metrics"],
    )
    mlflow_preprocessor = MLFlowPreprocessor(preprocessor)
    mlflow.pyfunc.log_model(
        python_model=mlflow_preprocessor,
        artifact_path="preprocessor",
        code_paths=["code_training"],
    )

    users = torch.tensor(df["userIdEncoded"].values).to("cpu")
    movies = torch.tensor(df["movieIdEncoded"].values).to("cpu")

    predictions = model(users, movies)
    predictions_np = predictions.detach().cpu().detach()
    df["prediction"] = predictions_np

    r_precisions, _ = r_precisions_at_k(df, relevant_threshold, top_k)
    mlflow.log_metric(f"r_precision_at_{top_k}", np.mean(r_precisions))


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameters-file", type=str)
    parser.add_argument("--config-file", type=str)
    parser.add_argument("--config-secrets-file", type=str)
    parser.add_argument("--remote-tracking", type=int)
    args = parser.parse_args()
    return args


def configure_remote_azure(config_file, config_secrets_file):
    load_azure_service_principal_environment_vars(config_file, config_secrets_file)

    config_dict = get_config_dict(config_file)
    subscription_id = config_dict["azure"]["dev"]["SUBSCRIPTION_ID"]
    resource_group = config_dict["azure"]["dev"]["RESOURCE_GROUP"]
    workspace_name = config_dict["azure"]["dev"]["WORKSPACE_NAME"]

    cred = EnvironmentCredential()

    ml_client = MLClient(
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name,
        credential=cred,
    )

    mlflow_tracking_uri = ml_client.workspaces.get(
        ml_client.workspace_name
    ).mlflow_tracking_uri

    mlflow.set_tracking_uri(mlflow_tracking_uri)


def main():
    args = parse_args()
    parameters_file = args.parameters_file
    config_file = args.config_file
    config_secrets_file = args.config_secrets_file
    remote_tracking = args.remote_tracking

    if remote_tracking == 1:
        configure_remote_azure(config_file, config_secrets_file)

    mlflow.set_experiment("movie-recommender-model-training")

    with mlflow.start_run():
        setup_data_and_train(parameters_file, config_file, config_secrets_file)


if __name__ == "__main__":
    main()
