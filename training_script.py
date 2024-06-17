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
    create_train_test_split,
    get_dev_db_params,
    load_dataframe,
    get_movie_ids_to_include,
)
from code_training.model_utils import RecSysModel, train
import os
import json


def main(parameters_file: str, config_file: str, config_secrets_file: str):
    artifacts_dir = "./my_artfiacts"
    os.makedirs(artifacts_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    with open(parameters_file) as fp:
        parameters = yaml.safe_load(fp)
    mlflow.log_params(parameters)
    params = get_dev_db_params(
        config_file,
        config_secrets_file,
    )

    df = load_dataframe(
        params["host"],
        params["database"],
        params["user"],
        params["password"],
        params["port"],
    )

    movie_ids_to_include = get_movie_ids_to_include(df, 20)
    movie_ids_to_include = list(map(lambda x: int(x), movie_ids_to_include))
    print(
        f"{len(movie_ids_to_include)/df['movieId'].unique().shape[0]*100:.2f}% of movies have a greater than 20 ratings"
    )
    with open(os.path.join(artifacts_dir, "movie_ids.json"), "w") as fp:
        json.dump(movie_ids_to_include, fp)
    mlflow.log_artifacts("./my_artfiacts/")

    df = df[df["movieId"].isin(movie_ids_to_include)]

    model = RecSysModel(df).to(device)

    df = model.preprocess_data(df)

    print(df.info())

    df_train, df_test = create_train_test_split(df, parameters["data"]["test_size"])

    train_loader, test_loader = create_dataloaders(
        df_train, df_test, device, parameters["data"]["batch_size"]
    )

    train(
        model,
        train_loader,
        test_loader,
        model_params=parameters["model"],
        metric_params=parameters["metrics"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameters-file", type=str)
    parser.add_argument("--config-file", type=str)
    parser.add_argument("--config-secrets-file", type=str)
    parser.add_argument("--remote-tracking", type=int)
    args = parser.parse_args()

    parameters_file = args.parameters_file

    print(args.config_file)
    print(args.config_secrets_file)

    load_azure_service_principal_environment_vars(
        args.config_file, args.config_secrets_file
    )

    config_dict = get_config_dict(args.config_file)
    subscription_id = config_dict["azure"]["dev"]["SUBSCRIPTION_ID"]
    resource_group = config_dict["azure"]["dev"]["RESOURCE_GROUP"]
    workspace_name = config_dict["azure"]["dev"]["WORKSPACE_NAME"]

    cred = EnvironmentCredential()

    if args.remote_tracking == 1:

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

    mlflow.set_experiment("movie-recommender-model-training")

    with mlflow.start_run():
        main(args.parameters_file, args.config_file, args.config_secrets_file)
