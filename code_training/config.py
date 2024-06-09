import os
from typing import Any

import yaml


def get_dev_db_params(
    config_file: str = "config.yaml", config_secret: str = None
) -> dict[str, Any]:
    with open(config_file, "r") as fp_config:
        config = yaml.safe_load(fp_config)
    db_params_dict = config["database"]["dev"]

    if config_secret is not None:
        with open(config_secret, "r") as fp_secret_config:
            config_secret_dict = yaml.safe_load(fp_secret_config)
        password = config_secret_dict["database"]["dev"]["password"]
    else:
        password = os.environ["DATABASE_PASSWORD"]
    db_params_dict["password"] = password

    return db_params_dict


def load_azure_service_principal_environment_vars(
    config_file: str = "config.yaml", config_secret_filename: str = None
):
    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)
    os.environ["AZURE_TENANT_ID"] = config_dict["azure"]["AZURE_TENANT_ID"]
    os.environ["AZURE_CLIENT_ID"] = config_dict["azure"]["AZURE_CLIENT_ID"]

    if os.path.exists(config_secret_filename):
        with open(config_secret_filename, "r") as fp:
            config_secret_dict = yaml.safe_load(fp)
        os.environ["AZURE_CLIENT_SECRET"] = config_secret_dict["azure"][
            "AZURE_CLIENT_SECRET"
        ]


def get_config_dict(config_file: str = "config.yaml") -> str:
    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)
    return config_dict


def get_config_secret_dict(config_secrets_file: str = "config_secret.yaml") -> str:
    with open(config_secrets_file, "r") as f:
        config_secret_dict = yaml.safe_load(f)
    return config_secret_dict
