"""Utility functions for working with Weights & Biases (wandb) runs."""

import logging
from collections.abc import Callable
from typing import Any

import pandas as pd
import wandb

log = logging.getLogger(__name__)


class ServerRunNotFoundError(Exception):
    """Exception raised when the server run is not found in wandb."""

    def __init__(self, run_name: str) -> None:
        """Initialize the exception with the run name.

        Parameters
        ----------
        run_name : str
            The name of the run that was not found.

        """
        super().__init__(f"Server run '{run_name}' not found in wandb.")
        self.run_name = run_name


class ClientRunNotFoundError(Exception):
    """Exception raised when the client run is not found in wandb."""

    def __init__(self, run_name: str) -> None:
        """Initialize the exception with the run name.

        Parameters
        ----------
        run_name : str
            The name of the run that was not found.

        """
        super().__init__(f"Client run '{run_name}' not found in wandb.")
        self.run_name = run_name


def get_n_local_experts(config: dict[str, Any]) -> int:
    """Get the number of local experts based on the configuration.

    Parameters
    ----------
    config : dict[str, Any]
        The configuration dictionary containing the number of total clients and
        the number of experts per client.

    Returns
    -------
    int
        The number of local experts.

    """
    n_total_clients = config["fl"]["n_total_clients"]
    n_total_experts = config["llm_config"]["model"]["ffn_config"]["ff_n_experts"]
    overlapping_factor = config["fl"]["experts_overlapping_factor"]
    return (n_total_experts * overlapping_factor) // n_total_clients


def get_experts_global_batch_size(config: dict[str, Any]) -> int:
    """Get the global batch size for experts based on the configuration.

    Parameters
    ----------
    config : dict[str, Any]
        The configuration dictionary containing the overlapping factor and the
        local batch size.

    Returns
    -------
    int
        The global batch size for experts.

    """
    overlapping_factor = config["fl"]["experts_overlapping_factor"]
    local_batch_size = config["llm_config"]["global_train_batch_size"]
    return overlapping_factor * local_batch_size


def get_non_experts_global_batch_size(config: dict[str, Any]) -> int:
    """Get the global batch size for non-experts based on the configuration.

    Parameters
    ----------
    config : dict[str, Any]
        The configuration dictionary containing the number of total clients and
        the number of experts per client.

    Returns
    -------
    int
        The global batch size for non-experts.

    """
    n_total_clients = config["fl"]["n_total_clients"]
    local_batch_size = config["llm_config"]["global_train_batch_size"]
    return n_total_clients * local_batch_size


def remove_runs_by_regex(
    base_name: str,
    regex: str,
) -> None:
    """Remove wandb runs that match a given regex pattern.

    Parameters
    ----------
    base_name : str
        The base name of the wandb project (e.g. "team_name/project_name").
    regex : str
        The regex pattern to match the runs to be removed.

    """
    api = wandb.Api(timeout=10000)
    runs = api.runs(
        path=f"{base_name}",
        filters={"display_name": {"$regex": f"{regex}"}},
    )
    for run in runs:
        if run.state != "running":
            log.info("Removing run %s with display name %s.", run.id, run.display_name)
            run.delete()


def get_run_uuid_from_config(
    run: wandb.apis.public.Run,  # pyright: ignore[reportAttributeAccessIssue]
) -> str | None:
    """Get the run UUID from the wandb run configuration.

    Parameters
    ----------
    run : wandb.apis.public.Run
        The wandb run object.

    Returns
    -------
    str | None
        The UUID of the run if it exists in the configuration, otherwise None.

    """
    return run.config.get("run_uuid", None)


def get_clientrun_property_from_config(
    run: wandb.apis.public.Run,  # pyright: ignore[reportAttributeAccessIssue]
    get_property_fn: Callable[[dict[str, Any]], Any],
) -> Any | None:  # noqa: ANN401
    """Get the run UUID from the wandb run configuration.

    Parameters
    ----------
    run : wandb.apis.public.Run
        The wandb run object.
    get_property_fn : Callable[[dict[str, Any]], Any]
        A function that takes the run configuration dictionary and returns the desired
        property.

    Returns
    -------
    Any | None
        The value of the property if it exists in the configuration, otherwise None.

    """
    return get_property_fn(run.config) if run.config else None


def add_run_uuid_to_config(team_name: str, project_name: str, run_uuid: str) -> None:
    """Add the run UUID to the wandb run configuration.

    Parameters
    ----------
    team_name : str
        The name of the team in wandb.
    project_name : str
        The name of the project in wandb.
    run_uuid : str
        The UUID of the run to be added to the configuration and to be used for
        filtering runs.

    """
    api = wandb.Api(timeout=10000)
    runs = api.runs(
        path=f"{team_name}/{project_name}",
        filters={"display_name": {"$regex": f"^{run_uuid}*"}},
    )
    for run in runs:
        run.config["run_uuid"] = run_uuid
        run.update()


def set_peri_ln_to_config(
    run: wandb.apis.public.Run,  # pyright: ignore[reportAttributeAccessIssue]
    *,
    peri_ln: bool,
) -> None:
    """Set the peri-norm setting to the wandb run configuration.

    Parameters
    ----------
    run : wandb.apis.public.Run
        The wandb run object.
    peri_ln : bool
        Whether to use peri-norm in the model configuration.

    """
    run.config["llm_config"]["model"]["use_peri_norm"] = peri_ln
    run.update()


def set_embedding_ln_to_config(
    run: wandb.apis.public.Run,  # pyright: ignore[reportAttributeAccessIssue]
    *,
    embedding_ln: bool,
) -> None:
    """Set the embedding layer normalization setting to the wandb run configuration.

    Parameters
    ----------
    run : wandb.apis.public.Run
        The wandb run object.
    embedding_ln : bool
        Whether to use layer normalization in the embedding layer.

    """
    run.config["llm_config"]["model"]["use_embedding_norm"] = embedding_ln
    run.update()


def change_run_uuid(
    team_name: str,
    project_name: str,
    old_run_uuid: str,
    new_run_uuid: str,
) -> None:
    """Change the run UUID in the wandb run configuration.

    Parameters
    ----------
    team_name : str
        The name of the team in wandb.
    project_name : str
        The name of the project in wandb.
    old_run_uuid : str
        The old UUID of the run to be changed and to be used for filtering runs.
    new_run_uuid : str
        The new UUID to be set in the configuration.

    """
    api = wandb.Api(timeout=10000)
    runs = api.runs(
        path=f"{team_name}/{project_name}",
        filters={"display_name": {"$regex": f"^{old_run_uuid}*"}},
    )
    for run in runs:
        run.config["run_uuid"] = new_run_uuid
        run.update()


def download_wandb_whole_history(
    run: wandb.apis.public.Run,  # pyright: ignore[reportAttributeAccessIssue]
) -> pd.DataFrame:
    """Download the entire history of a wandb run.

    Parameters
    ----------
    run : wandb.apis.public.Run
        The wandb run object.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the entire history of the wandb run.

    """
    history = run.scan_history()
    return pd.DataFrame(history)


def download_photon_metrics(
    base_name: str,
    run_uuid: str,
    *,
    timeout: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download Photon server and client metrics from wandb.

    Parameters
    ----------
    base_name : str
        The base name of the wandb project (e.g. "team_name/project_name").
    run_uuid : str
        The UUID of the run to be downloaded.
    timeout : int, optional
        The timeout for the wandb API requests, by default 100 seconds.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        A tuple containing two pandas DataFrames:
        - The first DataFrame contains the server metrics.
        - The second DataFrame contains the client metrics.

    Raises
    ------
    ServerRunNotFoundError
        If the server run is not found in wandb.
    ClientRunNotFoundError
        If no client runs are found for the given run UUID.

    """
    # Open WandB API
    api = wandb.Api(timeout=timeout)
    # Download Photon server side metrics
    server_run_name = f"{base_name}/{run_uuid}"
    server_run = api.run(server_run_name)
    if server_run is None:
        log.error(
            "Server run %s not found, trying %s_server.",
            server_run_name,
            server_run_name,
        )
        server_run_name = f"{base_name}/{run_uuid}_server"
        server_run = api.run(server_run_name)
    if server_run is None:
        raise ServerRunNotFoundError(run_uuid)
    # Get the number of clients recorded by using regex
    client_runs = api.runs(
        path=f"{base_name}",
        filters={"display_name": {"$regex": f"^{run_uuid}_client_"}},
    )
    n_clients = len(client_runs)
    log.debug("Found %s clients for run %s.", n_clients, run_uuid)
    if n_clients == 0:
        raise ClientRunNotFoundError(run_uuid)
    # Download Photon clients side metrics
    clients_metrics_df_list: list[pd.DataFrame] = []
    for client_run in client_runs:
        log.debug(
            "Downloading client metrics for run %s with display name %s.",
            client_run.id,
            client_run.display_name,
        )
        client_metrics_df = download_wandb_whole_history(client_run)
        # Add the `client_id` column based on the current client_id
        client_metrics_df["client_id"] = client_run.display_name.split("client_")[-1]
        # Append the client metrics to the list
        clients_metrics_df_list.append(client_metrics_df)
    # Return the Server and Clients metrics data frames
    return download_wandb_whole_history(server_run), (
        pd.concat(clients_metrics_df_list)
        if clients_metrics_df_list
        else pd.DataFrame()
    )
