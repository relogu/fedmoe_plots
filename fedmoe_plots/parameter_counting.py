"""Parameter counting utilities for model configurations.

This module provides functions to compute various parameter counts from YAML model
configurations, including trainable parameters, expert parameters, embedding parameters,
and more. It supports both dense models and mixture of experts (MoE) models.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class ParameterCounts:
    """Container for different parameter count metrics.

    Attributes
    ----------
    n_trainable : int
        Total number of trainable parameters (N_t).
    n_non_experts : int
        Number of non-expert parameters (N_ne).
    n_experts : int
        Number of expert parameters (N_e).
    n_non_embedding : int
        Number of non-embedding parameters (N_nemb).
    n_embedding : int
        Number of embedding parameters (N_emb).

    """

    n_trainable: int
    n_non_experts: int
    n_experts: int
    n_non_embedding: int
    n_embedding: int

    def __post_init__(self) -> None:
        """Validate parameter count relationships.

        Raises
        ------
        ValueError
            If the computed parameter counts are inconsistent.

        """
        # Verify that the counts are consistent
        if self.n_trainable != self.n_non_experts + self.n_experts:
            msg = (
                f"Inconsistent parameter counts: "
                f"n_trainable ({self.n_trainable}) != "
                f"n_non_experts ({self.n_non_experts}) + n_experts ({self.n_experts})"
            )
            raise ValueError(msg)

        if self.n_trainable != self.n_non_embedding + self.n_embedding:
            msg = (
                f"Inconsistent parameter counts: "
                f"n_trainable ({self.n_trainable}) != "
                f"n_non_embedding ({self.n_non_embedding}) +"
                f" n_embedding ({self.n_embedding})"
            )
            raise ValueError(msg)


def _get_model_config(config: dict[str, Any]) -> dict[str, Any]:
    """Extract model configuration from full config dictionary.

    Parameters
    ----------
    config : dict[str, Any]
        Full configuration dictionary that may contain a "model" key.

    Returns
    -------
    dict[str, Any]
        Model configuration dictionary.

    Raises
    ------
    KeyError
        If no model configuration is found.

    """
    if "model" in config:
        return config["model"]
    if all(key in config for key in ["d_model", "n_layers"]):
        # Assume this is already a model config
        return config
    msg = "No model configuration found in config dictionary"
    raise KeyError(msg)


def _get_config_value(
    model_config: dict[str, Any],
    key: str,
    default: Any = None,  # noqa: ANN401
) -> Any:  # noqa: ANN401
    """Get a configuration value with optional default.

    Parameters
    ----------
    model_config : dict[str, Any]
        Model configuration dictionary.
    key : str
        Configuration key to retrieve.
    default : Any, optional
        Default value if key is not found.

    Returns
    -------
    Any
        Configuration value or default.

    Raises
    ------
    KeyError
        If key is not found and no default is provided.

    """
    if key in model_config:
        return model_config[key]
    if default is not None:
        return default
    msg = f"Required configuration key '{key}' not found"
    raise KeyError(msg)


def compute_embedding_parameters(config: dict[str, Any]) -> int:
    """Compute the number of embedding parameters (N_emb).

    Parameters
    ----------
    config : dict[str, Any]
        Model configuration dictionary.

    Returns
    -------
    int
        Number of embedding parameters.

    """
    model_config = _get_model_config(config)

    d_model = _get_config_value(model_config, "d_model")
    vocab_size = _get_config_value(model_config, "vocab_size", 50368)

    # Input embeddings: vocab_size * d_model
    # Note: Most models share input and output embeddings, so we only count once
    n_embedding = vocab_size * d_model

    log.debug("Embedding parameters: %s", n_embedding)
    return n_embedding


def compute_attention_parameters(model_config: dict[str, Any]) -> int:
    """Compute the number of attention parameters per layer.

    Parameters
    ----------
    model_config : dict[str, Any]
        Model configuration dictionary.

    Returns
    -------
    int
        Number of attention parameters per layer.

    """
    d_model = _get_config_value(model_config, "d_model")

    # Standard attention: Q, K, V projections and output projection
    # Each projection is d_model -> d_model
    # Total: 4 * d_model * d_model per layer
    attn_params_per_layer = 4 * d_model * d_model

    # Add bias if not disabled
    no_bias = _get_config_value(model_config, "no_bias", default=True)
    if not no_bias:
        # 4 bias vectors of size d_model each
        attn_params_per_layer += 4 * d_model

    log.debug("Attention parameters per layer: %s", attn_params_per_layer)
    return attn_params_per_layer


def compute_norm_parameters(model_config: dict[str, Any]) -> int:
    """Compute the number of normalization parameters per layer.

    Parameters
    ----------
    model_config : dict[str, Any]
        Model configuration dictionary.

    Returns
    -------
    int
        Number of normalization parameters per layer.

    """
    d_model = _get_config_value(model_config, "d_model")
    use_peri_norm = _get_config_value(model_config, "use_peri_norm", default=False)

    # Standard: 2 layer norms per block (pre-attention and pre-ffn)
    # Each layer norm has d_model parameters (weight only, no bias typically)
    norm_params_per_layer = 2 * d_model

    # Additional norms if peri_norm is used
    if use_peri_norm:
        # Post-attention and post-ffn norms
        norm_params_per_layer += 2 * d_model

    log.debug("Normalization parameters per layer: %s", norm_params_per_layer)
    return norm_params_per_layer


def compute_dense_ffn_parameters(model_config: dict[str, Any]) -> int:
    """Compute the number of FFN parameters for dense models.

    Parameters
    ----------
    model_config : dict[str, Any]
        Model configuration dictionary.

    Returns
    -------
    int
        Number of FFN parameters per layer for dense models.

    """
    d_model = _get_config_value(model_config, "d_model")
    ffn_hidden_size = _get_config_value(model_config, "ffn_hidden_size", 4 * d_model)
    no_bias = _get_config_value(model_config, "no_bias", default=True)

    # Standard FFN: up projection + down projection
    # Up: d_model -> ffn_hidden_size
    # Down: ffn_hidden_size -> d_model
    ffn_params = d_model * ffn_hidden_size + ffn_hidden_size * d_model

    # Add bias if not disabled
    if not no_bias:
        ffn_params += ffn_hidden_size + d_model

    log.debug("Dense FFN parameters per layer: %s", ffn_params)
    return ffn_params


def compute_sigma_moe_parameters(model_config: dict[str, Any]) -> tuple[int, int]:
    """Compute FFN parameters for SigmaMoE models.

    Parameters
    ----------
    model_config : dict[str, Any]
        Model configuration dictionary.

    Returns
    -------
    tuple[int, int]
        Tuple of (expert_parameters, non_expert_parameters) per layer.

    """
    ffn_config = _get_config_value(model_config, "ffn_config", {})

    d_model = _get_config_value(model_config, "d_model")
    ff_n_experts = _get_config_value(ffn_config, "ff_n_experts", 8)
    ff_expert_size = _get_config_value(ffn_config, "ff_expert_size", 4 * d_model)
    v_dim = _get_config_value(ffn_config, "v_dim", d_model)
    no_bias = _get_config_value(model_config, "no_bias", default=True)

    # Expert parameters: up_proj and down_proj for each expert
    # up_proj: d_model -> ff_expert_size (per expert)
    # down_proj: ff_expert_size -> v_dim (per expert)
    expert_params_per_expert = d_model * ff_expert_size + ff_expert_size * v_dim

    if not no_bias:
        expert_params_per_expert += ff_expert_size + v_dim

    total_expert_params = ff_n_experts * expert_params_per_expert

    # Non-expert parameters: expert selection mechanism
    # expert_sel: d_model -> ff_n_experts
    non_expert_params = d_model * ff_n_experts

    if not no_bias:
        non_expert_params += ff_n_experts

    log.debug("SigmaMoE expert parameters per layer: %s", total_expert_params)
    log.debug("SigmaMoE non-expert parameters per layer: %s", non_expert_params)

    return total_expert_params, non_expert_params


def compute_experts_parameters(config: dict[str, Any]) -> int:
    """Compute the number of expert parameters (N_e).

    Parameters
    ----------
    config : dict[str, Any]
        Model configuration dictionary.

    Returns
    -------
    int
        Number of expert parameters.

    """
    model_config = _get_model_config(config)
    n_layers = _get_config_value(model_config, "n_layers")

    ffn_config = _get_config_value(model_config, "ffn_config", {})
    ffn_type = _get_config_value(ffn_config, "ffn_type", "mptmlp")

    if ffn_type == "sigma_moe":
        expert_params_per_layer, _ = compute_sigma_moe_parameters(model_config)
        total_expert_params = n_layers * expert_params_per_layer
    else:
        # Dense model has no expert parameters
        total_expert_params = 0

    log.debug("Total expert parameters: %s", total_expert_params)
    return total_expert_params


def compute_non_experts_parameters(config: dict[str, Any]) -> int:
    """Compute the number of non-expert parameters (N_ne).

    Parameters
    ----------
    config : dict[str, Any]
        Model configuration dictionary.

    Returns
    -------
    int
        Number of non-expert parameters.

    """
    model_config = _get_model_config(config)
    n_layers = _get_config_value(model_config, "n_layers")
    d_model = _get_config_value(model_config, "d_model")

    # Embedding parameters
    n_embedding = compute_embedding_parameters(config)

    # Per-layer parameters
    attn_params_per_layer = compute_attention_parameters(model_config)
    norm_params_per_layer = compute_norm_parameters(model_config)

    # FFN parameters
    ffn_config = _get_config_value(model_config, "ffn_config", {})
    ffn_type = _get_config_value(ffn_config, "ffn_type", "mptmlp")

    if ffn_type == "sigma_moe":
        _, ffn_non_expert_params_per_layer = compute_sigma_moe_parameters(model_config)
    else:
        ffn_non_expert_params_per_layer = compute_dense_ffn_parameters(model_config)

    # Total per-layer non-expert parameters
    per_layer_params = (
        attn_params_per_layer + norm_params_per_layer + ffn_non_expert_params_per_layer
    )

    # Final layer norm (if present)
    final_norm_params = d_model  # Typically just weight, no bias

    total_non_expert_params = (
        n_embedding + n_layers * per_layer_params + final_norm_params
    )

    log.debug("Total non-expert parameters: %s", total_non_expert_params)
    return total_non_expert_params


def compute_non_embedding_parameters(config: dict[str, Any]) -> int:
    """Compute the number of non-embedding parameters (N_nemb).

    Parameters
    ----------
    config : dict[str, Any]
        Model configuration dictionary.

    Returns
    -------
    int
        Number of non-embedding parameters.

    """
    model_config = _get_model_config(config)
    n_layers = _get_config_value(model_config, "n_layers")
    d_model = _get_config_value(model_config, "d_model")

    # Per-layer parameters
    attn_params_per_layer = compute_attention_parameters(model_config)
    norm_params_per_layer = compute_norm_parameters(model_config)

    # FFN parameters
    ffn_config = _get_config_value(model_config, "ffn_config", {})
    ffn_type = _get_config_value(ffn_config, "ffn_type", "mptmlp")

    if ffn_type == "sigma_moe":
        expert_params_per_layer, ffn_non_expert_params_per_layer = (
            compute_sigma_moe_parameters(model_config)
        )
        ffn_params_per_layer = expert_params_per_layer + ffn_non_expert_params_per_layer
    else:
        ffn_params_per_layer = compute_dense_ffn_parameters(model_config)

    # Total per-layer parameters
    per_layer_params = (
        attn_params_per_layer + norm_params_per_layer + ffn_params_per_layer
    )

    # Final layer norm (if present)
    final_norm_params = d_model

    total_non_embedding_params = n_layers * per_layer_params + final_norm_params

    log.debug("Total non-embedding parameters: %s", total_non_embedding_params)
    return total_non_embedding_params


def compute_trainable_parameters(config: dict[str, Any]) -> int:
    """Compute the total number of trainable parameters (N_t).

    Parameters
    ----------
    config : dict[str, Any]
        Model configuration dictionary.

    Returns
    -------
    int
        Total number of trainable parameters.

    """
    n_embedding = compute_embedding_parameters(config)
    n_non_embedding = compute_non_embedding_parameters(config)

    total_trainable = n_embedding + n_non_embedding

    log.debug("Total trainable parameters: %s", total_trainable)
    return total_trainable


def compute_parameter_counts(config: dict[str, Any]) -> ParameterCounts:
    """Compute all parameter counts for a model configuration.

    Parameters
    ----------
    config : dict[str, Any]
        Model configuration dictionary.

    Returns
    -------
    ParameterCounts
        Container with all parameter count metrics.

    """
    n_trainable = compute_trainable_parameters(config)
    n_experts = compute_experts_parameters(config)
    n_non_experts = compute_non_experts_parameters(config)
    n_embedding = compute_embedding_parameters(config)
    n_non_embedding = compute_non_embedding_parameters(config)

    return ParameterCounts(
        n_trainable=n_trainable,
        n_non_experts=n_non_experts,
        n_experts=n_experts,
        n_non_embedding=n_non_embedding,
        n_embedding=n_embedding,
    )


def print_parameter_summary(
    config: dict[str, Any], title: str = "Parameter Count Summary",
) -> None:
    """Print a formatted summary of parameter counts.

    Parameters
    ----------
    config : dict[str, Any]
        Model configuration dictionary.
    title : str, optional
        Title for the summary.

    """
    counts = compute_parameter_counts(config)

    log.info("\n%s", title)
    log.info("=" * len(title))
    log.info("Total trainable parameters (N_t):     %s", counts.n_trainable)
    log.info("Non-expert parameters (N_ne):         %s", counts.n_non_experts)
    log.info("Expert parameters (N_e):              %s", counts.n_experts)
    log.info("Non-embedding parameters (N_nemb):    %s", counts.n_non_embedding)
    log.info("Embedding parameters (N_emb):         %s", counts.n_embedding)

    # Additional insights
    if counts.n_experts > 0:
        expert_ratio = counts.n_experts / counts.n_trainable * 100
        log.info("Expert parameter ratio:               %.1f%%", expert_ratio)

    embedding_ratio = counts.n_embedding / counts.n_trainable * 100
    log.info("Embedding parameter ratio:            %.1f%%", embedding_ratio)
