"""Estimate total wall-clock training time for LLM training."""

from dataclasses import dataclass


@dataclass
class ExperimentWallTime:
    """Estimates total wall-clock training time for LLM training.

    This model provides a framework for analyzing the efficiency of different
    training strategies by considering both computation and communication costs.
    It includes an overlap factor to model scenarios where communication can be
    partially or fully hidden by computation.

    The formulas used are based on established models for training cost analysis,
    such as those described in "Training Compute-Optimal Large Language Models"
    by Hoffmann et al. (the "Chinchilla" paper) and "Scaling Laws for Neural
    Language Models" by Kaplan et al.
    """

    # --- Model and Dataset Parameters ---
    dataset_size: int  # D: Total number of tokens to be processed in the dataset.
    n_model_parameters: int  # d: Number of trainable parameters in the model.

    # --- Hardware and System Parameters ---
    n_workers: int  # M: Number of parallel compute units (e.g., GPUs).
    worker_flops_per_second: float  # S: Theoretical peak FLOPS per worker.
    worker_mfu: float  # MFU: Model FLOPS Utilization, efficiency of GPUs in [0,1].

    # --- Communication Parameters ---
    equivalent_communication_steps: int  # Total number of communication rounds.
    p2p_network_latency: float  # l: Network latency per communication round (seconds).

    # --- Training Configuration ---
    precision: str  # Precision of model parameters (e.g., "fp16", "bf16", "fp32").
    overlap_factor: float = 0.0  # Communication-computation overlap factor in [0,1].

    def precision_to_bits(self) -> int:
        """Convert precision string to number of bits.

        Returns
        -------
        int
            Number of bits for the given precision (e.g., 16 for "fp16").

        Raises
        ------
        ValueError
            If the precision is not supported.

        """
        if self.precision in {"fp16", "bf16"}:
            return 16
        if self.precision == "fp32":
            return 32
        msg = f"Unsupported precision: {self.precision}"
        raise ValueError(msg)

    def compute_time(self) -> float:
        """Estimate the compute time for training (in seconds).

        The formula is based on the total number of FLOPs required for training.
        Total FLOPs are estimated as 6 * d * D, where:
        - d is the number of model parameters.
        - D is the total number of tokens in the dataset.
        The factor of 6 is a common rule of thumb for training large language models,
        accounting for both forward and backward passes.

        Returns
        -------
        float
            Estimated compute time in seconds.

        """
        # Total FLOPs for training: C = 6 * d * D
        total_flops = 6 * self.n_model_parameters * self.dataset_size

        if self.n_workers == 0:
            return float("inf")

        # Total effective FLOPS across all workers: S_eff = MFU * S * M
        total_effective_flops_per_second = (
            self.worker_mfu * self.worker_flops_per_second * self.n_workers
        )

        if total_effective_flops_per_second == 0:
            return float("inf")

        # Compute time: t_compute = C / S_eff
        return total_flops / total_effective_flops_per_second

    def _communication_time_per_round(self, p2p_bandwidth_bps: float) -> float:
        """Calculate the time for a single communication round.

        This is based on a ring-all-reduce communication pattern.

        Parameters
        ----------
        p2p_bandwidth_bps : float
            Effective peer-to-peer bandwidth in bits per second.

        Returns
        -------
        float
            Time for one communication round in seconds.

        """
        if self.n_workers <= 1:
            return 0.0  # No communication between workers

        bits_per_parameter = self.precision_to_bits()
        if p2p_bandwidth_bps == 0:
            return float("inf")
        p2p_bandwidth_params_per_sec = p2p_bandwidth_bps / bits_per_parameter

        # Time to transfer all parameters: 2 * (M-1)/M * (d / B_params)
        # The formula used here is 2 * d / B_params * (1 - 1/M) which is equivalent.
        # It represents the time for all workers to exchange their parameters.
        communication_time = (
            (2 * self.n_model_parameters / p2p_bandwidth_params_per_sec)
            * (1 - 1 / self.n_workers)
        )

        # Add latency
        return communication_time + self.p2p_network_latency

    def total_time(self, p2p_bandwidth_bps: float) -> float:
        """Estimate the total wall-clock time for training (in seconds).

        This method combines compute time and communication time, accounting for
        potential overlap between them.

        Parameters
        ----------
        p2p_bandwidth_bps : float
            Effective peer-to-peer bandwidth in bits per second.

        Returns
        -------
        float
            Total estimated training time in seconds.

        """
        compute_time_val = self.compute_time()

        if self.n_workers <= 1:
            return compute_time_val

        comm_time_per_round = self._communication_time_per_round(p2p_bandwidth_bps)
        total_comm_time = self.equivalent_communication_steps * comm_time_per_round

        # Decompose total communication time into overlappable and non-overlappable.
        overlappable_comm_time = total_comm_time * self.overlap_factor
        non_overlappable_comm_time = total_comm_time * (1 - self.overlap_factor)

        # The overlappable part can be hidden by computation.
        # Calculate the part of overlappable communication that is not hidden.
        unhidden_overlappable_time = max(0, overlappable_comm_time - compute_time_val)

        # Total wall-clock time includes compute time, non-overlappable communication,
        # and the part of overlappable communication that couldn't be hidden.
        return (
            compute_time_val
            + non_overlappable_comm_time
            + unhidden_overlappable_time
        )
