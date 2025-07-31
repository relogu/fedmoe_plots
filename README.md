# FedMoE Plots

A comprehensive Python package for plotting utilities and data analysis in the Federated Mixture of Experts (FedMoE) project. This package provides tools for analyzing experimental results, creating publication-quality plots, and modeling training performance characteristics.

## 🔧 Features

- **Data Analysis**: Utilities for processing and analyzing experimental data from federated learning experiments
- **Plotting Utilities**: Publication-quality plotting functions with consistent styling and formatting
- **Parameter Counting**: Tools for computing various parameter metrics for dense and Mixture of Experts (MoE) models
- **Wall-Clock Time Modeling**: Framework for analyzing training efficiency and communication costs
- **Weights & Biases Integration**: Utilities for working with wandb runs and experiment tracking
- **Jupyter Notebook Support**: Comprehensive notebook examples for various plotting scenarios

## 📦 Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. Make sure you have uv installed, then:

```bash
# Clone the repository
git clone https://github.com/relogu/fedmoe_plots.git
cd fedmoe_plots

# Install dependencies
uv sync
```

## 🚀 Quick Start

### Basic Usage

```python
import fedmoe_plots
from fedmoe_plots import plotting_utils, data_analysis, wandb_utils

# Configure logging for Jupyter notebooks
plotting_utils.configure_logging_for_jupyter()

# Load and analyze experimental data
df = wandb_utils.load_wandb_data("your-wandb-run-id")
metrics = data_analysis.compute_metrics(df)

# Create plots
fig, ax = plotting_utils.create_publication_plot()
# ... your plotting code
```

### Running Scripts

When executing any Python scripts, use the local environment with uv:

```bash
uv run python your_script.py
```

## 📊 Modules

### `data_analysis.py`
- Process experimental data from federated learning runs
- Compute throughput metrics and performance statistics
- Handle missing data and column validation
- Extract training perplexity and other key metrics

### `plotting_utils.py`
- Publication-quality matplotlib configuration
- Consistent styling and formatting functions
- Jupyter notebook logging configuration
- Custom plot types for ML experiments

### `parameter_counting.py`
- Compute parameter counts for dense and MoE models
- Calculate trainable, expert, and embedding parameters
- Support for various model architectures
- YAML configuration parsing

### `wall_time_model.py`
- Model training time estimation
- Communication and computation cost analysis
- Overlap factor modeling for hidden communication
- Hardware utilization calculations

### `wandb_utils.py`
- Interface with Weights & Biases API
- Load and process experimental runs
- Handle server and client run coordination
- Data extraction and filtering utilities

## 📁 Project Structure

```
fedmoe_plots/
├── fedmoe_plots/           # Main package
│   ├── __init__.py
│   ├── data_analysis.py    # Data processing utilities
│   ├── parameter_counting.py # Model parameter calculations
│   ├── plotting_utils.py   # Plotting and visualization
│   ├── wall_time_model.py  # Training time modeling
│   ├── wandb_utils.py      # Weights & Biases integration
│   └── notebooks/          # Jupyter notebook examples
│       ├── arxiv_plots.ipynb
│       ├── dept_plots.ipynb
│       ├── experts_density.ipynb
│       ├── kernel_throughput_comparison.ipynb
│       ├── mclr_plots.ipynb
│       ├── mlsys_plots.ipynb
│       ├── mup_completep_scaling.ipynb
│       ├── optimizer_plots.ipynb
│       └── wall_clock_time_model.ipynb
├── tests/                  # Test suite
│   ├── demo_overlap_factor.py
│   ├── test_bold_all_ticks.py
│   ├── test_column_names.py
│   └── ...
├── pyproject.toml         # Project configuration
├── uv.lock               # Dependency lock file
└── README.md             # This file
```

## 🧪 Testing

Tests are located in the `tests/` directory. Run tests using:

```bash
uv run python -m pytest tests/
```

Individual test files can be run directly:

```bash
uv run python tests/test_bold_all_ticks.py
```

## 📓 Jupyter Notebooks

The `notebooks/` directory contains comprehensive examples:

- **`experts_density.ipynb`**: Expert utilization analysis
- **`mup_completep_scaling.ipynb`**: Scaling law analysis
- **`wall_clock_time_model.ipynb`**: Training time modeling

## 🔧 Development

### Code Style

This project uses:
- **Black** for code formatting (line length: 88)
- **isort** for import sorting
- **mypy** for type checking

Format code:
```bash
uv run black fedmoe_plots/
uv run isort fedmoe_plots/
```

### Dependencies

Key dependencies include:
- `matplotlib >= 3.10.3` - Plotting and visualization
- `pandas >= 2.3.1` - Data manipulation
- `numpy >= 2.2.6` - Numerical computing
- `seaborn >= 0.13.2` - Statistical plotting
- `wandb >= 0.21.0` - Experiment tracking
- `jupyter` ecosystem - Notebook support

## 📄 License

This project is licensed under the Apache License 2.0. See the `LICENSE` file for details.

## 👨‍💻 Author

**Lorenzo Sani** - [ls985@cam.ac.uk](mailto:ls985@cam.ac.uk)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Run the test suite (`uv run python -m pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 📚 Citation

If you use this package in your research, please cite:

```bibtex
@software{fedmoe_plots,
  author = {Sani, Lorenzo},
  title = {FedMoE Plots: Plotting utilities for Federated Mixture of Experts experiments},
  url = {https://github.com/relogu/fedmoe_plots},
  version = {0.0.1},
  year = {2025}
}
```

---

For more information about the FedMoE project and related research, please refer to the accompanying academic publications and documentation.