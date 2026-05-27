# Climate Prediction Project

## Project Structure

```
PythonScientificCalculation/
├── src/                    # Core source code
│   ├── __init__.py
│   ├── config.py          # Configuration
│   ├── dataset.py         # Dataset class
│   ├── model.py           # ClimateUNet model
│   ├── Show.py            # Visualization animator
│   └── visualize.py        # Utilities
├── scripts/               # Training & testing scripts
│   ├── train.py
│   └── test_model.py
├── demos/                 # Visualization demos
│   └── climate_visualizer.py
├── notebooks/             # Jupyter notebooks
│   └── train.ipynb
├── checkpoints/           # Model checkpoints
│   ├── model_final.pt
│   └── optimizer_final.pt
├── pt_data/              # Processed data
├── data/                 # Raw data
├── configs/              # Additional configs
├── outputs/              # Output files
└── converter.py, installer.py  # Data processing utilities
```

## Quick Start

```bash
# Train model
python scripts/train.py

# Test model
python scripts/test_model.py

# Run visualization demo
python demos/climate_visualizer.py
```