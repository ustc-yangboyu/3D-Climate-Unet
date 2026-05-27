# Climate Prediction Project

## Project Structure

```
PythonScientificCalculation/
├── src/                    # Core source code
│   ├── __init__.py
│   ├── config.py          # Configuration
│   ├── dataset.py         # Dataset class
│   ├── model.py           # ClimateUNet model
│   └── animator.py        # Visualization animator
├── scripts/               # Training, testing & data scripts
│   ├── train.py
│   ├── test_model.py
│   ├── converter.py       # Raw NC data → PyTorch tensors
│   └── installer.py       # ERA5-Land data downloader
├── demos/                 # Visualization demos
│   └── climate_visualizer.py
├── checkpoints/           # Model checkpoints
├── pt_data/               # Processed data
├── data/                  # Raw data
└── docs/                  # Documentation
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