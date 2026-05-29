# Climate Prediction Project

## Project Structure

```
PythonScientificCalculation/
├── src/                        # Core source code
│   ├── __init__.py
│   ├── config.py               # Configuration
│   ├── dataset.py              # Dataset class
│   ├── animator.py             # Visualization animator
│   └── models/                 # Neural network models
│       ├── __init__.py
│       ├── unet.py             # ClimateUNet 3D U-Net model
│       ├── attention.py        # Triple attention mechanisms
│       ├── blocks.py           # ConvBlock, EncoderBlock, DecoderBlock
│       └── embedding.py        # TimeEmbedding, TerrainEmbedding
├── scripts/                    # Training, testing & data scripts
│   ├── train.py
│   ├── test_model.py           # Model testing
│   ├── converter.py            # Raw NC data → PyTorch tensors
│   └── installer.py            # ERA5-Land data downloader
├── demos/                      # Visualization demos
│   └── climate_visualizer.py
├── checkpoints/                # Model checkpoints
|   ├── ckpt_epoch_final.pt
|   └── ckpt_epoch_1~15.pt
├── pt_data/                    # Processed data
|   ├── terrain_data.pt
|   └── era5_data.pt
├── data/                       # Raw data
└── docs/                       # Documentation
    ├── TechnicalReport.pdf
    └── arch.png
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