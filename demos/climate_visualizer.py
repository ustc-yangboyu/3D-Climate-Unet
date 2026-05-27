from matplotlib.animation import FuncAnimation
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src.config import DEVICE
from src.dataset import ClimateDataset
from src.models import ClimateUNet

plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 100

class ClimateVisualizer:

    CHANNEL_NAMES = [
        't2m', 'skt', 'd2m', 'u10', 'v10', 'ssrd',
        'ssr', 'sp', 'stl1', 'sshf', 'swvl1', 'lmlt'
    ]

    def __init__(self, model_path: str = "checkpoints/ckpt_final.pt"):
        self.device = DEVICE
        self.dataset = ClimateDataset(
            data_path="pt_data/era5_data.pt",
            terrain_path="pt_data/terrain_data.pt",
            split="test"
        )

        self.model = ClimateUNet(
            terrain_data = self.dataset.terrain.repeat(2, 1, 1),
            static_terrain_data = self.dataset.terrain
        )
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.std = self.dataset.data.std(dim = [0, 2, 3])
        self.mean = self.dataset.data.mean(dim = [0, 2, 3])

    def predict(self, sample_idx: int = 0):
        batch = self.dataset[sample_idx]
        input_data = batch["input"].unsqueeze(0).to(self.device)
        output_data = batch["output"].unsqueeze(0)

        time_dict = {
            "month": batch["month"].unsqueeze(0).to(self.device),
            "day": batch["day"].unsqueeze(0).to(self.device),
            "hour": batch["hour"].unsqueeze(0).to(self.device)
        }

        with torch.no_grad():
            prediction = self.model(input_data, time_dict)

        return {
            "input": input_data[0].cpu(),
            "prediction": prediction[0].cpu(),
            "actual": output_data[0],
            "terrain": batch["terrain"],
            "time": {
                "month": batch["month"].item(),
                "day": batch["day"].item(),
                "hour": batch["hour"].item()
            }
        }

    def plot_comparison(self, sample_idx: int = 0, channel: int = 0):
        result = self.predict(sample_idx)

        actual = result["actual"][channel].numpy()
        prediction = result["prediction"][channel].numpy()

        vmin = min(actual.min(), prediction.min())
        vmax = max(actual.max(), prediction.max())

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(
            f"Climate Prediction (Sample {sample_idx}, Channel: {self.CHANNEL_NAMES[channel]})",
            fontsize=14, fontweight='bold'
        )

        for t in range(4):
            axes[0, t].imshow(actual[t], cmap='coolwarm', vmin=vmin, vmax=vmax)
            axes[0, t].set_title(f'Actual t={t*3}h')
            axes[0, t].axis('off')

        for t in range(4):
            axes[1, t].imshow(prediction[t], cmap='coolwarm', vmin=vmin, vmax=vmax)
            axes[1, t].set_title(f'Predicted t={t*3}h')
            axes[1, t].axis('off')

        plt.tight_layout()
        return fig

    def plot_terrain_overlay(self, sample_idx: int = 0, channel: int = 0):
        result = self.predict(sample_idx)

        terrain = result["terrain"].numpy()
        actual = result["actual"][channel].numpy()

        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        fig.suptitle(f"Terrain Overlay (Sample {sample_idx})", fontsize=14, fontweight='bold')

        time_steps = [0, 3, 6, 9, 11, 5]

        for i, t in enumerate(time_steps[:6]):
            row, col = i // 3, i % 3
            im = axes[row, col].imshow(actual[t], cmap='coolwarm', alpha=0.7)
            axes[row, col].imshow(terrain, cmap='Greys', alpha=0.3)
            axes[row, col].set_title(f'Time Step {t}')
            axes[row, col].axis('off')
            plt.colorbar(im, ax=axes[row, col], shrink=0.6)

        plt.tight_layout()
        return fig

    def create_animation(self, sample_idx: int = 0, channel: int = 0):
        result = self.predict(sample_idx)

        actual = result["actual"][channel].numpy()
        prediction = result["prediction"][channel].numpy()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Time Evolution (Sample {sample_idx})", fontsize=14, fontweight='bold')

        vmin = min(actual.min(), prediction.min())
        vmax = max(actual.max(), prediction.max())

        ims = [
            axes[0].imshow(actual[0], cmap='coolwarm', vmin=vmin, vmax=vmax),
            axes[1].imshow(prediction[0], cmap='coolwarm', vmin=vmin, vmax=vmax)
        ]

        titles = ['Actual', 'Predicted']
        for ax, title, _ in zip(axes, titles, ims):
            ax.set_title(f'{title} t=0')
            ax.axis('off')

        def update(frame):
            t = frame % 12
            ims[0].set_array(actual[t])
            ims[1].set_array(prediction[t])
            for ax, title, _ in zip(axes, titles, ims):
                ax.set_title(f'{title} t={t}')
            return ims

        ani = FuncAnimation(fig, update, frames=12, interval=500, blit=False)
        return fig, ani


def main():
    visualizer = ClimateVisualizer()
    os.makedirs('demos', exist_ok=True)

    fig = visualizer.plot_comparison(sample_idx = 0, channel=0)
    fig.savefig('demos/comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    fig = visualizer.plot_terrain_overlay(sample_idx = 0, channel=0)
    fig.savefig('demos/terrain_overlay.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    fig, ani = visualizer.create_animation(sample_idx = 0, channel=0)
    ani.save('demos/prediction_animation.gif', writer='pillow', fps=2, dpi=100)
    plt.close(fig)

if __name__ == "__main__":
    import os
    main()