from datetime import datetime, timedelta

from torch.utils.data import Dataset
import torch

from src.config import TRAIN_RATIO, VAL_RATIO


class ClimateDataset(Dataset):
    def __init__(
        self,
        data_path: str = "pt_data/era5_data.pt",
        terrain_path: str = "pt_data/terrain_data.pt",
        split: str = "train"
    ):
        self.split = split
        self.data_path = data_path
        self.terrain_path = terrain_path

        self.data = torch.load(data_path)[:, :, 1:, :-1]
        self.terrain = torch.load(terrain_path).squeeze(0)[1:, :-1]

        for c in range(self.data.shape[1]):
            channel_data = self.data[:, c, :, :]
            nan_mask = torch.isnan(channel_data)
            if nan_mask.any():
                channel_mean = channel_data[~nan_mask].mean()
                self.data[:, c, :, :][nan_mask] = channel_mean

        mean = self.data.mean(dim = [0, 2, 3])
        std = self.data.std(dim = [0, 2, 3])
        self.mean = mean
        self.std = std
        self.data = (self.data - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)

        total_times = self.data.shape[0]
        total_samples = total_times - 84

        train_end = int(total_samples * TRAIN_RATIO)
        val_end = train_end + int(total_samples * VAL_RATIO)

        self.train_indices = torch.tensor(range(0, train_end, 1))
        self.train_indices = self.train_indices[torch.randperm(self.train_indices.size(0))]

        self.val_indices = torch.tensor(range(train_end, val_end, 1))
        self.val_indices = self.val_indices[torch.randperm(self.val_indices.size(0))]

        self.test_indices = torch.tensor(range(val_end, total_samples, 1))
        self.test_indices = self.test_indices[torch.randperm(self.test_indices.size(0))]

    def idx2time(self, idx: int) -> tuple[int, int, int]:
        start = datetime(2020, 1, 1)
        t = start + timedelta(hours=int(idx))
        return t.month, t.day, t.hour

    def __len__(self) -> int:
        if self.split == "train":
            return len(self.train_indices)
        elif self.split == "val":
            return len(self.val_indices)
        else:
            return len(self.test_indices)

    def __getitem__(self, idx: int) -> dict:
        if self.split == "train":
            time_idx = self.train_indices[idx]
        elif self.split == "val":
            time_idx = self.val_indices[idx]
        else:
            time_idx = self.test_indices[idx]

        input_data = self.data[time_idx:time_idx + 72]
        output_data = self.data[time_idx + 72:time_idx + 84]
        month, day, hour = self.idx2time(time_idx)

        return {
            "input": input_data.transpose(0, 1),
            "output": output_data.transpose(0, 1),
            "terrain": self.terrain,
            "month": torch.tensor(month),
            "day": torch.tensor(day),
            "hour": torch.tensor(hour)
        }


if __name__ == "__main__":
    train_ds = ClimateDataset(split="train")
    val_ds = ClimateDataset(split="val")
    test_ds = ClimateDataset(split="test")

    print(f"训练集样本数: {len(train_ds)}")
    print(f"验证集样本数: {len(val_ds)}")
    print(f"测试集样本数: {len(test_ds)}")

    train_start = train_ds.train_indices[0]
    train_end = train_ds.train_indices[-1] + 72
    val_start = val_ds.val_indices[0]
    test_start = test_ds.test_indices[0]

    print(f"时间分离 train→val: {train_end <= val_start}")
    print(f"时间分离 val→test: {val_ds.val_indices[-1] + 72 <= test_start}")

    sample = train_ds[0]
    print(f"Input shape: {sample['input'].shape}")
    print(f"Output shape: {sample['output'].shape}")
    print(f"Terrain shape: {sample['terrain'].shape}")