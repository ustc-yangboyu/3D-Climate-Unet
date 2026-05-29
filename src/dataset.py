from datetime import datetime, timedelta
from torch.utils.data import Dataset
from torchvision import transforms
import torch


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
        self.transform = transforms.Compose([transforms.Normalize(mean, std)])
        self.data = self.transform(self.data)

        total_times = self.data.shape[0]
        test_hours = int((total_times - 84) * 0.05)
        train_end = total_times - test_hours

        self.train_indices = torch.tensor(range(0, train_end - 84, 1))
        random_indices = torch.randperm(self.train_indices.size(0))
        self.train_indices = self.train_indices[random_indices]

        self.test_indices = torch.tensor(range(train_end, total_times - 84, 1))
        random_indices = torch.randperm(self.test_indices.size(0))
        self.test_indices = self.test_indices[random_indices]

    def idx2time(self, idx: int) -> tuple[int, int, int]:
        start = datetime(2020, 1, 1)
        t = start + timedelta(hours = int(idx))
        return t.month, t.day, t.hour

    def __len__(self) -> int:
        if self.split == "train":
            return len(self.train_indices)
        else:
            return len(self.test_indices)

    def __getitem__(self, idx: int) -> dict:
        if self.split == "train":
            time_idx = self.train_indices[idx]
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
    train_ds = ClimateDataset(split = "train")
    test_ds = ClimateDataset(split = "test")

    print(f"训练集样本数: {len(train_ds)}")
    print(f"测试集样本数: {len(test_ds)}")

    train_start = train_ds.train_indices[0]
    train_end = train_ds.train_indices[-1] + 72
    test_start = test_ds.test_indices[0]

    print(f"测试集开始时间: {test_start}")
    print(f"时间分离: {train_end <= test_start}")

    sample = train_ds[0]
    print(f"Input shape: {sample['input'].shape}")
    print(f"Output shape: {sample['output'].shape}")
    print(f"Terrain shape: {sample['terrain'].shape}")