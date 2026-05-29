from src.config import DEVICE, PT_DATA_DIR, PT_DATA_FILE, TERRAIN_DATA_FILE
from src.dataset import ClimateDataset
from src.models import ClimateUNet
from tqdm import tqdm
import argparse
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ClimateUNet model")
    parser.add_argument(
        "--ckpt", 
        type = str, 
        default = "checkpoints/ckpt_final.pt",
        help = "Path to checkpoint (default: checkpoints/ckpt_final.pt)"
    )
    args = parser.parse_args()
    data = torch.load(PT_DATA_DIR + "/" + PT_DATA_FILE)[:, :, 1:, :-1]
    std = data.std(dim = [0, 2, 3])
    mean = data.mean(dim = [0, 2, 3])

    device = DEVICE

    dataset = ClimateDataset(
        data_path=PT_DATA_DIR + "/" + PT_DATA_FILE,
        terrain_path=PT_DATA_DIR + "/" + TERRAIN_DATA_FILE,
        split = "test"
    )

    model = ClimateUNet(
        terrain_data = dataset.terrain.repeat(2, 1, 1), 
        static_terrain_data = dataset.terrain
    )
    checkpoint = torch.load(args.ckpt, map_location = device, weights_only = False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from {args.ckpt}")
    model.eval()
    model.to(device)

    all_max_errs = []
    all_mean_errs = []
    all_true_t2m_errs = []

    with torch.no_grad():
        pbar = tqdm(dataset, desc = "Testing")
        for idx, batch in enumerate(pbar):
            input_data = batch["input"].unsqueeze(0).to(device)
            output_data = batch["output"].unsqueeze(0).to(device)

            month_data = batch["month"].unsqueeze(0).to(device)
            day_data = batch["day"].unsqueeze(0).to(device)
            hour_data = batch["hour"].unsqueeze(0).to(device)

            time_dict = {
                "month": month_data,
                "day": day_data,
                "hour": hour_data
            }

            output = model(input_data, time_dict)

            t2m_pred = output[:, 0, :, :, :]
            t2m_true = output_data[:, 0, :, :, :]

            max_err = (t2m_pred - t2m_true).abs().max().item()
            mean_err = (t2m_pred - t2m_true).abs().mean().item()
            true_t2m_err = mean_err * std[0].item()

            all_max_errs.append(max_err)
            all_mean_errs.append(mean_err)
            all_true_t2m_errs.append(true_t2m_err)

            #print(f"Sample {idx}: Max Error: {max_err:.4f}, Mean Error: {mean_err:.4f}, True T2M Error: {true_t2m_err:.4f}")

    avg_max_err = sum(all_max_errs) / len(all_max_errs)
    avg_mean_err = sum(all_mean_errs) / len(all_mean_errs)
    avg_true_t2m_err = sum(all_true_t2m_errs) / len(all_true_t2m_errs)

    print(f"Average Max Error: {avg_max_err:.4f}")
    print(f"Average Mean Error: {avg_mean_err:.4f}")
    print(f"Average True T2M Error: {avg_true_t2m_err:.4f}")
    print(f"Total samples: {len(all_max_errs)}")

    print("\nTest completed!")