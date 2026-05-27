import argparse
import os

from torch.utils.data import DataLoader
from src.dataset import ClimateDataset
from src.model import ClimateUNet
from src.config import DEVICE, PT_DATA_DIR, PT_DATA_FILE, TERRAIN_DATA_FILE, BATCH_SIZE, ACCUMULATE_STEPS, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from src import Show
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ClimateUNet model")
    parser.add_argument("--resume", "--ckpt", type=str, default=None, dest="resume",
                        help="Path to checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS,
                        help=f"Number of training epochs (default: {NUM_EPOCHS})")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Batch size (default: {BATCH_SIZE})")
    parser.add_argument("--accumulate-steps", type=int, default=ACCUMULATE_STEPS,
                        help=f"Gradient accumulation steps (default: {ACCUMULATE_STEPS})")
    args = parser.parse_args()

    train_dataset = ClimateDataset(
        data_path=PT_DATA_DIR + "/" + PT_DATA_FILE,
        terrain_path=PT_DATA_DIR + "/" + TERRAIN_DATA_FILE,
        split = "train"
    )
    eval_dataset = ClimateDataset(
        data_path=PT_DATA_DIR + "/" + PT_DATA_FILE,
        terrain_path=PT_DATA_DIR + "/" + TERRAIN_DATA_FILE,
        split = "test"
    )
    model = ClimateUNet(
        terrain_data = train_dataset.terrain.repeat(2, 1, 1), 
        static_terrain_data = train_dataset.terrain
    )
    optimizer = optim.AdamW(
        model.parameters(),
        lr = LEARNING_RATE,
        weight_decay = WEIGHT_DECAY
    )
    criterion = nn.MSELoss()
    device = DEVICE
    model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)
    accumulate_steps = args.accumulate_steps

    start_epoch = 0
    if args.resume:
        print(f"Loading checkpoint from {args.resume} ...")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed from epoch {checkpoint['epoch']}. Continuing from epoch {start_epoch + 1}/{args.epochs}")

    animator = Show.Animator(xlabel = "Epoch", ylabel = "Loss", legend = ["Train", "Eval"])

    cnt = 0
    for epoch in range(start_epoch, args.epochs):
        eval_loss = 0
        train_loss = 0.0
        total_train_loss = 0.0
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for idx, data in enumerate(pbar):
            input_data = data["input"].to(device)
            output_data = data["output"].to(device)
            month_data = data["month"].to(device)
            day_data = data["day"].to(device)
            hour_data = data["hour"].to(device)
            time_dict = {
                "month" : month_data,
                "day" : day_data,
                "hour" : hour_data
            }
            output = model(input_data, time_dict)
            loss = criterion(output, output_data) / accumulate_steps
            loss.backward()
            train_loss += loss.item()
            cnt += 1

            if cnt % accumulate_steps == 0:
                total_train_loss += train_loss
                optimizer.step()
                optimizer.zero_grad()
                animator.add(epoch + idx / len(train_loader), (train_loss, None))
                train_loss = 0
                cnt = 0

        num_train_steps = len(train_loader) // accumulate_steps
        if num_train_steps > 0:
            train_loss = total_train_loss / num_train_steps

        os.makedirs("checkpoints", exist_ok=True)
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "config": {
                "batch_size": args.batch_size,
                "accumulate_steps": args.accumulate_steps,
                "num_epochs": args.epochs,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
            },
        }
        torch.save(checkpoint, f"checkpoints/ckpt_epoch_{epoch + 1}.pt")

        model.eval()
        with torch.no_grad():
            pbar = tqdm(eval_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Eval]")
            for idx, data in enumerate(pbar):
                input_data = data["input"].to(device)
                output_data = data["output"].to(device)
                month_data = data["month"].to(device)
                day_data = data["day"].to(device)
                hour_data = data["hour"].to(device)
                time_dict = {
                    "month" : month_data,
                    "day" : day_data,
                    "hour" : hour_data
                }

                output = model(input_data, time_dict)
                loss = criterion(output, output_data)
                eval_loss += loss.item()
        
        eval_loss /= len(eval_loader)
        animator.add(epoch + 1, (train_loss, eval_loss))
        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}, Eval Loss: {eval_loss:.4f}")


    torch.save(checkpoint, "checkpoints/ckpt_final.pt")
    print(f"Training complete. Final checkpoint saved to checkpoints/ckpt_final.pt")
