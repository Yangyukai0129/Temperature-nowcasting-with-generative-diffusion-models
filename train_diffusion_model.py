import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np

# === 引用 ===
from dataset import WeatherDataset, prepare_file_list
from diffusion.models import UNet
from diffusion.scheduler import get_cosine_schedule

# === 訓練邏輯 (保持不變) ===
# === 訓練邏輯 ===
def train_one_epoch(model, loader, optimizer, criterion, device, alpha_cumprod):
    model.train()
    running_loss = 0.0
    pbar = tqdm(loader, desc="Training")
    
    for cond, target in pbar:
        cond = cond.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        B = cond.shape[0]
        t = torch.randint(0, 1000, (B,), device=device).long()

        noise = torch.randn_like(target)
        sqrt_alpha_bar = torch.sqrt(alpha_cumprod[t])[:, None, None, None]
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_cumprod[t])[:, None, None, None]
        
        x_t = sqrt_alpha_bar * target + sqrt_one_minus_alpha_bar * noise

        optimizer.zero_grad()
        pred_noise = model(x_t, cond, t)
        
        loss = criterion(pred_noise, noise)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})

    return running_loss / len(loader)

def main():
    BATCH_SIZE = 32
    NUM_WORKERS = 8
    LR_STAGE1 = 1e-4
    LR_STAGE2 = 1e-5
    EPOCHS_STAGE1 = 40
    EPOCHS_STAGE2 = 10
    
    # ⚠️ 這裡決定了你要用多少資料訓練
    # "20200101" = 用 2020 以前的訓練，保留 2020 以後的不看 (可以用於最後的 inference 測試)
    # "20991231" = 用所有資料訓練 (模型看過所有答案，適合用來做最終產品)
    SPLIT_DATE = "20200101" 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置：{device}")

    # === 統計數據 ===
    STATS_PATH = "data/data_stats.pt"
    cond_stats, target_stats = None, None
    if os.path.exists(STATS_PATH):
        stats = torch.load(STATS_PATH)
        cond_stats = {'mean': stats['cond_mean'], 'std': stats['cond_std']}
        target_stats = {'mean': stats['target_mean'], 'std': stats['target_std']}
    else:
        print("⚠️ Warning: Stats not found!")

    # === 載入資料 ===
    DATA_DIR = "data/npy_files"
    
    if os.path.exists(DATA_DIR):
        # 呼叫 dataset.py 裡的函式
        # train_list: 2020以前
        # test_list: 2020以後 (這裡不使用，留給之後的 GED.py 用)
        train_list, test_list = prepare_file_list(DATA_DIR, split_date=SPLIT_DATE)
        
        print(f"Training Samples: {len(train_list)}")
        print(f"Test Samples (Reserved): {len(test_list)}") # 顯示一下保留了多少資料
        
        if len(train_list) == 0:
            print("❌ Error: No training data found.")
            return

        # 只建立 Training Dataset
        train_dataset = WeatherDataset(train_list, cond_stats, target_stats)
    else:
        print(f"❌ Data directory {DATA_DIR} not found.")
        return

    # === DataLoader ===
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

    # === 模型建立 ===
    sample_cond, sample_target = next(iter(train_loader))
    cond_channels = sample_cond.shape[1]
    out_channels = sample_target.shape[1]
    print(f"Detected channels -> Cond: {cond_channels}, Out: {out_channels}")

    model = UNet(in_channels=out_channels, out_channels=out_channels, 
                 cond_channels=cond_channels, time_dim=32).to(device) 
    
    beta = get_cosine_schedule(1000).to(device)
    alpha = 1.0 - beta
    alpha_cumprod = torch.cumprod(alpha, dim=0).to(device)

    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=LR_STAGE1, weight_decay=1e-5)
    
    save_dir = "./checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    # === Training Loop ===
    print("=== Start Stage 1 Training ===")
    
    for epoch in range(EPOCHS_STAGE1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, alpha_cumprod)
        print(f"Epoch {epoch+1}/{EPOCHS_STAGE1} - Train Loss: {train_loss:.6f}")
        
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"{save_dir}/unet_stage1_ep{epoch+1}.pth")

    print("=== Start Stage 2 Fine-tuning ===")
    for param_group in optimizer.param_groups:
        param_group['lr'] = LR_STAGE2
    
    for epoch in range(EPOCHS_STAGE2):
        current_epoch = EPOCHS_STAGE1 + epoch + 1
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, alpha_cumprod)
        print(f"Epoch {current_epoch} - Train Loss: {train_loss:.6f}")

        if (epoch + 1) % 5 == 0:
             torch.save(model.state_dict(), f"{save_dir}/unet_stage2_ep{epoch+1}.pth")

    # 最終存檔
    torch.save(model.state_dict(), f"{save_dir}/unet_final.pth")
    print("Done. Model saved to checkpoints/unet_final.pth")

if __name__ == "__main__":
    main()