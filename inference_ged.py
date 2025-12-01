import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from torch.utils.data import DataLoader

# === 1. 引用模組 ===
from diffusion.models import UNet
from diffusion.scheduler import get_cosine_schedule
from diffusion.inference import ddim_inference
# 這裡引用 prepare_file_list 來取得測試集
from dataset import WeatherDataset, prepare_file_list

# === 2. Metrics 計算工具 (新增 RMSE) ===
def compute_daywise_metrics(pred, target):
    """
    計算 MSE 與 RMSE
    回傳: (mse_tuple, rmse_tuple)
    """
    mse_fn = nn.MSELoss(reduction='mean')
    
    # --- 計算整體 ---
    mse_all = mse_fn(pred, target).item()
    rmse_all = np.sqrt(mse_all) # RMSE = sqrt(MSE)
    
    # --- 計算分日 (假設 target 是 24 channels: 3天 * 8時段) ---
    if pred.shape[1] == 24:
        # Day 1
        mse_day1 = mse_fn(pred[:, 0:8], target[:, 0:8]).item()
        rmse_day1 = np.sqrt(mse_day1)
        
        # Day 2
        mse_day2 = mse_fn(pred[:, 8:16], target[:, 8:16]).item()
        rmse_day2 = np.sqrt(mse_day2)
        
        # Day 3
        mse_day3 = mse_fn(pred[:, 16:24], target[:, 16:24]).item()
        rmse_day3 = np.sqrt(mse_day3)
    else:
        mse_day1, mse_day2, mse_day3 = -1, -1, -1 
        rmse_day1, rmse_day2, rmse_day3 = -1, -1, -1

    return (mse_day1, mse_day2, mse_day3, mse_all), (rmse_day1, rmse_day2, rmse_day3, rmse_all)

# === 3. 主程式 ===
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 4   # 推論時 Batch size (視顯存大小調整)
    N_SAMPLES = 15   # 每個樣本生成幾次 (Ensemble)
    
    # [關鍵設定] 必須跟當初規劃的切分點一致
    SPLIT_DATE = "20200101" 
    
    print(f"Using device: {device}")
    
    # --- A. 準備統計數據 ---
    STATS_PATH = "data/data_stats.pt"
    cond_stats, target_stats = None, None
    if os.path.exists(STATS_PATH):
        stats = torch.load(STATS_PATH, map_location=device)
        cond_stats = {'mean': stats['cond_mean'], 'std': stats['cond_std']}
        target_stats = {'mean': stats['target_mean'], 'std': stats['target_std']}
    else:
        print("⚠️ Warning: Stats not found. Results will be wrong.")

    # --- B. 準備測試資料 (使用 prepare_file_list) ---
    DATA_DIR = "data/8day_3day"
    
    if os.path.exists(DATA_DIR):
        print(f"Splitting data by date: {SPLIT_DATE}")
        # 我們只關心 test_list
        _, test_list = prepare_file_list(DATA_DIR, split_date=SPLIT_DATE)
        
        print(f"Found {len(test_list)} test samples (Date >= {SPLIT_DATE})")
        
        if len(test_list) == 0:
            print("❌ No test data found! Check SPLIT_DATE or your data.")
            exit()
            
        # 建立 Dataset
        test_dataset = WeatherDataset(test_list, cond_stats, target_stats)
        
        # 使用 DataLoader
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    else:
        print(f"❌ Data directory {DATA_DIR} not found.")
        exit()

    # --- C. 載入模型 ---
    print("Loading model...")
    # 自動偵測通道數
    sample_cond, sample_target = test_dataset[0]
    cond_channels = sample_cond.shape[0]
    out_channels = sample_target.shape[0]
    
    model = UNet(in_channels=out_channels, out_channels=out_channels, cond_channels=cond_channels, time_dim=32).to(device)
    
    checkpoint_path = "./checkpoints/unet_final.pth" 
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded weights from {checkpoint_path}")
    else:
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        exit()
        
    beta = get_cosine_schedule(1000).to(device)
    model.eval()

    # --- D. 開始推論 (Ensemble) ---
    print(f"Start sampling {N_SAMPLES} times for {len(test_dataset)} inputs...")
    
    final_preds = []
    final_targets = []
    
    with torch.no_grad():
        for batch_idx, (cond, target) in enumerate(tqdm(test_loader, desc="Batch Inference")):
            cond = cond.to(device)
            
            # 針對這一個 Batch，進行 N_SAMPLES 次生成
            batch_ensemble = []
            for _ in range(N_SAMPLES):
                # 生成結果 (Normalized)
                gen_norm = ddim_inference(model, cond, beta, device=device)
                
                # 還原數值 (Denormalize)
                gen_raw = test_dataset.target_normalizer.denormalize(gen_norm)
                batch_ensemble.append(gen_raw.cpu())
            
            # 計算 Ensemble Mean
            batch_ensemble = torch.stack(batch_ensemble) # [N, B, C, H, W]
            batch_mean = batch_ensemble.mean(dim=0)      # [B, C, H, W]
            
            # 還原 Ground Truth
            target_raw = test_dataset.target_normalizer.denormalize(target.to(device)).cpu()
            
            final_preds.append(batch_mean)
            final_targets.append(target_raw)

    # --- E. 彙整結果與計算 Metrics ---
    final_preds = torch.cat(final_preds, dim=0)
    final_targets = torch.cat(final_targets, dim=0)
    
    print("\nCalculating metrics...")
    
    # 呼叫新的 metrics 函式
    (mse_d1, mse_d2, mse_d3, final_mse), (rmse_d1, rmse_d2, rmse_d3, final_rmse) = compute_daywise_metrics(final_preds, final_targets)

    print("\n=== Final Results (GDE Mean) ===")
    print(f"Total MSE : {final_mse:.6f}")
    print(f"Total RMSE: {final_rmse:.6f}")
    print("-" * 30)
    
    if mse_d1 != -1:
        print(f"Day 1 - MSE: {mse_d1:.6f}, RMSE: {rmse_d1:.6f}")
        print(f"Day 2 - MSE: {mse_d2:.6f}, RMSE: {rmse_d2:.6f}")
        print(f"Day 3 - MSE: {mse_d3:.6f}, RMSE: {rmse_d3:.6f}")

    # --- F. 存檔 ---
    # os.makedirs("results", exist_ok=True)
    # torch.save(final_preds, "results/gde_mean_prediction.pt")
    # torch.save(final_targets, "results/ground_truth.pt")
    # print("\nResults saved to ./results/")