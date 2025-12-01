import numpy as np
import os
import torch
from tqdm import tqdm
from dataset import prepare_file_list 

def compute_stats_for_folder(file_list, max_samples=3000):
    """
    計算 Mean 和 Std
    """
    print(f"Computing stats from {len(file_list)} samples...")
    
    num_samples = min(max_samples, len(file_list))
    indices = np.random.choice(len(file_list), num_samples, replace=False)
    
    cond_data_list = []
    target_data_list = []
    
    for idx in tqdm(indices, desc="Loading samples for stats"):
        item = file_list[idx]
        cond_path = item[0]
        target_path = item[1]
        
        c = np.load(cond_path) # [T, V, H, W]
        t = np.load(target_path)
        
        # 展平維度，符合 UNet 輸入 [C, H, W]
        T_c, V_c, H, W = c.shape
        c = c.reshape(T_c * V_c, H, W)
        
        T_t, V_t, _, _ = t.shape
        t = t.reshape(T_t * V_t, H, W)

        cond_data_list.append(c)
        target_data_list.append(t)

    cond_stack = np.stack(cond_data_list, axis=0)     # [N, C, H, W]
    target_stack = np.stack(target_data_list, axis=0)
    
    # 計算 Channel-wise Mean/Std
    cond_mean = np.mean(cond_stack, axis=(0, 2, 3))
    cond_std = np.std(cond_stack, axis=(0, 2, 3))
    
    target_mean = np.mean(target_stack, axis=(0, 2, 3))
    target_std = np.std(target_stack, axis=(0, 2, 3))
    
    return cond_mean, cond_std, target_mean, target_std

if __name__ == "__main__":
    # [修正] 路徑要跟 preprocess_data.py 一致
    DATA_DIR = "data/8day_3day" 
    STATS_PATH = "data/data_stats.pt"
    # [修正] 必須跟 Training 設定的切分日期一致 (只用訓練集算統計)
    SPLIT_DATE = "20200101" 
    
    if not os.path.exists(DATA_DIR):
        print(f"❌ {DATA_DIR} not found. Run preprocess_data.py first.")
        exit()

    print("Getting file list...")
    # 只取 train_list (2020以前)
    train_list, _ = prepare_file_list(DATA_DIR, split_date=SPLIT_DATE)
    
    if len(train_list) == 0:
        print("❌ No training data found.")
        exit()

    c_mean, c_std, t_mean, t_std = compute_stats_for_folder(train_list)

    print("\n=== Stats Result ===")
    print(f"Cond Mean: {c_mean}")
    print(f"Target Mean: {t_mean}")

    print(f"Saving to {STATS_PATH}...")
    torch.save({
        'cond_mean': torch.from_numpy(c_mean).float(),
        'cond_std': torch.from_numpy(c_std).float(),
        'target_mean': torch.from_numpy(t_mean).float(),
        'target_std': torch.from_numpy(t_std).float()
    }, STATS_PATH)
    
    print("✅ Done.")