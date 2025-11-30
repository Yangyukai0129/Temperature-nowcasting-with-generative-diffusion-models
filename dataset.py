import torch
import numpy as np
import os
from torch.utils.data import Dataset
from diffusion.utils import Normalizer
from tqdm import tqdm
import datetime

# === 新增：通用的檔案列表切分函式 ===
def prepare_file_list(data_dir, split_date="20200101", date_format="%Y%m%d"):
    """
    掃描 data_dir 下的 .npy 檔案，並根據 split_date 切分為 Train/Test。
    回傳格式: List of tuples [(cond_path, target_path, time_path), ...]
    """
    print(f"Scanning files in {data_dir}...")
    
    cond_dir = os.path.join(data_dir, "cond")
    target_dir = os.path.join(data_dir, "target")
    time_dir = os.path.join(data_dir, "time")
    
    # 只需要讀取 time 資料夾的列表，其他路徑用推算的 (節省 I/O)
    if not os.path.exists(time_dir):
        raise FileNotFoundError(f"Directory not found: {time_dir}")
        
    filenames = sorted([f for f in os.listdir(time_dir) if f.endswith('.npy')])
    
    # 設定切分時間點
    split_dt = datetime.datetime.strptime(split_date, date_format)
    
    train_files = []
    test_files = []
    
    print(f"Filtering files by date (Split: {split_dt})...")
    
    for fname in tqdm(filenames, desc="Reading time files"):
        # 組合路徑
        c_path = os.path.join(cond_dir, fname)
        t_path = os.path.join(target_dir, fname)
        tm_path = os.path.join(time_dir, fname)
        
        # 讀取時間
        try:
            ts = np.load(tm_path)
            # 處理 numpy 0-d array
            if isinstance(ts, np.ndarray) and ts.ndim == 0:
                ts = ts.item()
            elif isinstance(ts, np.ndarray) and ts.size == 1:
                ts = ts.item()
                
            # 轉換為 datetime 物件 (處理 numpy datetime64)
            # 轉為 ms 精度再轉 datetime 比較安全
            dt = ts.astype('datetime64[ms]').astype(datetime.datetime)
            
            # 切分
            record = (c_path, t_path, tm_path)
            if dt < split_dt:
                train_files.append(record)
            else:
                test_files.append(record)
                
        except Exception as e:
            print(f"Warning: Failed to process {fname}: {e}")
            continue

    print(f"[Result] Train: {len(train_files)} | Test: {len(test_files)}")
    return train_files, test_files

class WeatherDataset(Dataset):
    def __init__(self, file_list, cond_stats=None, target_stats=None):
        """
        file_list: List of tuples [(cond_path, target_path, time_val), ...]
                   或者是只包含路徑的 tuple，視 preprocess_stats 回傳而定
        """
        self.file_list = file_list
        
        # 初始化 Normalizer
        c_mean = cond_stats['mean'] if cond_stats else 0
        c_std = cond_stats['std'] if cond_stats else 1
        t_mean = target_stats['mean'] if target_stats else 0
        t_std = target_stats['std'] if target_stats else 1
        
        self.cond_normalizer = Normalizer(c_mean, c_std)
        self.target_normalizer = Normalizer(t_mean, t_std)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # 1. 讀取路徑 (這裡假設 file_list 裡存的是完整路徑)
        item = self.file_list[idx]
        cond_path = item[0]
        target_path = item[1]
        
        # 2. 讀取 .npy
        # Shape: [Time, Var, H, W]
        cond_np = np.load(cond_path)
        target_np = np.load(target_path)

        cond = torch.from_numpy(cond_np).float()
        target = torch.from_numpy(target_np).float()

        # 3. [關鍵修改] 展平維度
        # 將 [Time, Var, H, W] 變成 [Time * Var, H, W]
        # 這樣才能丟進 UNet 的 Conv2d
        T_c, V_c, H, W = cond.shape
        cond = cond.view(T_c * V_c, H, W)

        T_t, V_t, _, _ = target.shape
        target = target.view(T_t * V_t, H, W)

        # 4. 標準化
        # Normalizer 會自動處理廣播，不用擔心維度
        cond = self.cond_normalizer.normalize(cond)
        target = self.target_normalizer.normalize(target)

        return cond, target