# import torch
# import numpy as np
# import os
# from torch.utils.data import Dataset
# from diffusion.utils import Normalizer
# from tqdm import tqdm
# import datetime

# # === 新增：通用的檔案列表切分函式 ===
# def prepare_file_list(data_dir, split_date="20200101", date_format="%Y%m%d"):
#     print(f"Scanning files in {data_dir}...")
    
#     cond_dir = os.path.join(data_dir, "cond")
#     target_dir = os.path.join(data_dir, "target")
#     time_dir = os.path.join(data_dir, "time")
    
#     # 確保路徑存在
#     if not os.path.exists(time_dir):
#         raise FileNotFoundError(f"Directory not found: {time_dir}")
        
#     filenames = sorted([f for f in os.listdir(time_dir) if f.endswith('.npy')])
    
#     # 設定切分時間點
#     split_dt = datetime.datetime.strptime(split_date, date_format)
    
#     train_files = []
#     test_files = []
    
#     print(f"Filtering files by date (Split: {split_dt})...")
    
#     for fname in tqdm(filenames, desc="Reading time files"):
#         # 組合路徑
#         c_path = os.path.join(cond_dir, fname)
#         t_path = os.path.join(target_dir, fname)
#         tm_path = os.path.join(time_dir, fname)
        
#         try:
#             # 讀取時間
#             ts = np.load(tm_path)
            
#             # 處理 0-d array (也就是只有一個值的 array)
#             # 修正重點：不要過早使用 .item()，保持它為 numpy 物件以便轉換
#             if isinstance(ts, np.ndarray) and ts.size == 1:
#                 # 強制視為 datetime64[ns]，相容性最高
#                 ts = ts.reshape(-1)[0]
            
#             # 統一轉換邏輯：先轉 datetime64[ms] 再轉 python datetime
#             # 這樣無論原始存的是 ns, us, s 單位，或是 datetime64 物件，都能正常運作
#             dt = ts.astype('datetime64[ms]').astype(datetime.datetime)
            
#             # 切分 (這裡使用 <，代表 split_date 當天屬於測試集)
#             record = (c_path, t_path, tm_path)
#             if dt < split_dt:
#                 train_files.append(record)
#             else:
#                 test_files.append(record)
                
#         except Exception as e:
#             # 如果還是讀不到，可能檔案壞了或格式不對，印出警告但不中斷
#             print(f"Warning: Failed to process {fname}: {e}")
#             continue

#     print(f"[Result] Train: {len(train_files)} | Test: {len(test_files)}")
#     return train_files, test_files

# class WeatherDataset(Dataset):
#     def __init__(self, file_list, cond_stats=None, target_stats=None):
#         self.file_list = file_list
#         self.cond_stats = cond_stats
#         self.target_stats = target_stats
        
#         # 我們需要確保長寬是 32 的倍數 (2^5)，適應 UNet 4~5 層下採樣
#         self.size_divisor = 32 

#     def __len__(self):
#         return len(self.file_list)
    
#     def pad_image(self, img):
#         """
#         將影像 (C, H, W) 補零至 32 的倍數
#         """
#         _, h, w = img.shape
#         # 計算需要補多少
#         pad_h = (self.size_divisor - h % self.size_divisor) % self.size_divisor
#         pad_w = (self.size_divisor - w % self.size_divisor) % self.size_divisor
        
#         if pad_h == 0 and pad_w == 0:
#             return img
        
#         # np.pad 格式: ((channel_front, channel_back), (top, bottom), (left, right))
#         # 使用 'edge' 模式比較適合氣象資料，避免補 0 造成邊界梯度異常
#         img_padded = np.pad(img, ((0, 0), (0, pad_h), (0, pad_w)), mode='edge')
#         return img_padded

#     def __getitem__(self, idx):
#         # 1. 取得路徑
#         cond_path, target_path, time_path = self.file_list[idx]
        
#         # 2. 讀取檔案
#         cond = np.load(cond_path)     # (T, C, H, W)
#         target = np.load(target_path) # (T, C, H, W)
        
#         # 3. 展平維度 (Time -> Channels)
#         # cond: (64, 1, H, W) -> (64, H, W)
#         T_c, C_c, H, W = cond.shape
#         cond = cond.reshape(T_c * C_c, H, W)
        
#         T_t, C_t, _, _ = target.shape
#         target = target.reshape(T_t * C_t, H, W)

#         # 4. 【新增】執行 Padding
#         # 確保 H, W 都能被 32 整除
#         cond = self.pad_image(cond)
#         target = self.pad_image(target)

#         # 5. 轉 Tensor
#         cond = torch.from_numpy(cond).float()
#         target = torch.from_numpy(target).float()
        
#         # 6. 正規化 (預留位置)
#         if self.cond_stats is not None:
#             pass 
#         if self.target_stats is not None:
#             pass

#         return cond, target
    
import torch
import numpy as np
import os
import datetime
from torch.utils.data import Dataset
from tqdm import tqdm

# === 1. 引用你的 utils.py ===
from diffusion.utils import Normalizer


# === 2. 檔案列表準備函式 (維持不變) ===
def prepare_file_list(data_dir, split_date="20200101", date_format="%Y%m%d"):
    print(f"Scanning files in {data_dir}...")
    cond_dir = os.path.join(data_dir, "cond")
    target_dir = os.path.join(data_dir, "target")
    time_dir = os.path.join(data_dir, "time")
    
    if not os.path.exists(time_dir):
        raise FileNotFoundError(f"Directory not found: {time_dir}")
        
    filenames = sorted([f for f in os.listdir(time_dir) if f.endswith('.npy')])
    split_dt = datetime.datetime.strptime(split_date, date_format)
    train_files, test_files = [], []
    
    print(f"Filtering files by date (Split: {split_dt})...")
    
    for fname in tqdm(filenames, desc="Reading time files"):
        c_path = os.path.join(cond_dir, fname)
        t_path = os.path.join(target_dir, fname)
        tm_path = os.path.join(time_dir, fname)
        
        try:
            ts = np.load(tm_path)
            if isinstance(ts, np.ndarray) and ts.size == 1:
                ts = ts.reshape(-1)[0]
            
            dt = ts.astype('datetime64[ms]').astype(datetime.datetime)
            record = (c_path, t_path, tm_path)
            
            if dt < split_dt:
                train_files.append(record)
            else:
                test_files.append(record)     
        except Exception as e:
            continue

    print(f"[Result] Train: {len(train_files)} | Test: {len(test_files)}")
    return train_files, test_files

# === 3. 整合後的 WeatherDataset ===
class WeatherDataset(Dataset):
    def __init__(self, file_list, cond_stats=None, target_stats=None):
        self.file_list = file_list
        self.size_divisor = 32 
        
        # === 這裡使用你的 Normalizer ===
        # 注意：train_diffusion_model.py 傳進來的是字典 {'mean': ..., 'std': ...}
        # 你的 Normalizer __init__ 需要分開傳 mean, std
        
        if cond_stats is not None:
            self.cond_normalizer = Normalizer(cond_stats['mean'], cond_stats['std'])
        else:
            self.cond_normalizer = None

        if target_stats is not None:
            self.target_normalizer = Normalizer(target_stats['mean'], target_stats['std'])
        else:
            self.target_normalizer = None

    def __len__(self):
        return len(self.file_list)
    
    def pad_image(self, img):
        _, h, w = img.shape
        pad_h = (self.size_divisor - h % self.size_divisor) % self.size_divisor
        pad_w = (self.size_divisor - w % self.size_divisor) % self.size_divisor
        if pad_h == 0 and pad_w == 0: return img
        img_padded = np.pad(img, ((0, 0), (0, pad_h), (0, pad_w)), mode='edge')
        return img_padded

    def __getitem__(self, idx):
        cond_path, target_path, _ = self.file_list[idx]
        
        # 讀取
        cond = np.load(cond_path)     # (T, C, H, W)
        target = np.load(target_path) # (T, C, H, W)
        
        # 展平維度 (Time -> Channels)
        T_c, C_c, H, W = cond.shape
        cond = cond.reshape(T_c * C_c, H, W)
        T_t, C_t, _, _ = target.shape
        target = target.reshape(T_t * C_t, H, W)

        # Padding
        cond = self.pad_image(cond)
        target = self.pad_image(target)

        # 轉 Tensor
        cond = torch.from_numpy(cond).float()
        target = torch.from_numpy(target).float()
        
        # === 正規化 ===
        if self.cond_normalizer is not None:
            # 你的 Normalizer 預設處理 4D (B, C, H, W)
            # 這裡 cond 是 3D (C, H, W)
            # 技巧：unsqueeze(0) 變 (1, C, H, W) -> normalize -> squeeze(0) 變回 (C, H, W)
            cond = self.cond_normalizer.normalize(cond.unsqueeze(0)).squeeze(0)
            
        if self.target_normalizer is not None:
            target = self.target_normalizer.normalize(target.unsqueeze(0)).squeeze(0)

        return cond, target