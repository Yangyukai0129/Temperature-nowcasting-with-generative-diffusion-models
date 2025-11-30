import torch

class Normalizer:
    def __init__(self, mean, std):
        """
        mean, std: 可以是 float, list, numpy, 或 tensor
        """
        self.mean = self._to_tensor(mean)
        self.std = self._to_tensor(std)
        self.device = None

    def _to_tensor(self, val):
        if val is None:
            return torch.tensor(0.0)
        if not torch.is_tensor(val):
            val = torch.tensor(val)
        # 確保轉為 Float 並且形狀為 [1, C, 1, 1] 以支援廣播
        if val.ndim == 1:
            val = val.view(1, -1, 1, 1)
        elif val.ndim == 0:
            val = val.view(1, 1, 1, 1)
        return val.float()

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        self.device = device
        return self

    def normalize(self, x):
        # 自動檢查裝置，若不同則移動 mean/std
        if x.device != self.mean.device:
            self.to(x.device)
        return (x - self.mean) / (self.std + 1e-6)

    def denormalize(self, x):
        if x.device != self.mean.device:
            self.to(x.device)
        return x * (self.std + 1e-6) + self.mean