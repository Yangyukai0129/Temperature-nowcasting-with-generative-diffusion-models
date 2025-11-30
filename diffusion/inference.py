import torch
from tqdm import tqdm

@torch.no_grad()
def ddim_inference(model, cond, beta, device, eta=0.0, num_steps=15):
    # 準備 Alpha / Beta
    alpha = 1.0 - beta
    alpha_cumprod = torch.cumprod(alpha, dim=0).to(device)

    # 處理輸入形狀
    # 如果已經是 4D [B, C, H, W]，就不用 view
    if cond.ndim == 5:
        B, T_c, V_c, H, W = cond.shape
        cond = cond.view(B, T_c * V_c, H, W)
    elif cond.ndim == 4:
        pass # 已經是正確形狀
    else:
        raise ValueError(f"Unexpected cond shape: {cond.shape}")

    # 初始噪聲
    shape = (cond.shape[0], model.out_channels, cond.shape[2], cond.shape[3])
    x_t = torch.randn(shape, device=device)

    # 設定時間步
    step_size = 1000 // num_steps
    timesteps = list(range(0, 1000, step_size))[::-1]

    for i, t in enumerate(timesteps):
        t_tensor = torch.full((cond.shape[0],), t, device=device, dtype=torch.long)
        
        # 預測噪聲
        # [修正] 移除傳入 beta
        pred_noise = model(x_t, cond, t_tensor) 

        # DDIM 採樣公式
        alpha_t = alpha_cumprod[t]
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)

        x0_pred = (x_t - sqrt_one_minus_alpha_t * pred_noise) / sqrt_alpha_t
        
        if i < len(timesteps) - 1:
            t_next = timesteps[i + 1]
            alpha_next = alpha_cumprod[t_next]
            sigma_t = eta * torch.sqrt((1 - alpha_next) / (1 - alpha_t) * (1 - alpha_t / alpha_next))
            noise = sigma_t * torch.randn_like(x_t)
            x_t = torch.sqrt(alpha_next) * x0_pred + torch.sqrt(1 - alpha_next - sigma_t**2) * pred_noise + noise
        else:
            x_t = x0_pred

    return x_t