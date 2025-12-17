import torch
import numpy as np
import PIL.Image as Image
from typing import Dict, List, Optional, Tuple, Union
from pytorch_msssim import ms_ssim


def compute_metrics(
    a: Union[np.array, Image.Image],
    b: Union[np.array, Image.Image],
    max_val: float = 255.0,
) -> Tuple[float, float]:
    """Returns PSNR (dB) and MS-SSIM (dB) between images `a` and `b`.
    
    MS-SSIM is converted to dB scale to match HFLIC paper reporting:
    MS-SSIM_dB = -10 * log10(1 - MS-SSIM_linear)
    """
    if isinstance(a, Image.Image):
        a = np.asarray(a)
    if isinstance(b, Image.Image):
        b = np.asarray(b)

    a = torch.from_numpy(a.copy()).float().unsqueeze(0)
    if a.size(3) == 3:
        a = a.permute(0, 3, 1, 2)
    b = torch.from_numpy(b.copy()).float().unsqueeze(0)
    if b.size(3) == 3:
        b = b.permute(0, 3, 1, 2)

    mse = torch.mean((a - b) ** 2).item()
    p = 20 * np.log10(max_val) - 10 * np.log10(mse)
    
    # Compute MS-SSIM in linear scale first
    m_linear = ms_ssim(a, b, data_range=max_val).item()
    
    # Convert to dB scale to match HFLIC paper reporting
    # MS-SSIM_dB = -10 * log10(1 - MS-SSIM_linear)
    # Handle edge case where MS-SSIM is exactly 1.0 (perfect match)
    if m_linear >= 1.0:
        m_dB = float('inf')
    elif m_linear <= 0.0:
        m_dB = 0.0
    else:
        m_dB = -10.0 * np.log10(1.0 - m_linear)
    
    return p, m_dB
