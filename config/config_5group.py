from utils.utils import Config

def model_config():
    config = Config({
        # ELIC
        "N": 192,
        "M": 320,
        "slice_num": 5,
        "context_window": 5,
        "slice_ch": [16, 16, 32, 64, 192],
        "quant": "ste",
        
        # Phase 1: Match HFLIC paper (Eq. 7: Lperc + λ·R)
        # HFLIC code uses all implicit weight 1.0 for Charbonnier(sum), LPIPS, Style
        # BPP included because we have no MSE pre-training (HFLIC omits BPP after 500-epoch MSE pre-train)
        "lambda_char": 1.0,
        "lambda_lpips": 1.0,
        "lambda_style": 1.0,
        "lambda_bpp_rate": 1.0,
        
        # Phase 2: MSE + LPIPS + (1-VSI) + Style + GAN + BPP
        # Matches HFLIC hardcoded Phase 2: 3e-4*Char + 2*LPIPS + Style + GAN + BPP
        "lambda_rd": 0.01,   # MSE weight (rd_loss already scaled by 255^2)
        "lambda_vsi": 5.0,   # VSI: increased from 0.5 so saliency-aware quality has real gradient influence
        "lambda_face": 0,
        "lambda_gan": 1,  # GAN adversarial loss weight for Phase 2

    })

    return config
