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
        
        # Phase 1: Original HFLIC loss weights
        "lambda_char": 2e-6,  # Charbonnier loss weight (Phase 1)
        "lambda_lpips": 1,  # LPIPS perceptual loss (Phase 1 and Phase 2)
        "lambda_style": 1e2,
        "lambda_bpp_rate": 0.30,  # bpp rate loss weight
        
        # Phase 2: RD + LPIPS + (1-CKDN) + Style + GAN + bpp (no GMSD)
        "lambda_rd": 0.01,   # MSE (Rate-Distortion) loss weight for Phase 2
        "lambda_ckdn": 0.5,  # CKDN weight: 0.5*(1-0.3)≈0.35, complements LPIPS
        "lambda_face": 0,
        "lambda_gan": 1,  # GAN adversarial loss weight for Phase 2

    })

    return config
