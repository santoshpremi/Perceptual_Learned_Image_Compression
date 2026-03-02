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
        "lambda_lpips": 2.0,  # LPIPS perceptual loss (Phase 1 and Phase 2)
        "lambda_style": 1.0,
        "lambda_bpp_rate": 1.0,   # bpp rate loss weight (HFLIC uses 1.0 in Phase 2)
        
        # Phase 2: RD + LPIPS + (1-VSI) + Style + GAN + bpp
        "lambda_rd": 0.01,   # MSE weight (rd_loss already scaled by 255^2, so conventional small lambda)
        "lambda_vsi": 0.5,   # VSI weight (FR): Visual Saliency-induced Index (higher=better)
        "lambda_face": 0,
        "lambda_gan": 1,  # GAN adversarial loss weight for Phase 2

    })

    return config
