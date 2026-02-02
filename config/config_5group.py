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
        "lambda_lpips": 1,  # LPIPS perceptual loss weight (Phase 1)
        "lambda_style": 1e2,
        "lambda_rate": 0.3,
        
        # Phase 2: Additional perceptual loss weights
        "lambda_rd": 1e-2,  # MSE (Rate-Distortion) loss weight for Phase 2
        "lambda_dists": 1.0,  # DISTS perceptual loss weight (Phase 2)
        "lambda_face": 0,
        "lambda_gan": 1,  # GAN adversarial loss weight for Phase 2

    })

    return config
