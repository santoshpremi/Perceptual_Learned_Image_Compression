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
        
        # Phase 2: RD + (1-AHIQ) + (1-TOPIQ) + Style + GAN + bpp
        "lambda_rd": 0.01,   # MSE (Rate-Distortion) loss weight for Phase 2
        "lambda_ahiq": 0.5,  # AHIQ weight (FR): Attention-based hybrid (ViT+CNN, bottom-up)
        "lambda_topiq": 0.7,  # TOPIQ weight (FR): Semantic-guided (top-down, ResNet50)
        "lambda_face": 0,
        "lambda_gan": 1,  # GAN adversarial loss weight for Phase 2

    })

    return config
