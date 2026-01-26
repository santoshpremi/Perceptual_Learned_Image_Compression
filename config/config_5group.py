from utils.utils import Config

def model_config():
    config = Config({
        # ELIC Network Architecture
        "N": 192,
        "M": 320,
        "slice_num": 5,
        "context_window": 5,
        "slice_ch": [16, 16, 32, 64, 192],
        "quant": "ste",
        
        # ============================================================
        # Phase 1: Original HFLIC loss weights (pre-training without GAN)
        # ============================================================
        "lambda_char": 2e-6,  # Charbonnier loss weight (original HFLIC)
        "lambda_lpips": 1,
        "lambda_style": 1e2,
        "lambda_rate": 0.3,
        
        # ============================================================
        # Phase 2: Enhanced perceptual loss weights (with GAN)
        # Matched to successful DISTS-only experiment weights
        # ============================================================
        "lambda_rd": 0.01,     # MSE for pixel-level fidelity
        "lambda_face": 0,      # Face-specific loss (disabled)
        
        # GAN Training (Conservative approach for stability)
        # lambda_gan reduced to 0.1 to prevent PSNR collapse while maintaining perceptual benefits
        "lambda_gan": 0.1,     # Reduced from 1.0 to prevent PSNR degradation
        "use_multi_scale_disc": False,  # Single-scale discriminator (faster, matches original HFLIC)
        "use_r1_penalty": True,        # Keep R1 gradient penalty for stability
        "r1_weight": 10.0,             # R1 penalty weight
        "ttur_ratio": 1.0,             # Standard training (reduced from 4.0)
        
        # TOPIQ-FR: State-of-the-art FR metric from CVPR 2023
        # Using pyiqa library for IQA metrics
        # Weight set to 0.5 to match successful DISTS experiment (lambda_dists=0.5)
        "lambda_topiq": 0.5,   # TOPIQ-FR weight (matching DISTS weight from successful experiment)
    })

    return config
