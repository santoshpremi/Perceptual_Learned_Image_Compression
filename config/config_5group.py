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
        
        # GAN Training (with stability improvements)
        # Using lambda_gan=1.0 (same as DISTS experiment) - stability handled by Multi-Scale D + TTUR + R1
        "lambda_gan": 1.0,     # Adversarial loss weight (matching successful DISTS config)
        "use_multi_scale_disc": True,  # Enable Multi-Scale Discriminator
        "use_r1_penalty": True,        # Enable R1 gradient penalty
        "r1_weight": 10.0,             # R1 penalty weight
        "ttur_ratio": 4.0,             # TTUR: D learning rate = G learning rate * ttur_ratio
        
        # TOPIQ-FR: State-of-the-art FR metric from CVPR 2023
        # Using pyiqa library for IQA metrics
        # Weight set to 0.5 to match successful DISTS experiment (lambda_dists=0.5)
        "lambda_topiq": 0.5,   # TOPIQ-FR weight (matching DISTS weight from successful experiment)
    })

    return config
