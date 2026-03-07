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
        
        # Phase 1: Balanced for training from scratch (no MSE pre-training)
        # Charbonnier(sum) ~50k × 2e-6 = 0.1 (21%), LPIPS ~0.3 (64%), Style 0.0001×100=0.01 (2%), BPP 0.2×0.3=0.06 (13%)
        # This balance prevents BPP explosion while maintaining perceptual quality
        "lambda_char": 2e-6,
        "lambda_lpips": 1.0,
        "lambda_style": 100,
        "lambda_bpp_rate": 0.6,
        
        # Phase 2 OURS: MSE + LPIPS + (1-VSI) + Style + GAN + BPP
        # Our modification: MSE instead of Charbonnier, add VSI
        "lambda_rd": 0.01,   # MSE weight (rd_loss already scaled by 255^2)
        "lambda_vsi": 5.0,   # VSI: saliency-aware quality
        "lambda_face": 0,
        "lambda_gan": 1,  # GAN adversarial loss weight for Phase 2
        
        # Phase 2 BASELINE: Original HFLIC - Charbonnier + LPIPS + Style + GAN + BPP
        # HFLIC hardcoded weights: 3e-4 × Char + 2.0 × LPIPS + 1.0 × Style + 1.0 × GAN
        # Only lambda_bpp_rate needs to be changed for different operating points
        # Change lambda_bpp_rate to: 0.3 (high BPP), 1.0 (medium BPP), 2.0 (low BPP)

    })

    return config
