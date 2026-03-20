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
        
        # ═══════════════════════════════════════════════════════════════════
        # SHARED HYPERPARAMETERS (used by Phase 1, Phase 2 Baseline, Phase 2 Ours)
        # ═══════════════════════════════════════════════════════════════════
        
        # Loss weights - common across phases
        "lambda_lpips": 1.0,          # LPIPS perceptual loss weight
        "lambda_style": 100.0,        # Style loss weight (HFLIC uses 1e2=100)
        "lambda_bpp_rate": 1.0,       # Control run for ours; sweep 0.35, 0.5, 0.9 for R-D points
        "lambda_gan": 1.0,            # GAN adversarial loss weight (Phase 2 only)
        
        # Phase 1 specific (Charbonnier-based, no GAN)
        "lambda_char": 2e-6,          # Charbonnier loss weight for Phase 1
        
        # Phase 2 OURS specific (MSE + DISTS/TOPIQ)
        "lambda_rd": 0.01,            # Raw MSE weight (matches old successful DISTS+LPIPS regime)
        "lambda_dists": 0.5,          # DISTS image-level structure/texture weight
        "lambda_vsi": 0,              # VSI disabled (set >0 to enable)
        "lambda_topiq": 0,            # TOPIQ disabled (set >0 to enable)
        "lambda_face": 0,             # Face loss weight (for face-specific training)
        
        # ═══════════════════════════════════════════════════════════════════
        # NOTES:
        # - Phase 1: Uses lambda_char, lambda_lpips, lambda_style, lambda_bpp_rate
        # - Phase 2 Baseline: Uses lambda_char, lambda_lpips, lambda_style, lambda_gan, lambda_bpp_rate
        # - Phase 2 Ours: Uses lambda_rd, lambda_lpips, lambda_dists, lambda_style, lambda_gan, lambda_bpp_rate
        # - All phases now read lambda_style from config (100.0) for consistency
        # ═══════════════════════════════════════════════════════════════════
    })

    return config
