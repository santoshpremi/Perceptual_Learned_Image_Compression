"""
Strategy 3 - Unified Phase 2 Configuration with Automatic Scheduling
=====================================================================
Target: PSNR 32.29 dB, MS-SSIM 15.04 dB, LPIPS 0.0708, BPP 0.17

This configuration automatically switches between two sub-phases:
- Phase 2a (epochs 0-14): Rate Anchoring - Establish BPP at 0.17
- Phase 2b (epochs 15+): Perceptual Refinement - Maximize quality

Single training run - no need to switch configs manually!
"""

from utils.utils import Config

def model_config(epoch=0):
    """
    Returns config with automatic phase switching based on epoch.
    
    Args:
        epoch: Current training epoch (0-indexed from Phase 2 start)
    
    Returns:
        Config object with appropriate weights for current phase
    """
    # Phase 2a: Rate Anchoring (epochs 0-14, i.e., first 15 epochs)
    # Phase 2b: Perceptual Refinement (epochs 15+)
    is_phase2a = epoch < 15
    
    if is_phase2a:
        # Phase 2a: Rate Anchoring Strategy
        # Goal: Establish BPP at 0.17 while building reconstruction quality
        config_dict = {
            # ELIC Architecture (unchanged)
            "N": 192,
            "M": 320,
            "slice_num": 5,
            "context_window": 5,
            "slice_ch": [16, 16, 32, 64, 192],
            "quant": "ste",
            
            # Phase 2a: Rate Anchoring
            "lambda_char": 2e-6,      # Charbonnier (fallback, not used with RD)
            "lambda_rd": 0.05,        # STRONG reconstruction weight for PSNR
            
            # Perceptual losses - Minimal to avoid rate inflation
            "lambda_lpips": 0.8,      # Reduced LPIPS
            "lambda_style": 50,       # Reduced style loss
            "lambda_gan": 0.5,        # Reduced GAN
            "lambda_dists": 0.1,      # MINIMAL DISTS (texture)
            "lambda_pieapp": 0.1,     # MINIMAL PIEAPP (structure)
            
            # Rate control - MAXIMUM
            "lambda_rate": 1.0,       # AGGRESSIVE rate control to anchor BPP
            "target_bpp": 0.17,       # Target BPP for adaptive rate controller
            
            # Other
            "lambda_face": 0,
            
            # Phase indicator (for logging)
            "phase": "2a_rate_anchor",
        }
    else:
        # Phase 2b: Perceptual Refinement Strategy
        # Goal: Maximize perceptual quality while maintaining BPP at 0.17
        config_dict = {
            # ELIC Architecture (unchanged)
            "N": 192,
            "M": 320,
            "slice_num": 5,
            "context_window": 5,
            "slice_ch": [16, 16, 32, 64, 192],
            "quant": "ste",
            
            # Phase 2b: Perceptual Refinement
            "lambda_char": 2e-6,      # Charbonnier (fallback, not used with RD)
            "lambda_rd": 0.02,        # Moderate RD (let perceptual losses take over)
            
            # Perceptual losses - FULL POWER
            "lambda_lpips": 1.5,      # Increased LPIPS for perceptual quality
            "lambda_style": 100,      # Full style loss
            "lambda_gan": 1.0,        # Full GAN for texture
            "lambda_dists": 0.4,       # ACTIVE DISTS (texture-aware)
            "lambda_pieapp": 0.5,     # ACTIVE PIEAPP (structure-aware)
            
            # Rate control - Moderate (BPP already anchored)
            "lambda_rate": 0.5,       # Relaxed but still controlled
            "target_bpp": 0.17,       # Target BPP for adaptive rate controller
            
            # Other
            "lambda_face": 0,
            
            # Phase indicator (for logging)
            "phase": "2b_perceptual",
        }
    
    return Config(config_dict)
