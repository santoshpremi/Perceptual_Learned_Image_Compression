import math
import torch
import torch.nn as nn
from pytorch_msssim import ms_ssim
from loss import perceptual_loss as ps

from models.vgg import Vgg16 

# Add imports for DISTS (Phase 2 only)
try:
    from piq import DISTS  # DISTS perceptual metric for Phase 2
    PIQ_AVAILABLE = True
except ImportError:
    print("Warning: piq library not found. Install with: pip install piq")
    PIQ_AVAILABLE = False

# Add imports for VSI (Phase 2, Full-Reference metric) via pyiqa
try:
    import pyiqa
    VSI_AVAILABLE = True  # set False in __init__ if create_metric fails
except ImportError:
    VSI_AVAILABLE = False
    pyiqa = None
    print("Warning: pyiqa not found. Install with: pip install pyiqa") 

class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, metrics='mse'):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.metrics = metrics

    def set_lmbda(self, lmbda):
        self.lmbda = lmbda

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        if self.metrics == 'mse':
            out["mse_loss"] = self.mse(output["x_hat"], target)
            out["ms_ssim_loss"] = None
            out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]
        elif self.metrics == 'ms-ssim':
            out["mse_loss"] = None
            out["ms_ssim_loss"] = 1 - ms_ssim(output["x_hat"], target, data_range=1.0)
            out["loss"] = self.lmbda * out["ms_ssim_loss"] + out["bpp_loss"]

        return out


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        return torch.mean(torch.sqrt((x - y).pow(2) + self.eps**2))
    
class GANLoss(nn.Module):
    """Define GAN loss.
    Args:
        gan_type (str): Support 'vanilla', 'lsgan', 'wgan', 'hinge'.
        real_label_val (float): The value for real label. Default: 1.0.
        fake_label_val (float): The value for fake label. Default: 0.0.
        loss_weight (float): Loss weight. Default: 1.0.
            Note that loss_weight is only for generators; and it is always 1.0
            for discriminators.
    """

    def __init__(self,
                 gan_type = 'hinge',
                 real_label_val=1.0,
                 fake_label_val=0.0,
                 loss_weight=1.0):
        super().__init__()
        self.gan_type = gan_type
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val
        self.loss_weight = loss_weight

        if self.gan_type == 'hinge':
            self.loss = nn.ReLU()
        else:
            raise NotImplementedError(
                f'GAN type {self.gan_type} is not implemented.')

    def _wgan_loss(self, input, target):
        """wgan loss.
        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.
        Returns:
            Tensor: wgan loss.
        """

        return -input.mean() if target else input.mean()

    def get_target_label(self, input, target_is_real):
        """Get target label.
        Args:
            input (Tensor): Input tensor.
            target_is_real (bool): Whether the target is real or fake.
        Returns:
            (bool | Tensor): Target tensor. Return bool for wgan, otherwise,
                return Tensor.
        """

        target_val = (
            self.real_label_val if target_is_real else self.fake_label_val)
        return input.new_ones(input.size()) * target_val

    def forward(self, input, target_is_real, is_disc=False, mask=None):
        """
        Args:
            input (Tensor): The input for the loss module, i.e., the network
                prediction.
            target_is_real (bool): Whether the target is real or fake.
            is_disc (bool): Whether the loss for discriminators or not.
                Default: False.
        Returns:
            Tensor: GAN loss value.
        """

        target_label = self.get_target_label(input, target_is_real)
        if self.gan_type == 'hinge':
            if is_disc:  # for discriminators in hinge-gan
                input = -input if target_is_real else input
                loss = self.loss(1 + input).mean()
            else:  # for generators in hinge-gan
                loss = -input.mean()
        else:  # other gan types
            loss = self.loss(input, target_label)

        # loss_weight is always 1.0 for discriminators
        return loss if is_disc else loss * self.loss_weight

class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()
        self.mae = nn.L1Loss()

    def gram_matrix(self, x):
        (bs, ch, h, w) = x.size()
        f = x.view(bs, ch, w*h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (ch * h * w)
        return G

    def forward(self,target_feature, inputs):
        G = self.gram_matrix(inputs)
        target = self.gram_matrix(target_feature)
        self.loss = self.mae(G, target)
        return self.loss 

class RateDistortionPOELICLoss(nn.Module):
    """Phase 1 loss: Original HFLIC loss with Charbonnier and LPIPS.
    
    Uses Charbonnier loss and LPIPS perceptual metric as in original HFLIC paper.
    """

    def __init__(self, lmbda=1e-2, device="cuda", gpu_id=None, metrics='mse'):
        super().__init__()
        # Use Charbonnier loss (original HFLIC)
        self.charbonnier = CharbonnierLoss()
        self.style = StyleLoss()
        self.lpips = ps.PerceptualLoss(model='net-lin', net='vgg',
                               use_gpu=torch.cuda.is_available(), gpu_ids=gpu_id)
        print(gpu_id)
        self.lmbda = lmbda
        self.metrics = metrics
        self.vgg = Vgg16().to(device).eval()

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W
        
        # Compute BPP loss
        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
    
        x_hat_feat = self.vgg(output["x_hat"])
        target_feat = self.vgg(target)

        # Use Charbonnier loss (original HFLIC)
        out["charbonnier"] = self.charbonnier(output["x_hat"], target)
        out["rd_loss"] = None  # Not used in original HFLIC
        
        out["lpips"] = self.lpips(output["x_hat"], target).mean()
        
        x_hat_feat = [feat for feat in x_hat_feat]
        target_feat = [feat for feat in target_feat]
        style_loss = 0.0
        for i in range(4):
            style_loss += self.style(x_hat_feat[i], target_feat[i])
        out["style_loss"] = style_loss

        return out


class RateDistortionPOELICLossPhase2(nn.Module):
    """Phase 2 loss: RD + LPIPS + (1-VSI) + Style + GAN + bpp.
    
    Training loss: RD + LPIPS + (1-VSI) + Style + GAN + bpp rate.
    
    LPIPS: Learned Perceptual Image Patch Similarity (VGG-based)
           - Full-reference, low-level perceptual similarity
           
    VSI: Visual Saliency-induced Index (full-reference)
         - Quality weighted by visual saliency (where humans look)
         - Higher = better; use (1 - VSI) as loss
    """

    def __init__(self, lmbda=1e-2, device="cuda", gpu_id=None, metrics='mse'):
        super().__init__()
        self.mse = nn.MSELoss()
        self.gan = GANLoss()
        self.style = StyleLoss()
        self.device = device
        
        # LPIPS: Full-reference perceptual loss (used in training)
        self.lpips = ps.PerceptualLoss(model='net-lin', net='vgg',
                                       use_gpu=torch.cuda.is_available(), gpu_ids=gpu_id)

        # VSI: Visual Saliency-induced Index (full-reference, higher=better)
        if pyiqa is not None:
            try:
                self.vsi_metric = pyiqa.create_metric('vsi', device=device).eval()
                self._vsi_available = True
                print("✓ VSI (FR) initialized: Visual Saliency-induced Index")
            except Exception as e:
                self.vsi_metric = None
                self._vsi_available = False
                print(f"✗ VSI not available: {e}")
        else:
            self.vsi_metric = None
            self._vsi_available = False
            print("✗ VSI not available (pyiqa required)")

        print(f"GPU ID: {gpu_id}")
        self.lmbda = lmbda
        self.metrics = metrics
        self.vgg = Vgg16().to(device).eval()

    def forward(self, output, target):
        """
        Forward pass for Phase 2 loss computation.
        
        Args:
            output: Model output dict with 'x_hat' and 'likelihoods'
            target: Ground truth images
        """
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )

        x_hat_feat = self.vgg(output["x_hat"])
        target_feat = self.vgg(target)

        out["rd_loss"] = self.mse(output["x_hat"], target) * (255 ** 2)
        out["charbonnier"] = None

        # LPIPS: Full-reference perceptual loss (used in training)
        out["lpips"] = self.lpips(output["x_hat"], target).mean()

        # VSI: Visual Saliency-induced Index (FR, higher=better, use 1-VSI as loss)
        if self._vsi_available and self.vsi_metric is not None:
            x_hat_clamped = torch.clamp(output["x_hat"], 0.0, 1.0)
            target_clamped = torch.clamp(target, 0.0, 1.0)
            if torch.isfinite(x_hat_clamped).all() and torch.isfinite(target_clamped).all():
                try:
                    vsi_score = self.vsi_metric(x_hat_clamped, target_clamped)
                    out["vsi"] = vsi_score.mean() if vsi_score.numel() > 1 else vsi_score
                except Exception as e:
                    print(f"VSI computation failed: {e}")
                    out["vsi"] = torch.tensor(0.0, device=output["x_hat"].device, dtype=output["x_hat"].dtype)
            else:
                out["vsi"] = torch.tensor(0.0, device=output["x_hat"].device, dtype=output["x_hat"].dtype)
        else:
            out["vsi"] = torch.tensor(0.0, device=output["x_hat"].device, dtype=output["x_hat"].dtype, requires_grad=False)

        x_hat_feat = [feat for feat in x_hat_feat]
        target_feat = [feat for feat in target_feat]
        style_loss = 0.0
        for i in range(4):
            style_loss += self.style(x_hat_feat[i], target_feat[i])
        out["style_loss"] = style_loss

        return out

class RateDistortionPOELICLossBaseline(nn.Module):
    """Phase 2 BASELINE loss: Original HFLIC - Charbonnier + LPIPS + Style + GAN + BPP.
    
    This is the ORIGINAL HFLIC Phase 2 loss for fair comparison.
    NO MSE, NO VSI - just the original HFLIC components.
    
    Loss = λ_char × Charbonnier(sum) + λ_lpips × LPIPS + λ_style × Style + λ_gan × GAN + λ_bpp × BPP
    
    HFLIC hardcoded weights: 3e-4 × Char + 2.0 × LPIPS + 1.0 × Style + 1.0 × GAN + 1.0 × BPP
    """

    def __init__(self, lmbda=1e-2, device="cuda", gpu_id=None, metrics='mse'):
        super().__init__()
        self.charbonnier = CharbonnierLoss()
        self.gan = GANLoss()
        self.style = StyleLoss()
        self.device = device
        
        self.lpips = ps.PerceptualLoss(model='net-lin', net='vgg',
                                       use_gpu=torch.cuda.is_available(), gpu_ids=gpu_id)

        print(f"[BASELINE] Phase 2 initialized: Charbonnier + LPIPS + Style + GAN + BPP")
        print(f"GPU ID: {gpu_id}")
        self.lmbda = lmbda
        self.metrics = metrics
        self.vgg = Vgg16().to(device).eval()

    def forward(self, output, target):
        """
        Forward pass for BASELINE Phase 2 loss computation.
        
        Args:
            output: Model output dict with 'x_hat' and 'likelihoods'
            target: Ground truth images
        """
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )

        x_hat_feat = self.vgg(output["x_hat"])
        target_feat = self.vgg(target)

        out["charbonnier"] = self.charbonnier(output["x_hat"], target)
        out["rd_loss"] = None
        out["vsi"] = None

        out["lpips"] = self.lpips(output["x_hat"], target).mean()

        x_hat_feat = [feat for feat in x_hat_feat]
        target_feat = [feat for feat in target_feat]
        style_loss = 0.0
        for i in range(4):
            style_loss += self.style(x_hat_feat[i], target_feat[i])
        out["style_loss"] = style_loss

        return out


class RateDistortionPOELICLossDISTSLPIPS(nn.Module):
    """Phase 2 loss: λ_rd × MSE + λ_lpips × LPIPS + λ_dists × DISTS + λ_style × Style + λ_gan × GAN + λ_bpp × BPP.
    
    Enhanced method combining DISTS (image-level distortion) with LPIPS (patch-level perceptual similarity).
    
    Loss = λ_rd × MSE + λ_lpips × LPIPS + λ_dists × DISTS + λ_style × Style + λ_gan × GAN + λ_bpp × BPP
    
    DISTS: Deep Image Structure and Texture Similarity (better for structure preservation)
    LPIPS: Learned Perceptual Image Patch Similarity (better for local details)
    """

    def __init__(self, lmbda=1e-2, device="cuda", gpu_id=None, metrics='mse'):
        super().__init__()
        self.mse = nn.MSELoss()
        self.gan = GANLoss()
        self.style = StyleLoss()
        self.device = device
        
        # LPIPS: Full-reference perceptual loss (patch-level)
        self.lpips = ps.PerceptualLoss(model='net-lin', net='vgg',
                                       use_gpu=torch.cuda.is_available(), gpu_ids=gpu_id)

        # DISTS: Deep Image Structure and Texture Similarity (image-level)
        if PIQ_AVAILABLE:
            self.dists = DISTS().to(device).eval()
            self._dists_available = True
            print("✓ DISTS (FR) initialized: Deep Image Structure and Texture Similarity")
        else:
            self.dists = None
            self._dists_available = False
            print("✗ DISTS not available. Install with: pip install piq")

        print(f"[OURS] Phase 2 initialized: λ_rd × MSE + λ_lpips × LPIPS + λ_dists × DISTS + λ_style × Style + λ_gan × GAN + λ_bpp × BPP")
        print(f"GPU ID: {gpu_id}")
        self.lmbda = lmbda
        self.metrics = metrics
        self.vgg = Vgg16().to(device).eval()

    def forward(self, output, target):
        """
        Forward pass for DISTS+LPIPS Phase 2 loss computation.
        
        Args:
            output: Model output dict with 'x_hat' and 'likelihoods'
            target: Ground truth images
        """
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )

        x_hat_feat = self.vgg(output["x_hat"])
        target_feat = self.vgg(target)

        # Keep MSE in pixel space so it has comparable scale to the
        # previously successful DISTS+LPIPS runs.
        out["rd_loss"] = self.mse(output["x_hat"], target) * (255 ** 2)
        out["charbonnier"] = None

        # LPIPS: Patch-level perceptual similarity
        out["lpips"] = self.lpips(output["x_hat"], target).mean()

        # DISTS: Image-level structure and texture similarity
        if self._dists_available and self.dists is not None:
            x_hat_clamped = torch.clamp(output["x_hat"], 0.0, 1.0)
            target_clamped = torch.clamp(target, 0.0, 1.0)
            if torch.isfinite(x_hat_clamped).all() and torch.isfinite(target_clamped).all():
                try:
                    dists_score = self.dists(x_hat_clamped, target_clamped)
                    out["dists"] = dists_score.mean() if dists_score.numel() > 1 else dists_score
                except Exception as e:
                    print(f"DISTS computation failed: {e}")
                    out["dists"] = torch.tensor(0.0, device=output["x_hat"].device, dtype=output["x_hat"].dtype)
            else:
                out["dists"] = torch.tensor(0.0, device=output["x_hat"].device, dtype=output["x_hat"].dtype)
        else:
            out["dists"] = torch.tensor(0.0, device=output["x_hat"].device, dtype=output["x_hat"].dtype, requires_grad=False)

        x_hat_feat = [feat for feat in x_hat_feat]
        target_feat = [feat for feat in target_feat]
        style_loss = 0.0
        for i in range(4):
            style_loss += self.style(x_hat_feat[i], target_feat[i])
        out["style_loss"] = style_loss

        # VSI not used for DISTS+LPIPS method (kept for compatibility)
        out["vsi"] = torch.tensor(0.0, device=output["x_hat"].device, dtype=output["x_hat"].dtype, requires_grad=False)

        return out


class RateDistortionPOELICFaceLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter for face images."""

    def __init__(self, lmbda=1e-2, device = "cuda", gpu_id=None):
        super().__init__()
        self.charbonnier = CharbonnierLoss()
        self.gan = GANLoss()
        self.style = StyleLoss()
        
        # Initialize DISTS for perceptual loss
        if PIQ_AVAILABLE:
            self.dists = DISTS().to(device).eval()
            print("DISTS loss initialized successfully")
        else:
            self.dists = None
            print("Warning: DISTS not available. Install piq library.")
        
        self.mse = nn.MSELoss()

        print(gpu_id)
        self.lmbda = lmbda
        self.vgg = Vgg16().to(device).eval()


    def forward(self, output, target, mask):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W
        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        x_hat = output["x_hat"]
        x_tidle = mask * target + (1-mask) * x_hat
        out["x_tidle"] = x_tidle

        kernel_h = 16
        kernel_w = 16
        x_tidle_patch = x_tidle.unfold(3, kernel_h, kernel_w).unfold(2, kernel_h, kernel_w).permute(2, 3, 0, 1, 4, 5).reshape(-1, 3, kernel_h, kernel_w)
        target_patch = target.unfold(3, kernel_h, kernel_w).unfold(2, kernel_h, kernel_w).permute(2, 3, 0, 1, 4, 5).reshape(-1, 3, kernel_h, kernel_w)
        # print(x_tidle_patch.size(), target_patch.size())
        x_tidle_feat  = self.vgg(x_tidle_patch)
        target_feat = self.vgg(target_patch)

        # x_tidle_feat  = self.vgg(x_tidle)
        # target_feat = self.vgg(target)
        out["charbonnier"] = self.charbonnier((1-mask) * x_hat, target)
        
        # Compute DISTS loss
        if self.dists is not None:
            x_tidle_clamped = torch.clamp(x_tidle, 0.0, 1.0)
            target_clamped = torch.clamp(target, 0.0, 1.0)
            out["dists"] = self.dists(x_tidle_clamped, target_clamped)
        else:
            out["dists"] = torch.tensor(0.0, device=output["x_hat"].device, requires_grad=False)
        
        x_tidle_feat  = [feat for feat in  x_tidle_feat]
        target_feat = [feat for feat in  target_feat]
        style_loss = 0.0
        for i in range(4):
            style_loss += self.style(x_tidle_feat[i], target_feat[i])
            # print(x_tidle_feat[i].size(), target_feat[i].size())
        out["style_loss"]  = style_loss / 4.0
        out["face_loss"] =  self.mse(mask * target, mask * x_hat)
        return out
