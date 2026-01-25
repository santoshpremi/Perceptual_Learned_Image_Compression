import math
import torch
import torch.nn as nn
from pytorch_msssim import ms_ssim
from loss import perceptual_loss as ps

from models.vgg import Vgg16 

# Add imports for pyiqa library
# Using TOPIQ-FR only - state-of-the-art Full-Reference IQA metric from CVPR 2023
try:
    import pyiqa
    PYIQA_AVAILABLE = True
    print("pyiqa library loaded successfully")
except ImportError:
    print("Warning: pyiqa library not found. Install with: pip install pyiqa")
    PYIQA_AVAILABLE = False

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
        return torch.sqrt((x - y).pow(2) + self.eps**2).mean()
    
class GANLoss(nn.Module):
    """Define GAN loss.
    
    IMPORTANT: Using 'vanilla' (BCE) loss to match original HFLIC paper.
    Original HFLIC uses sigmoid discriminator output with non-saturating GAN loss.
    Reference: https://github.com/beiluo97/HFLIC
    
    Args:
        gan_type (str): Support 'vanilla' (BCE), 'hinge'. Default: 'vanilla' for HFLIC.
        real_label_val (float): The value for real label. Default: 1.0.
        fake_label_val (float): The value for fake label. Default: 0.0.
        loss_weight (float): Loss weight. Default: 1.0.
            Note that loss_weight is only for generators; and it is always 1.0
            for discriminators.
    """

    def __init__(self,
                 gan_type='vanilla',  # Changed from 'hinge' to 'vanilla' to match original HFLIC
                 real_label_val=1.0,
                 fake_label_val=0.0,
                 loss_weight=1.0):
        super().__init__()
        self.gan_type = gan_type
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val
        self.loss_weight = loss_weight

        if self.gan_type == 'vanilla':
            # Non-saturating GAN loss (BCE) - matches original HFLIC/HiFiC
            # Works with sigmoid discriminator output [0, 1]
            self.loss = nn.BCELoss()
        elif self.gan_type == 'hinge':
            # Hinge loss - requires unbounded logits (no sigmoid)
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
        
        if self.gan_type == 'vanilla':
            # BCE loss for non-saturating GAN (original HFLIC)
            # Works with sigmoid discriminator output
            loss = self.loss(input, target_label)
        elif self.gan_type == 'hinge':
            if is_disc:  # for discriminators in hinge-gan
                input = -input if target_is_real else input
                loss = self.loss(1 + input).mean()
            else:  # for generators in hinge-gan
                loss = -input.mean()
        else:
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
    """Phase 1 loss: Original HFLIC loss with Charbonnier.
    
    Uses Charbonnier loss as in original HFLIC paper for exact reproduction.
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
        out["style_loss"] = style_loss / 4.0

        return out


class RateDistortionPOELICLossPhase2(nn.Module):
    """Phase 2 loss: Enhanced HFLIC Phase 2 loss with MSE, GAN, and TOPIQ-FR.
    
    Uses MSE (Rate-Distortion) loss for Phase 2.
    Includes: MSE (RD) + LPIPS + Style + GAN + Rate (BPP) + TOPIQ-FR
    
    Using pyiqa library for:
    - TOPIQ-FR: State-of-the-art Full-Reference IQA metric from CVPR 2023
    """

    def __init__(self, lmbda=1e-2, device="cuda", gpu_id=None, metrics='mse'):
        super().__init__()
        self.device = device
        
        # Use MSE for rate-distortion loss in Phase 2
        self.mse = nn.MSELoss()
        self.gan = GANLoss()
        self.style = StyleLoss()
        self.lpips = ps.PerceptualLoss(model='net-lin', net='vgg',
                               use_gpu=torch.cuda.is_available(), gpu_ids=gpu_id)
        
        # Initialize TOPIQ-FR from pyiqa library
        self.topiq = None
        
        if PYIQA_AVAILABLE:
            try:
                # TOPIQ-FR: State-of-the-art Full-Reference IQA (CVPR 2023)
                # Higher is better - need to negate for loss
                self.topiq = pyiqa.create_metric('topiq_fr', device=device)
                print("pyiqa TOPIQ-FR initialized successfully")
            except Exception as e:
                print(f"Warning: Could not initialize pyiqa TOPIQ: {e}")
                self.topiq = None
        else:
            print("Warning: pyiqa not available. Install with: pip install pyiqa")
        
        print(f"GPU IDs: {gpu_id}")
        self.lmbda = lmbda
        self.metrics = metrics
        self.vgg = Vgg16().to(device).eval()

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W
        
        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
    
        x_hat_feat = self.vgg(output["x_hat"])
        target_feat = self.vgg(target)

        # Phase 2: Use MSE (Rate-Distortion) loss
        out["rd_loss"] = self.mse(output["x_hat"], target)
        out["charbonnier"] = None  # Not used in Phase 2
        
        out["lpips"] = self.lpips(output["x_hat"], target).mean()
        
        # Clamp images to [0, 1] for perceptual metrics
        x_hat_clamped = torch.clamp(output["x_hat"], 0.0, 1.0)
        target_clamped = torch.clamp(target, 0.0, 1.0)
        
        # Compute TOPIQ-FR loss using pyiqa (state-of-the-art FR metric from CVPR 2023)
        # TOPIQ is "higher is better" so negate for loss: loss = 1 - topiq
        if self.topiq is not None:
            try:
                topiq_score = self.topiq(x_hat_clamped, target_clamped).mean()
                out["topiq"] = 1.0 - topiq_score  # Convert to loss (lower is better)
            except Exception as e:
                out["topiq"] = torch.tensor(0.0, device=output["x_hat"].device, requires_grad=False)
        else:
            out["topiq"] = torch.tensor(0.0, device=output["x_hat"].device, requires_grad=False)
        
        x_hat_feat = [feat for feat in x_hat_feat]
        target_feat = [feat for feat in target_feat]
        style_loss = 0.0
        for i in range(4):
            style_loss += self.style(x_hat_feat[i], target_feat[i])
        out["style_loss"] = style_loss / 4.0

        return out

class RateDistortionPOELICFaceLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, device = "cuda", gpu_id=None):
        super().__init__()
        self.charbonnier = CharbonnierLoss()
        self.gan = GANLoss()
        self.style = StyleLoss()
        # self.lpips = lpips.LPIPS(net='vgg').cuda()
        self.lpips =  ps.PerceptualLoss(model='net-lin', net='vgg',
                               use_gpu=torch.cuda.is_available(),gpu_ids=gpu_id)
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
        out["lpips"]       = self.lpips(x_tidle, target).mean()
        
        x_tidle_feat  = [feat for feat in  x_tidle_feat]
        target_feat = [feat for feat in  target_feat]
        style_loss = 0.0
        for i in range(4):
            style_loss += self.style(x_tidle_feat[i], target_feat[i])
            # print(x_tidle_feat[i].size(), target_feat[i].size())
        out["style_loss"]  = style_loss / 4.0
        out["face_loss"] =  self.mse(mask * target, mask * x_hat)
        return out
