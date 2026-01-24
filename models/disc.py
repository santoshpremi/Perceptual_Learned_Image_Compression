import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from torch.nn.utils import spectral_norm

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


class Conv2d(nn.Module):
    def __init__(self, nch_in, nch_out, kernel_size=4, stride=1, padding=1, bias=True, snorm=False):
        super(Conv2d, self).__init__()
        if snorm:
            # self.conv = SpectralNorm(nn.Conv2d(nch_in, nch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
            self.conv = spectral_norm(nn.Conv2d(nch_in, nch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        else:
            self.conv = nn.Conv2d(nch_in, nch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        return self.conv(x)

class Norm2d(nn.Module):
    def __init__(self, nch, norm_mode):
        super(Norm2d, self).__init__()
        if norm_mode == 'bnorm':
            self.norm = nn.BatchNorm2d(nch)
        elif norm_mode == 'inorm':
            self.norm = nn.InstanceNorm2d(nch)

    def forward(self, x):
        return self.norm(x)


class ReLU(nn.Module):
    def __init__(self, relu):
        super(ReLU, self).__init__()
        if relu > 0:
            self.relu = nn.LeakyReLU(relu, True)
        elif relu == 0:
            self.relu = nn.ReLU(True)

    def forward(self, x):
        return self.relu(x)

class CNR2d(nn.Module):
    def __init__(self, nch_in, nch_out, kernel_size=4, stride=1, padding=1, norm='bnorm', relu=0.0, drop=[], bias=[], snorm=False):
        super().__init__()

        if bias == []:
            if norm == 'bnorm':
                bias = False
            else:
                bias = True

        layers = []
        layers += [Conv2d(nch_in, nch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, snorm=snorm)]

        # if snorm:
        #     layers += [SpectralNorm(layers[-1].conv)]

        if norm != []:
            layers += [Norm2d(nch_out, norm)]

        if relu != []:
            layers += [ReLU(relu)]

        if drop != []:
            layers += [nn.Dropout2d(drop)]

        self.cbr = nn.Sequential(*layers)

    def forward(self, x):
        return self.cbr(x)

class Discriminator(nn.Module):
    def __init__(self, nch_in=3, nch_ker=64, norm='bnorm'):
        super(Discriminator, self).__init__()

        self.nch_in = nch_in
        self.nch_ker = nch_ker
        self.norm = norm

        if norm == 'bnorm':
            self.bias = False
        else:
            self.bias = True

        # dsc1 : 256 x 256 x 3 -> 128 x 128 x 64
        # dsc2 : 128 x 128 x 64 -> 64 x 64 x 128
        # dsc3 : 64 x 64 x 128 -> 32 x 32 x 256
        # dsc4 : 32 x 32 x 256 -> 32 x 32 x 512
        # dsc5 : 32 x 32 x 512 -> 32 x 32 x 1

        self.dsc1 = CNR2d(1 * self.nch_in,  1 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.2, snorm=True)
        self.dsc2 = CNR2d(1 * self.nch_ker, 2 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.2, snorm=True)
        self.dsc3 = CNR2d(2 * self.nch_ker, 4 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.2, snorm=True)
        self.dsc4 = CNR2d(4 * self.nch_ker, 8 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.2, snorm=True)
        self.dsc5 = CNR2d(8 * self.nch_ker, 1,                kernel_size=4, stride=1, padding=1, norm=[],        relu=[], bias=False)

        # self.dsc1 = CNR2d(1 * self.nch_in,  1 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=[], relu=0.2)
        # self.dsc2 = CNR2d(1 * self.nch_ker, 2 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=[], relu=0.2)
        # self.dsc3 = CNR2d(2 * self.nch_ker, 4 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=[], relu=0.2)
        # self.dsc4 = CNR2d(4 * self.nch_ker, 8 * self.nch_ker, kernel_size=4, stride=1, padding=1, norm=[], relu=0.2)
        # self.dsc5 = CNR2d(8 * self.nch_ker, 1,                kernel_size=4, stride=1, padding=1, norm=[], relu=[], bias=False)
        
    def forward(self, x):

        x = self.dsc1(x)
        x = self.dsc2(x)
        x = self.dsc3(x)
        x = self.dsc4(x)
        x = self.dsc5(x)

        x = torch.sigmoid(x)

        return x


class MultiScaleDiscriminator(nn.Module):
    """Multi-Scale Discriminator for GAN stability.
    
    Uses multiple discriminators at different scales to provide feedback
    on both global structure (low frequency) and fine texture (high frequency).
    This helps prevent mode collapse and improves training stability.
    
    Reference: "High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs"
    """
    
    def __init__(self, num_scales=3, nch_in=3, nch_ker=64, norm='bnorm'):
        super(MultiScaleDiscriminator, self).__init__()
        
        self.num_scales = num_scales
        
        # Create discriminators for each scale
        self.discriminators = nn.ModuleList([
            Discriminator(nch_in=nch_in, nch_ker=nch_ker, norm=norm)
            for _ in range(num_scales)
        ])
        
        # Downsampling layer for multi-scale processing
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
    
    def forward(self, x):
        """Forward pass through all scales.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            List of discriminator outputs at each scale
        """
        outputs = []
        for i, disc in enumerate(self.discriminators):
            outputs.append(disc(x))
            if i < self.num_scales - 1:  # Don't downsample after last discriminator
                x = self.downsample(x)
        return outputs


def r1_gradient_penalty(real_pred, real_img, weight=10.0):
    """R1 Gradient Penalty for GAN training stability.
    
    Penalizes the discriminator for having large gradients on real images,
    which helps prevent the discriminator from becoming too strong and
    causing training instability.
    
    Reference: "Which Training Methods for GANs do actually Converge?"
    
    Args:
        real_pred: Discriminator prediction on real images
        real_img: Real image tensor (requires_grad=True)
        weight: Penalty weight (default: 10.0)
        
    Returns:
        R1 gradient penalty loss
    """
    # Handle multi-scale discriminator output
    if isinstance(real_pred, list):
        # Average penalty across all scales
        penalty = 0.0
        for pred in real_pred:
            grad_real, = torch.autograd.grad(
                outputs=pred.sum(),
                inputs=real_img,
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )
            penalty += grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
        penalty = penalty / len(real_pred)
    else:
        grad_real, = torch.autograd.grad(
            outputs=real_pred.sum(),
            inputs=real_img,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )
        penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
    
    return weight * penalty


def compute_discriminator_loss(pred_real, pred_fake, gan_loss_fn, real_img=None, r1_weight=10.0, use_r1=True):
    """Compute total discriminator loss with optional R1 penalty.
    
    Args:
        pred_real: Discriminator predictions on real images
        pred_fake: Discriminator predictions on fake images
        gan_loss_fn: GAN loss function
        real_img: Real images (needed for R1 penalty, requires_grad=True)
        r1_weight: R1 penalty weight
        use_r1: Whether to apply R1 gradient penalty
        
    Returns:
        Total discriminator loss
    """
    # Handle multi-scale discriminator
    if isinstance(pred_real, list):
        loss_D_real = 0.0
        loss_D_fake = 0.0
        for pred_r, pred_f in zip(pred_real, pred_fake):
            loss_D_real += gan_loss_fn(pred_r, True, is_disc=True)
            loss_D_fake += gan_loss_fn(pred_f, False, is_disc=True)
        loss_D_real = loss_D_real / len(pred_real)
        loss_D_fake = loss_D_fake / len(pred_fake)
    else:
        loss_D_real = gan_loss_fn(pred_real, True, is_disc=True)
        loss_D_fake = gan_loss_fn(pred_fake, False, is_disc=True)
    
    loss_D_total = (loss_D_real + loss_D_fake) * 0.5
    
    # Add R1 gradient penalty
    if use_r1 and real_img is not None:
        r1_loss = r1_gradient_penalty(pred_real, real_img, weight=r1_weight)
        loss_D_total = loss_D_total + r1_loss
    
    return loss_D_total


def compute_generator_loss(pred_fake, gan_loss_fn):
    """Compute generator adversarial loss.
    
    Args:
        pred_fake: Discriminator predictions on fake images
        gan_loss_fn: GAN loss function
        
    Returns:
        Generator adversarial loss
    """
    # Handle multi-scale discriminator
    if isinstance(pred_fake, list):
        loss_G = 0.0
        for pred_f in pred_fake:
            loss_G += gan_loss_fn(pred_f, False, is_disc=False)
        loss_G = loss_G / len(pred_fake)
    else:
        loss_G = gan_loss_fn(pred_fake, False, is_disc=False)
    
    return loss_G
