import torch
import torch.nn as nn
import torch.nn.functional as F
#from utils.dist import *
from loss.rd_loss import GANLoss
from models.disc import r1_gradient_penalty, compute_discriminator_loss, compute_generator_loss


def train_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, logger_train, tb_logger, current_step, config=None
):
    model.train()
    device = next(model.parameters()).device

    for i, d in enumerate(train_dataloader):
        d = d.to(device)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d)

        out_criterion = criterion(out_net, d)
        # Phase 1: Original HFLIC - Charbonnier + LPIPS + Style + Rate (BPP)
        # Total loss = lambda_char * Charbonnier + lambda_lpips * LPIPS + lambda_style * Style + lambda_rate * Rate (BPP)
        if config is not None:
            # Check if using Charbonnier (original HFLIC) or RD loss (for backward compatibility)
            if "charbonnier" in out_criterion and out_criterion.get("charbonnier") is not None:
                # Original HFLIC: Use Charbonnier loss
                total_loss = (config.get("lambda_char", 2e-6) * out_criterion["charbonnier"] + 
                         config["lambda_lpips"] * out_criterion["lpips"] + 
                         config["lambda_style"] * out_criterion["style_loss"] + 
                         config["lambda_rate"] * out_criterion["bpp_loss"])
            elif "rd_loss" in out_criterion and out_criterion["rd_loss"] is not None:
                # Fallback to RD loss if Charbonnier not available (backward compatibility)
                total_loss = (config.get("lambda_rd", 1e-2) * out_criterion["rd_loss"] + 
                             config["lambda_lpips"] * out_criterion["lpips"] + 
                             config["lambda_style"] * out_criterion["style_loss"] + 
                             config["lambda_rate"] * out_criterion["bpp_loss"])
        else:
            # Fallback to unweighted sum if config not provided
            if "charbonnier" in out_criterion and out_criterion.get("charbonnier") is not None:
                # Original HFLIC: Use Charbonnier loss
                total_loss = (out_criterion["charbonnier"] + 
                         out_criterion["lpips"] + 
                         out_criterion["style_loss"] + 
                         out_criterion["bpp_loss"])
            elif "rd_loss" in out_criterion and out_criterion["rd_loss"] is not None:
                # Fallback to RD loss if Charbonnier not available
                total_loss = (out_criterion["rd_loss"] + 
                             out_criterion["lpips"] + 
                             out_criterion["style_loss"] + 
                             out_criterion["bpp_loss"])
        out_criterion["loss"] = total_loss
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        current_step += 1
        if current_step % 100 == 0:
            tb_logger.add_scalar('{}'.format('[train]: loss'), out_criterion["loss"].item(), current_step)
            tb_logger.add_scalar('{}'.format('[train]: bpp_loss'), out_criterion["bpp_loss"].item(), current_step)
            if out_criterion.get("rd_loss") is not None:
                tb_logger.add_scalar('{}'.format('[train]: rd_loss'), out_criterion["rd_loss"].item(), current_step)
            if out_criterion.get("mse_loss") is not None:
                tb_logger.add_scalar('{}'.format('[train]: mse_loss'), out_criterion["mse_loss"].item(), current_step)
            if out_criterion.get("ms_ssim_loss") is not None:
                tb_logger.add_scalar('{}'.format('[train]: ms_ssim_loss'), out_criterion["ms_ssim_loss"].item(), current_step)

        if i % 100 == 0:
            # For Phase 1 training with perceptual losses
            if "rd_loss" in out_criterion and out_criterion["rd_loss"] is not None:
                logger_train.info(
                    f"Train epoch {epoch + 1}: ["
                    f"{i*len(d):5d}/{len(train_dataloader.dataset)}"
                    f" ({100. * i / len(train_dataloader):.0f}%)] "
                    f'Loss: {out_criterion["loss"].item():.4f} | '
                    f'RD loss: {out_criterion["rd_loss"].item():.4f} | '
                    f'LPIPS loss: {out_criterion["lpips"].item():.4f} | '
                    f'Style loss: {out_criterion["style_loss"].item():.4f} | '
                    f'Bpp loss: {out_criterion["bpp_loss"].item():.2f} | '
                    f"Aux loss: {aux_loss.item():.2f}"
                )
            elif "charbonnier" in out_criterion and out_criterion.get("charbonnier") is not None:
                logger_train.info(
                    f"Train epoch {epoch + 1}: ["
                    f"{i*len(d):5d}/{len(train_dataloader.dataset)}"
                    f" ({100. * i / len(train_dataloader):.0f}%)] "
                    f'Loss: {out_criterion["loss"].item():.4f} | '
                    f'Charbonnier loss: {out_criterion["charbonnier"].item():.4f} | '
                    f'LPIPS loss: {out_criterion["lpips"].item():.4f} | '
                    f'Style loss: {out_criterion["style_loss"].item():.4f} | '
                    f'Bpp loss: {out_criterion["bpp_loss"].item():.2f} | '
                    f"Aux loss: {aux_loss.item():.2f}"
                )
            elif out_criterion.get("ms_ssim_loss") is None:
                mse_val = out_criterion.get("mse_loss")
                mse_str = f'{mse_val.item():.4f}' if mse_val is not None else 'N/A'
                logger_train.info(
                    f"Train epoch {epoch + 1}: ["
                    f"{i*len(d):5d}/{len(train_dataloader.dataset)}"
                    f" ({100. * i / len(train_dataloader):.0f}%)] "
                    f'Loss: {out_criterion["loss"].item():.4f} | '
                    f'MSE loss: {mse_str} | '
                    f'Bpp loss: {out_criterion["bpp_loss"].item():.2f} | '
                    f"Aux loss: {aux_loss.item():.2f}"
                )
            else:
                logger_train.info(
                    f"Train epoch {epoch + 1}: ["
                    f"{i*len(d):5d}/{len(train_dataloader.dataset)}"
                    f" ({100. * i / len(train_dataloader):.0f}%)] "
                    f'Loss: {out_criterion["loss"].item():.4f} | '
                    f'MS-SSIM loss: {out_criterion["ms_ssim_loss"].item():.4f} | '
                    f'Bpp loss: {out_criterion["bpp_loss"].item():.2f} | '
                    f"Aux loss: {aux_loss.item():.2f}"
                )

    return current_step

def train_one_epoch_gan(
    model, model_disc, criterion, train_dataloader, optimizer, aux_optimizer, optimizer_D, epoch, clip_max_norm, logger_train, tb_logger, current_step, config=None
):
    """Enhanced Phase 2 GAN training with stability improvements.
    
    Includes:
    - Multi-Scale Discriminator support
    - R1 Gradient Penalty for GAN stability
    - TTUR (Two-Timescale Update Rule) via different learning rates
    - TOPIQ-FR perceptual loss (pyiqa)
    """
    model.train()
    device = next(model.parameters()).device
    gan_loss = GANLoss('hinge', loss_weight=2.0, real_label_val=1.0, fake_label_val=0.0)
    
    # Get R1 penalty settings from config
    use_r1 = config.get("use_r1_penalty", True) if config else True
    r1_weight = config.get("r1_weight", 10.0) if config else 10.0
    
    for i, d in enumerate(train_dataloader):
        d = d.to(device)
        
        # Enable gradients for R1 penalty computation
        if use_r1:
            d.requires_grad_(True)

        optimizer_D.zero_grad()
        
        # 1. Forward Generator
        out_net = model(d)
        
        # 2. Train Discriminator with R1 penalty
        pred_fake = model_disc(out_net["x_hat"].detach())
        pred_real = model_disc(d)
        
        # Compute discriminator loss with optional R1 penalty
        loss_D_total = compute_discriminator_loss(
            pred_real, pred_fake, gan_loss,
            real_img=d if use_r1 else None,
            r1_weight=r1_weight,
            use_r1=use_r1
        )
        
        loss_D_total.backward()
        optimizer_D.step()
        
        # Disable gradients on input for generator training
        if use_r1:
            d.requires_grad_(False)

        # 3. Train Generator
        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        pred_fake = model_disc(out_net["x_hat"])
        loss_G_fake = compute_generator_loss(pred_fake, gan_loss)

        out_criterion = criterion(out_net, d)
        
        # Phase 2 Enhanced Loss: MSE + LPIPS + Style + GAN + BPP + TOPIQ-FR
        if config is not None:
            # Get TOPIQ loss from criterion
            topiq_val = out_criterion.get("topiq", 0)
            
            # Convert to tensor if needed
            if not isinstance(topiq_val, torch.Tensor):
                topiq_val = torch.tensor(0.0, device=device)
            
            loss_G_total = (
                config.get("lambda_rd", 0.01) * out_criterion["rd_loss"] + 
                config.get("lambda_lpips", 1.0) * out_criterion["lpips"] + 
                config.get("lambda_style", 100.0) * out_criterion["style_loss"] + 
                config.get("lambda_gan", 1.0) * loss_G_fake + 
                config.get("lambda_rate", 0.3) * out_criterion["bpp_loss"] +
                config.get("lambda_topiq", 0.5) * topiq_val
            )
        else:
            # Default values (matching successful DISTS experiment)
            topiq_val = out_criterion.get("topiq", 0)
            if not isinstance(topiq_val, torch.Tensor):
                topiq_val = torch.tensor(0.0, device=device)
            loss_G_total = (
                0.01 * out_criterion["rd_loss"] + 
                1.0 * out_criterion["lpips"] + 
                100.0 * out_criterion["style_loss"] + 
                1.0 * loss_G_fake + 
                0.3 * out_criterion["bpp_loss"] +
                0.5 * topiq_val
            )
        
        loss_G_total.backward()

        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        current_step += 1

        # TensorBoard logging
        if current_step % 100 == 0:
            tb_logger.add_scalar('[train]: loss', loss_G_total.item(), current_step)
            tb_logger.add_scalar('[train]: bpp_loss', out_criterion["bpp_loss"].item(), current_step)
            tb_logger.add_scalar('[train]: lr', optimizer.param_groups[0]['lr'], current_step)
            tb_logger.add_scalar('[train]: aux_loss', aux_loss.item(), current_step)
            tb_logger.add_scalar('[train]: D_loss', loss_D_total.item(), current_step)
            
            if out_criterion.get("rd_loss") is not None:
                tb_logger.add_scalar('[train]: rd_loss', out_criterion["rd_loss"].item(), current_step)
            if isinstance(out_criterion.get("topiq"), torch.Tensor):
                tb_logger.add_scalar('[train]: topiq_loss', out_criterion["topiq"].item(), current_step)
          
        # Console logging
        if i % 100 == 0:
            rd_str = f'{out_criterion["rd_loss"].item():.4f}'
            
            # Get TOPIQ loss value
            topiq_val = out_criterion.get("topiq", 0)
            topiq_str = f'{topiq_val.item():.4f}' if isinstance(topiq_val, torch.Tensor) else '0.0000'
            
            # Handle multi-scale discriminator output for logging
            if isinstance(loss_G_fake, torch.Tensor):
                gan_str = f'{loss_G_fake.item():.4f}'
            else:
                gan_str = f'{loss_G_fake:.4f}'
            
            logger_train.info(
                f"Train epoch {epoch + 1}: ["
                f"{i*len(d):5d}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)] "
                f'Loss: {loss_G_total.item():.4f} | '
                f'RD: {rd_str} | '
                f'LPIPS: {out_criterion["lpips"].item():.4f} | '
                f'TOPIQ: {topiq_str} | '
                f'Style: {out_criterion["style_loss"].item():.4f} | '
                f'GAN: {gan_str} | '
                f'D: {loss_D_total.item():.4f} | '
                f'BPP: {out_criterion["bpp_loss"].item():.2f} | '
                f"Aux: {aux_loss.item():.2f}"
            )

    return current_step

def train_one_epoch_gan_face(
    model, model_disc, criterion, train_dataloader, optimizer, aux_optimizer, optimizer_D, epoch, clip_max_norm, logger_train, tb_logger, current_step, config
):
    model.train()
    device = next(model.parameters()).device
    gan_loss = GANLoss('hinge', loss_weight=2.0, real_label_val=1.0, fake_label_val=0.0)
    for i, d in enumerate(train_dataloader):
        d = d.to(device)
        mask = d[:, 3:, :, :]   # /255.0     mask_roi
        img = d[:, :3, :, :]    # /255.0
        optimizer_D.zero_grad()
        # 1.forward G
        out_net = model(img)
        out_net["x_tidle"] = mask * img + (1-mask) * out_net["x_hat"]
        # 2.backward netDimg
        pred_fake = model_disc(out_net["x_tidle"].detach())
        pred_real = model_disc(img)

        loss_D_real = gan_loss(pred_real, True, is_disc=True)
        loss_D_fake = gan_loss(pred_fake, False,is_disc=True)
        loss_D_total = (loss_D_real + loss_D_fake) * 0.5
        
        loss_D_total.backward()
        optimizer_D.step()

        # 3.backward netG
        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        pred_fake = model_disc(out_net["x_tidle"])
        loss_G_fake = gan_loss(pred_fake, False, is_disc=False)

        out_criterion = criterion(out_net, img, mask)
        loss_G_total = (config["lambda_char"]* out_criterion["charbonnier"] + config["lambda_lpips"] * out_criterion["lpips"] + config["lambda_style"] * out_criterion["style_loss"] + config["lambda_gan"] * loss_G_fake + config["lambda_rate"] * out_criterion["bpp_loss"] + config["lambda_face"] * out_criterion["face_loss"])
        
        loss_G_total.backward(torch.ones_like(loss_G_total))
        out_criterion["loss"] =  torch.mean(loss_G_total)
        out_criterion["lpips"] = torch.mean(out_criterion["lpips"])
        out_criterion["face_loss"] = torch.mean(out_criterion["face_loss"])
        
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        current_step += 1

        if current_step % 100 == 0:
            tb_logger.add_scalar('{}'.format('[train]: loss'), out_criterion["loss"].item(), current_step)
            tb_logger.add_scalar('{}'.format('[train]: bpp_loss'), out_criterion["bpp_loss"].item(), current_step)
            tb_logger.add_scalar('{}'.format('[train]: lr'), optimizer.param_groups[0]['lr'], current_step)
            tb_logger.add_scalar('{}'.format('[train]: aux_loss'), aux_loss.item(), current_step)
          
        # print(out_criterion["loss"].size(),out_criterion["charbonnier"].size(),out_criterion["lpips"].size(),out_criterion["style_loss"].size())
        if i % 100 == 0:
                logger_train.info(
                    f"Train epoch {epoch + 1}: ["
                    f"{i*len(d):5d}/{len(train_dataloader.dataset)}"
                    f" ({100. * i / len(train_dataloader):.0f}%)] "
                    f'Loss: {out_criterion["loss"].item():.4f} | '
                    f'Charbonnier loss: {out_criterion["charbonnier"].item():.4f} | '

                    f'Lpips loss: {out_criterion["lpips"].item():.4f} | '
                    f'Style loss : {out_criterion["style_loss"].item():.4f} | '
                    f'Face loss : {out_criterion["face_loss"].item():.6f} | '
                    f'Adv D loss: {loss_D_total.item():.4f} | '
                    f'Adv G loss: {loss_G_fake.item():.4f} | '

                    f'Bpp loss: {out_criterion["bpp_loss"].item():.2f} | '
                    f"Aux loss: {aux_loss.item():.2f}"
                )
            

    return current_step
