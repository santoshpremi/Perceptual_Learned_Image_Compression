import torch
import torch.nn as nn
import torch.nn.functional as F
#from utils.dist import *
from loss.rd_loss import GANLoss


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
        # Phase 1: Original HFLIC - Charbonnier + LPIPS + Style + bpp rate
        # Total loss = lambda_char * Charbonnier + lambda_lpips * LPIPS + lambda_style * Style + lambda_bpp_rate * bpp rate
        if config is not None:
            # Check if using Charbonnier (original HFLIC) or RD loss (for backward compatibility)
            if "charbonnier" in out_criterion and out_criterion.get("charbonnier") is not None:
                # Original HFLIC: Use Charbonnier loss
                total_loss = (config.get("lambda_char", 2e-6) * out_criterion["charbonnier"] + 
                         config["lambda_lpips"] * out_criterion["lpips"] + 
                         config["lambda_style"] * out_criterion["style_loss"] + 
                         config["lambda_bpp_rate"] * out_criterion["bpp_loss"])
            elif "rd_loss" in out_criterion and out_criterion["rd_loss"] is not None:
                # Fallback to RD loss if Charbonnier not available (backward compatibility)
                total_loss = (config.get("lambda_rd", 1e-2) * out_criterion["rd_loss"] + 
                             config["lambda_lpips"] * out_criterion["lpips"] + 
                             config["lambda_style"] * out_criterion["style_loss"] + 
                             config["lambda_bpp_rate"] * out_criterion["bpp_loss"])
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
                    f'Bpp rate loss: {out_criterion["bpp_loss"].item():.2f} | '
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
                    f'Bpp rate loss: {out_criterion["bpp_loss"].item():.2f} | '
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
                    f'Bpp rate loss: {out_criterion["bpp_loss"].item():.2f} | '
                    f"Aux loss: {aux_loss.item():.2f}"
                )
            else:
                logger_train.info(
                    f"Train epoch {epoch + 1}: ["
                    f"{i*len(d):5d}/{len(train_dataloader.dataset)}"
                    f" ({100. * i / len(train_dataloader):.0f}%)] "
                    f'Loss: {out_criterion["loss"].item():.4f} | '
                    f'MS-SSIM loss: {out_criterion["ms_ssim_loss"].item():.4f} | '
                    f'Bpp rate loss: {out_criterion["bpp_loss"].item():.2f} | '
                    f"Aux loss: {aux_loss.item():.2f}"
                )

    return current_step

def train_one_epoch_gan(
    model, model_disc, criterion, train_dataloader, optimizer, aux_optimizer, optimizer_D, epoch, clip_max_norm, logger_train, tb_logger, current_step, config=None
):
    model.train()
    device = next(model.parameters()).device
    gan_loss = GANLoss('hinge', loss_weight=2.0, real_label_val=1.0, fake_label_val=0.0)
    for i, d in enumerate(train_dataloader):
        d = d.to(device)

        optimizer_D.zero_grad()
        
        # 1.forward G
        out_net = model(d)
        # 2.backward netD
        pred_fake = model_disc(out_net["x_hat"].detach())
        pred_real = model_disc(d)

        loss_D_real = gan_loss(pred_real, True, is_disc=True)
        loss_D_fake = gan_loss(pred_fake, False,is_disc=True)
        loss_D_total = (loss_D_real + loss_D_fake) * 0.5
        
        loss_D_total.backward()
        optimizer_D.step()

        # 3.backward netG
        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        pred_fake = model_disc(out_net["x_hat"])
        loss_G_fake = gan_loss(pred_fake, False, is_disc=False)

        out_criterion = criterion(out_net, d)
        # Phase 2: RD + LPIPS + DISTS + (1-VSI) + Style + GAN + bpp
        # VSI is a FR quality score (higher=better), so use (1-VSI) as loss
        vsi = out_criterion.get("vsi")
        dists = out_criterion.get("dists")
        
        # VSI term: only if VSI is available and non-zero
        if config is not None and vsi is not None and isinstance(vsi, torch.Tensor) and vsi.item() != 0:
            vsi_term = config.get("lambda_vsi", 0.5) * (1.0 - vsi)
        else:
            vsi_term = torch.tensor(0.0, device=out_criterion["rd_loss"].device)
        
        # DISTS term: only if DISTS is available and non-zero
        if config is not None and dists is not None and isinstance(dists, torch.Tensor) and dists.item() != 0:
            dists_term = config.get("lambda_dists", 1.0) * dists
        else:
            dists_term = torch.tensor(0.0, device=out_criterion["rd_loss"].device)
        
        if config is not None:
            loss_G_total = (config.get("lambda_rd", 0.01) * out_criterion["rd_loss"] +
                          config.get("lambda_lpips", 2.0) * out_criterion["lpips"] +
                          dists_term +
                          vsi_term +
                          config["lambda_style"] * out_criterion["style_loss"] +
                          config["lambda_gan"] * loss_G_fake +
                          config["lambda_bpp_rate"] * out_criterion["bpp_loss"])
        else:
            loss_G_total = (out_criterion["rd_loss"] +
                          out_criterion["lpips"] +
                          dists_term +
                          vsi_term +
                          out_criterion["style_loss"] +
                          loss_G_fake +
                          out_criterion["bpp_loss"])
        loss_G_total.backward()

        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        current_step += 1

        if current_step % 100 == 0:
            tb_logger.add_scalar('[train]: loss', loss_G_total.item(), current_step)
            tb_logger.add_scalar('[train]: bpp_loss', out_criterion["bpp_loss"].item(), current_step)
            tb_logger.add_scalar('[train]: lr', optimizer.param_groups[0]['lr'], current_step)
            tb_logger.add_scalar('[train]: aux_loss', aux_loss.item(), current_step)
            if out_criterion.get("rd_loss") is not None:
                tb_logger.add_scalar('[train]: rd_loss', out_criterion["rd_loss"].item(), current_step)
            if out_criterion.get("vsi") is not None and isinstance(out_criterion["vsi"], torch.Tensor) and out_criterion["vsi"].item() != 0:
                tb_logger.add_scalar('[train]: vsi', out_criterion["vsi"].item(), current_step)
            if out_criterion.get("lpips") is not None and isinstance(out_criterion["lpips"], torch.Tensor):
                tb_logger.add_scalar('[train]: lpips', out_criterion["lpips"].item(), current_step)
            if out_criterion.get("dists") is not None and isinstance(out_criterion["dists"], torch.Tensor) and out_criterion["dists"].item() != 0:
                tb_logger.add_scalar('[train]: dists', out_criterion["dists"].item(), current_step)
          
        if i % 100 == 0:
                # Phase 2: RD + LPIPS + DISTS + (1-VSI) + Style + GAN + bpp
                rd_str = f'{out_criterion["rd_loss"].item():.4f}'
                vsi_val = out_criterion.get("vsi")
                vsi_str = f'{vsi_val.item():.4f}' if isinstance(vsi_val, torch.Tensor) and vsi_val.item() != 0 else 'N/A'
                lpips_val = out_criterion.get("lpips")
                lpips_str = f'{lpips_val.item():.4f}' if isinstance(lpips_val, torch.Tensor) else 'N/A'
                dists_val = out_criterion.get("dists")
                dists_str = f'{dists_val.item():.4f}' if isinstance(dists_val, torch.Tensor) and dists_val.item() != 0 else 'N/A'

                logger_train.info(
                    f"Train epoch {epoch + 1}: ["
                    f"{i*len(d):5d}/{len(train_dataloader.dataset)}"
                    f" ({100. * i / len(train_dataloader):.0f}%)] "
                    f'Loss: {loss_G_total.item():.4f} | '
                    f'RD loss: {rd_str} | '
                    f'LPIPS: {lpips_str} | '
                    f'DISTS: {dists_str} | '
                    f'Style: {out_criterion["style_loss"].item():.4f} | '
                    f'GAN: {loss_G_fake.item():.4f} | '
                    f'Bpp: {out_criterion["bpp_loss"].item():.2f} | '
                    f"Aux: {aux_loss.item():.2f}"
                )
            

    return current_step

def train_one_epoch_gan_baseline(
    model, model_disc, criterion, train_dataloader, optimizer, aux_optimizer, optimizer_D, epoch, clip_max_norm, logger_train, tb_logger, current_step, config=None
):
    """BASELINE Phase 2 training: Original HFLIC loss - Charbonnier + LPIPS + Style + GAN + BPP.
    
    This is the ORIGINAL HFLIC Phase 2 for fair comparison.
    HFLIC hardcoded weights: 3e-4 × Char + 2.0 × LPIPS + 1.0 × Style + 1.0 × GAN + 1.0 × BPP
    """
    model.train()
    device = next(model.parameters()).device
    gan_loss = GANLoss('hinge', loss_weight=2.0, real_label_val=1.0, fake_label_val=0.0)
    
    for i, d in enumerate(train_dataloader):
        d = d.to(device)

        optimizer_D.zero_grad()
        
        out_net = model(d)
        pred_fake = model_disc(out_net["x_hat"].detach())
        pred_real = model_disc(d)

        loss_D_real = gan_loss(pred_real, True, is_disc=True)
        loss_D_fake = gan_loss(pred_fake, False, is_disc=True)
        loss_D_total = (loss_D_real + loss_D_fake) * 0.5
        
        loss_D_total.backward()
        optimizer_D.step()

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        pred_fake = model_disc(out_net["x_hat"])
        loss_G_fake = gan_loss(pred_fake, False, is_disc=False)

        out_criterion = criterion(out_net, d)
        
        # HFLIC original Phase 2 weights (hardcoded as in paper)
        # Only lambda_bpp_rate is configurable for different operating points
        lambda_char = 3e-4      # HFLIC hardcoded
        lambda_lpips = 2.0      # HFLIC hardcoded
        lambda_style = 1.0      # HFLIC hardcoded
        lambda_gan = 1.0        # HFLIC hardcoded
        lambda_bpp = config.get("lambda_bpp_rate", 1.0) if config is not None else 1.0
        
        loss_G_total = (lambda_char * out_criterion["charbonnier"] +
                       lambda_lpips * out_criterion["lpips"] +
                       lambda_style * out_criterion["style_loss"] +
                       lambda_gan * loss_G_fake +
                       lambda_bpp * out_criterion["bpp_loss"])
        
        loss_G_total.backward()

        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        current_step += 1

        if current_step % 100 == 0:
            tb_logger.add_scalar('[train]: loss', loss_G_total.item(), current_step)
            tb_logger.add_scalar('[train]: bpp_loss', out_criterion["bpp_loss"].item(), current_step)
            tb_logger.add_scalar('[train]: charbonnier', out_criterion["charbonnier"].item(), current_step)
            tb_logger.add_scalar('[train]: lpips', out_criterion["lpips"].item(), current_step)
            tb_logger.add_scalar('[train]: lr', optimizer.param_groups[0]['lr'], current_step)
          
        if i % 100 == 0:
            logger_train.info(
                f"Train epoch {epoch + 1}: ["
                f"{i*len(d):5d}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)] "
                f'Loss: {loss_G_total.item():.4f} | '
                f'Char: {out_criterion["charbonnier"].item():.4f} | '
                f'LPIPS: {out_criterion["lpips"].item():.4f} | '
                f'Style: {out_criterion["style_loss"].item():.4f} | '
                f'GAN: {loss_G_fake.item():.4f} | '
                f'Bpp: {out_criterion["bpp_loss"].item():.2f} | '
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
        loss_G_total = (config["lambda_char"]* out_criterion["charbonnier"] + config.get("lambda_dists", 1.0) * out_criterion["dists"] + config["lambda_style"] * out_criterion["style_loss"] + config["lambda_gan"] * loss_G_fake + config["lambda_bpp_rate"] * out_criterion["bpp_loss"] + config["lambda_face"] * out_criterion["face_loss"])
        
        loss_G_total.backward(torch.ones_like(loss_G_total))
        out_criterion["loss"] =  torch.mean(loss_G_total)
        out_criterion["dists"] = torch.mean(out_criterion["dists"])
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
          
        # print(out_criterion["loss"].size(),out_criterion["charbonnier"].size(),out_criterion["dists"].size(),out_criterion["style_loss"].size())
        if i % 100 == 0:
                dists_val = out_criterion.get("dists", 0)
                dists_str = f'{dists_val.item():.4f}' if isinstance(dists_val, torch.Tensor) else '0.0000'
                logger_train.info(
                    f"Train epoch {epoch + 1}: ["
                    f"{i*len(d):5d}/{len(train_dataloader.dataset)}"
                    f" ({100. * i / len(train_dataloader):.0f}%)] "
                    f'Loss: {out_criterion["loss"].item():.4f} | '
                    f'Charbonnier loss: {out_criterion["charbonnier"].item():.4f} | '

                    f'DISTS loss: {dists_str} | '
                    f'Style loss : {out_criterion["style_loss"].item():.4f} | '
                    f'Face loss : {out_criterion["face_loss"].item():.6f} | '
                    f'Adv D loss: {loss_D_total.item():.4f} | '
                    f'Adv G loss: {loss_G_fake.item():.4f} | '

                    f'Bpp rate loss: {out_criterion["bpp_loss"].item():.2f} | '
                    f"Aux loss: {aux_loss.item():.2f}"
                )
            

    return current_step
