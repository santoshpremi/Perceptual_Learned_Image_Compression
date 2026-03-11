import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.func import image2patch
from utils.metrics import compute_metrics
from utils.utils import *
from utils.func import image2patch, patch2image
from loss.rd_loss import GANLoss
from loss import perceptual_loss as ps


def test_one_epoch(epoch, test_dataloader, model, criterion, save_dir, logger_val, tb_logger, config=None):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    ms_ssim_loss = AverageMeter()
    charbonnier = AverageMeter()
    lpips = AverageMeter()
    style_loss = AverageMeter()
    aux_loss = AverageMeter()
    psnr = AverageMeter()
    ms_ssim = AverageMeter()

    with torch.no_grad():
        for i, d in enumerate(test_dataloader):
            d = d.to(device)
            
            # Handle padding for images not divisible by 64
            B, C, H, W = d.shape
            pad_h = 0
            pad_w = 0
            if H % 64 != 0:
                pad_h = 64 * (H // 64 + 1) - H
            if W % 64 != 0:
                pad_w = 64 * (W // 64 + 1) - W
            
            # Pad image if needed
            if pad_h > 0 or pad_w > 0:
                d_pad = F.pad(d, (0, pad_w, 0, pad_h), mode='constant', value=0)
            else:
                d_pad = d
            
            # Forward pass with padded image
            out_net = model(d_pad)
            
            # Crop output back to original size before computing loss
            if pad_h > 0 or pad_w > 0:
                out_net['x_hat'] = out_net['x_hat'][:, :, :H, :W]
            
            # Now compute loss with correctly sized tensors
            out_criterion = criterion(out_net, d)

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            # Phase 1: compute total loss if not already computed, using config lambda weights
            if "loss" not in out_criterion or out_criterion.get("loss") is None:
                # Check if using Charbonnier (original HFLIC) or RD loss (for backward compatibility)
                if "charbonnier" in out_criterion and out_criterion.get("charbonnier") is not None:
                    # Original HFLIC: Use Charbonnier loss
                    if config is not None:
                        # Use config lambda values for proper loss weighting (consistent with training)
                        out_criterion["loss"] = (config.get("lambda_char", 2e-6) * out_criterion["charbonnier"] + 
                                               config["lambda_lpips"] * out_criterion["lpips"] + 
                                               config["lambda_style"] * out_criterion["style_loss"] + 
                                               config["lambda_bpp_rate"] * out_criterion["bpp_loss"])
                    else:
                        # Fallback to unweighted sum if config not provided
                        out_criterion["loss"] = (out_criterion["charbonnier"] + 
                                               out_criterion["lpips"] + 
                                               out_criterion["style_loss"] + 
                                               out_criterion["bpp_loss"])
                elif "rd_loss" in out_criterion and out_criterion["rd_loss"] is not None:
                    # Fallback to RD loss if Charbonnier not available (backward compatibility)
                    if config is not None:
                        # Use config lambda values for proper loss weighting (consistent with training)
                        out_criterion["loss"] = (config.get("lambda_rd", 1e-2) * out_criterion["rd_loss"] + 
                                               config["lambda_lpips"] * out_criterion["lpips"] + 
                                               config["lambda_style"] * out_criterion["style_loss"] + 
                                               config["lambda_bpp_rate"] * out_criterion["bpp_loss"])
                    else:
                        # Fallback to unweighted sum if config not provided
                        out_criterion["loss"] = (out_criterion["rd_loss"] + 
                                               out_criterion["lpips"] + 
                                               out_criterion["style_loss"] + 
                                               out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            if out_criterion.get("mse_loss") is not None:
                mse_loss.update(out_criterion["mse_loss"])
            if out_criterion.get("ms_ssim_loss") is not None:
                ms_ssim_loss.update(out_criterion["ms_ssim_loss"])
            # Log perceptual losses for Phase 1 training
            if "rd_loss" in out_criterion and out_criterion["rd_loss"] is not None:
                # RD loss case - no charbonnier to log
                lpips.update(out_criterion["lpips"].item())
                style_loss.update(out_criterion["style_loss"].item())
            elif "charbonnier" in out_criterion and out_criterion.get("charbonnier") is not None:
                charbonnier.update(out_criterion["charbonnier"].item())
                lpips.update(out_criterion["lpips"].item())
                style_loss.update(out_criterion["style_loss"].item())

            rec = torch2img(out_net['x_hat'])
            img = torch2img(d)
            p, m = compute_metrics(rec, img)
            psnr.update(p)
            ms_ssim.update(m)

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            rec.save(os.path.join(save_dir, '%03d_rec.png' % i))
            img.save(os.path.join(save_dir, '%03d_gt.png' % i))

    tb_logger.add_scalar('{}'.format('[val]: loss'), loss.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format('[val]: bpp_loss'), bpp_loss.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format('[val]: psnr'), psnr.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format('[val]: ms-ssim'), ms_ssim.avg, epoch + 1)
    
    # Log RD loss if available
    if "rd_loss" in out_criterion and out_criterion.get("rd_loss") is not None:
        # We need to track this during the loop, but for now just log if available
        pass
    
    # Log perceptual losses for Phase 1 if available
    if lpips.count > 0:
        tb_logger.add_scalar('{}'.format('[val]: lpips'), lpips.avg, epoch + 1)
        tb_logger.add_scalar('{}'.format('[val]: style loss'), style_loss.avg, epoch + 1)
        if charbonnier.count > 0:
            tb_logger.add_scalar('{}'.format('[val]: charbonnier loss'), charbonnier.avg, epoch + 1)
            logger_val.info(
                f"Test epoch {epoch + 1}: Average losses: "
                f"Loss: {loss.avg:.4f} | "
                f"Charbonnier loss: {charbonnier.avg:.4f} | "
                f"LPIPS loss: {lpips.avg:.4f} | "
                f"Style loss: {style_loss.avg:.4f} | "
                f"Bpp rate loss: {bpp_loss.avg:.4f} | "
                f"Aux loss: {aux_loss.avg:.2f} | "
                f"PSNR: {psnr.avg:.6f} dB | "
                f"MS-SSIM: {ms_ssim.avg:.6f} dB"
            )
        else:
            # RD loss case
            logger_val.info(
                f"Test epoch {epoch + 1}: Average losses: "
                f"Loss: {loss.avg:.4f} | "
                f"LPIPS loss: {lpips.avg:.4f} | "
                f"Style loss: {style_loss.avg:.4f} | "
                f"Bpp rate loss: {bpp_loss.avg:.4f} | "
                f"Aux loss: {aux_loss.avg:.2f} | "
                f"PSNR: {psnr.avg:.6f} dB | "
                f"MS-SSIM: {ms_ssim.avg:.6f} dB"
            )
    elif mse_loss.count > 0:
        logger_val.info(
            f"Test epoch {epoch + 1}: Average losses: "
            f"Loss: {loss.avg:.4f} | "
            f"MSE loss: {mse_loss.avg:.4f} | "
            f"Bpp rate loss: {bpp_loss.avg:.4f} | "
            f"Aux loss: {aux_loss.avg:.2f} | "
            f"PSNR: {psnr.avg:.6f} dB | "
            f"MS-SSIM: {ms_ssim.avg:.6f} dB"
        )
        tb_logger.add_scalar('{}'.format('[val]: mse_loss'), mse_loss.avg, epoch + 1)
    elif ms_ssim_loss.count > 0:
        logger_val.info(
            f"Test epoch {epoch + 1}: Average losses: "
            f"Loss: {loss.avg:.4f} | "
            f"MS-SSIM loss: {ms_ssim_loss.avg:.4f} | "
            f"Bpp rate loss: {bpp_loss.avg:.4f} | "
            f"Aux loss: {aux_loss.avg:.2f} | "
            f"PSNR: {psnr.avg:.6f} dB | "
            f"MS-SSIM: {ms_ssim.avg:.6f} dB"
        )
        tb_logger.add_scalar('{}'.format('[val]: ms_ssim_loss'), ms_ssim_loss.avg, epoch + 1)

    return loss.avg


def compress_one_image(model, x, stream_path, H, W, img_name):
    with torch.no_grad():
        out = model.compress(x)

    shape = out["shape"]
    output = os.path.join(stream_path, img_name)
    with Path(output).open("wb") as f:
        write_uints(f, (H, W))
        write_body(f, shape, out["strings"])

    size = filesize(output)
    bpp = float(size) * 8 / (H * W)
    output = os.path.join(stream_path, img_name+"_%04f"%(bpp))
    print(output)
    with Path(output).open("wb") as f:
        write_uints(f, (H, W))
        write_body(f, shape, out["strings"])
    cost_time = out["cost_time"]
    entropy_time = out["entropy_time"]
    encoder_time = out["encoder_time"]
    return bpp, cost_time, entropy_time, encoder_time


def decompress_one_image(model, stream_path, img_name):
    output = os.path.join(stream_path, img_name)
    with Path(output).open("rb") as f:
        original_size = read_uints(f, 2)
        strings, shape = read_body(f)

    with torch.no_grad():
        out = model.decompress(strings, shape)

    x_hat = out["x_hat"]
    x_hat = x_hat[:, :, 0 : original_size[0], 0 : original_size[1]]
    cost_time = out["cost_time"]
    entropy_time = out["entropy_time"]
    decoder_time = out["decoder_time"]
    return x_hat, cost_time, entropy_time, decoder_time


def test_model(test_dataloader, net, logger_test, save_dir, epoch, gpu_id):
    net.eval()
    device = next(net.parameters()).device
    avg_psnr = AverageMeter()
    avg_ms_ssim = AverageMeter()
    avg_lpips = AverageMeter()
    avg_bpp = AverageMeter()

    avg_enc_dec_time = AverageMeter()
    avg_enc_time = AverageMeter()
    avg_dec_time = AverageMeter()
    avg_enc_entropy_time = AverageMeter()
    avg_dec_entropy_time = AverageMeter()
    avg_encoder_time = AverageMeter()
    avg_decoder_time = AverageMeter()
    
    # Initialize LPIPS model for evaluation
    lpips_model = ps.PerceptualLoss(model='net-lin', net='vgg',
                                   use_gpu=torch.cuda.is_available(),
                                   gpu_ids=[gpu_id] if isinstance(gpu_id, int) else gpu_id)
    lpips_model.eval()
    
    with torch.no_grad():
        for i, img in enumerate(test_dataloader):
            img = img.to(device)
            B, C, H, W = img.shape
            pad_h = 0
            pad_w = 0
            if H % 64 != 0:
                pad_h = 64 * (H // 64 + 1) - H
            if W % 64 != 0:
                pad_w = 64 * (W // 64 + 1) - W

            img_pad = F.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0)
            bpp, enc_time, enc_entropy_time, encoder_time = compress_one_image(model=net, x=img_pad, stream_path=save_dir, H=H, W=W, img_name=str(i))
            x_hat, dec_time, dec_entropy_time, decoder_time = decompress_one_image(model=net, stream_path=save_dir, img_name=str(i))
            
            # Crop x_hat to original size (remove padding)
            x_hat_cropped = x_hat[:, :, :H, :W]
            
            # Convert to PIL for PSNR/MS-SSIM computation
            rec = torch2img(x_hat_cropped)
            img_pil = torch2img(img)
            #img_pil.save(os.path.join(save_dir, '%03d_gt.png' % i))
            rec.save(os.path.join(save_dir, '%03d_rec.png' % i))
            
            # Compute PSNR and MS-SSIM
            p, m = compute_metrics(rec, img_pil)
            avg_psnr.update(p)
            avg_ms_ssim.update(m)
            
            # Compute LPIPS (expects tensors in [0,1] range, shape [B,C,H,W])
            lpips_val = lpips_model(x_hat_cropped, img).mean().item()
            avg_lpips.update(lpips_val)
            
            avg_bpp.update(bpp)
            avg_enc_time.update(enc_time)
            avg_dec_time.update(dec_time)
            avg_enc_dec_time.update(enc_time + dec_time)
            avg_enc_entropy_time.update(enc_entropy_time)
            avg_dec_entropy_time.update(dec_entropy_time)
            avg_encoder_time.update(encoder_time)
            avg_decoder_time.update(decoder_time)
            logger_test.info(
                f"Image[{i}] | "
                f"Bpp rate loss: {bpp:.4f} | "
                f"PSNR: {p:.4f} dB | "
                f"MS-SSIM: {m:.4f} dB | "
                f"LPIPS: {lpips_val:.6f} "
                f"Time: {enc_time+dec_time:.4f} | "
                f"Enc Time: {enc_time:.4f} | "
                f"Entropy Enc Time: {enc_entropy_time:.4f} | "
                f"Dec Time: {dec_time:.4f} | "
                f"Entropy dec Time: {dec_entropy_time:.4f} | "
            )
    logger_test.info(
        f"Epoch:[{epoch + 1}] | "
        f"Avg Bpp: {avg_bpp.avg:.4f} | "
        f"Avg PSNR: {avg_psnr.avg:.4f} dB | "
        f"Avg MS-SSIM: {avg_ms_ssim.avg:.4f} dB | "
        f"Avg LPIPS: {avg_lpips.avg:.6f} "
        f"Avg Time: {avg_enc_dec_time.avg:.4f} | "
        f"Avg Enc Time: {avg_enc_time.avg:.4f} | "
        f"Avg Dec Time: {avg_dec_time.avg:.4f} | "
        f"Avg Enc Entropy Time: {avg_enc_entropy_time.avg:.4f} | "
        f"Avg Dec Entropy Time: {avg_dec_entropy_time.avg:.4f} | "
        f"Avg Encoder Time: {avg_encoder_time.avg:.4f} | "
        f"Avg Decoder Time: {avg_decoder_time.avg:.4f} | "            
   )


def test_one_epoch_gan(epoch, test_dataloader, model, model_disc,criterion, save_dir, logger_val, tb_logger, config=None):
    model.eval()
    device = next(model.parameters()).device
    gan_loss = GANLoss('hinge', loss_weight=2.0, real_label_val=1.0, fake_label_val=0.0)

    # LPIPS model: used as fallback if criterion does not return lpips (should not happen in Phase 2)
    lpips_model = ps.PerceptualLoss(model='net-lin', net='vgg',
                                   use_gpu=torch.cuda.is_available(),
                                   gpu_ids=[0])
    lpips_model.eval()

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    charbonnier = AverageMeter()
    rd_loss = AverageMeter()
    vsi = AverageMeter()
    lpips = AverageMeter()
    dists = AverageMeter()
    style_loss = AverageMeter()
    adv_loss = AverageMeter()
    aux_loss = AverageMeter()
    psnr = AverageMeter()
    ms_ssim = AverageMeter()

    with torch.no_grad():
        for i, d in enumerate(test_dataloader):
            d = d.to(device)
            
            # Handle padding for images not divisible by 64
            B, C, H, W = d.shape
            pad_h = 0
            pad_w = 0
            if H % 64 != 0:
                pad_h = 64 * (H // 64 + 1) - H
            if W % 64 != 0:
                pad_w = 64 * (W // 64 + 1) - W
            
            # Pad image if needed
            if pad_h > 0 or pad_w > 0:
                d_pad = F.pad(d, (0, pad_w, 0, pad_h), mode='constant', value=0)
            else:
                d_pad = d
            
            # Forward pass with padded image
            out_net = model(d_pad)
            
            # Crop output back to original size before computing loss
            if pad_h > 0 or pad_w > 0:
                out_net['x_hat'] = out_net['x_hat'][:, :, :H, :W]
            
            # Now compute loss with correctly sized tensors
            out_criterion = criterion(out_net, d)

            pred_fake = model_disc(out_net["x_hat"])
            loss_G_fake = gan_loss(pred_fake, False, is_disc=False)
            # Phase 2: RD + LPIPS + DISTS + (1-VSI) + Style + GAN + bpp (matches training objective)
            # VSI is a FR quality score (higher=better), so use (1-VSI) as loss
            vsi_val = out_criterion.get("vsi")
            dists_val = out_criterion.get("dists")
            
            # VSI term: only if VSI is available and non-zero
            if config is not None and vsi_val is not None and isinstance(vsi_val, torch.Tensor) and vsi_val.item() != 0:
                vsi_term = config.get("lambda_vsi", 0.5) * (1.0 - vsi_val)
            else:
                vsi_term = torch.tensor(0.0, device=out_criterion["rd_loss"].device)
            
            # DISTS term: only if DISTS is available and non-zero
            if config is not None and dists_val is not None and isinstance(dists_val, torch.Tensor) and dists_val.item() != 0:
                dists_term = config.get("lambda_dists", 1.0) * dists_val
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

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"].item())
            loss.update(loss_G_total.item())
            if vsi_val is not None and isinstance(vsi_val, torch.Tensor) and vsi_val.item() != 0:
                vsi.update(vsi_val.item())
            if dists_val is not None and isinstance(dists_val, torch.Tensor) and dists_val.item() != 0:
                dists.update(dists_val.item())
            style_loss.update(out_criterion["style_loss"].item())
            # LPIPS: included in validation aggregate loss
            if out_criterion.get("lpips") is not None and isinstance(out_criterion["lpips"], torch.Tensor):
                lpips.update(out_criterion["lpips"].item())
            else:
                lpips_val = lpips_model(out_net["x_hat"], d).mean().item()
                lpips.update(lpips_val)
            adv_loss.update(loss_G_fake.item())
            if out_criterion.get("rd_loss") is not None:
                rd_loss.update(out_criterion["rd_loss"].item())
            elif out_criterion.get("charbonnier") is not None:
                charbonnier.update(out_criterion["charbonnier"].item())

            rec = torch2img(out_net['x_hat'])
            img = torch2img(d)
            p, m = compute_metrics(rec, img)
            psnr.update(p)
            ms_ssim.update(m)

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            rec.save(os.path.join(save_dir, '%03d_rec.png' % i))
            img.save(os.path.join(save_dir, '%03d_gt.png' % i))

    tb_logger.add_scalar('{}'.format('[val]: loss'), loss.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format('[val]: bpp_loss'), bpp_loss.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format('[val]: psnr'), psnr.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format('[val]: ms-ssim'), ms_ssim.avg, epoch + 1)
    if vsi.count > 0:
        tb_logger.add_scalar('{}'.format('[val]: vsi'), vsi.avg, epoch + 1)
    if dists.count > 0:
        tb_logger.add_scalar('{}'.format('[val]: dists'), dists.avg, epoch + 1)
    if lpips.count > 0:
        tb_logger.add_scalar('{}'.format('[val]: lpips'), lpips.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format('[val]: style loss'), style_loss.avg, epoch + 1)
    if rd_loss.count > 0:
        tb_logger.add_scalar('{}'.format('[val]: rd_loss'), rd_loss.avg, epoch + 1)
    if charbonnier.count > 0:
        tb_logger.add_scalar('{}'.format('[val]: charbonnier loss'), charbonnier.avg, epoch + 1)

    rd_str = f"{rd_loss.avg:.4f}" if rd_loss.count > 0 else "N/A"
    charbonnier_str = f"{charbonnier.avg:.4f}" if charbonnier.count > 0 else "N/A"
    vsi_str = f"{vsi.avg:.4f}" if vsi.count > 0 else "N/A"
    lpips_str = f"{lpips.avg:.4f}" if lpips.count > 0 else "N/A"
    dists_str = f"{dists.avg:.4f}" if dists.count > 0 else "N/A"

    logger_val.info(
        f"Test epoch {epoch + 1}: Average losses: "
        f"Loss: {loss.avg:.4f} | "
        f"RD loss: {rd_str} | "
        f"Charbonnier loss: {charbonnier_str} | "
        f"LPIPS loss: {lpips_str} | "
        f"DISTS loss: {dists_str} | "
        f"VSI: {vsi_str} | "
        f"Style loss: {style_loss.avg:.4f} | "
        f"Adv loss: {adv_loss.avg:.4f} | "
        f"Bpp rate loss: {bpp_loss.avg:.4f} | "
        f"Aux loss: {aux_loss.avg:.2f} | "
        f"PSNR: {psnr.avg:.6f} dB | "
        f"MS-SSIM: {ms_ssim.avg:.6f} dB"
    )
    
    # Checkpoint: unscale rd_loss back to raw MSE so both terms are comparable (~0.003 + ~0.2)
    checkpoint_metric = rd_loss.avg / (255 ** 2) + lpips.avg if rd_loss.count > 0 else lpips.avg
    if dists.count > 0:
        checkpoint_metric += dists.avg
    return checkpoint_metric


def test_one_epoch_gan_baseline(epoch, test_dataloader, model, model_disc, criterion, save_dir, logger_val, tb_logger, config=None):
    """BASELINE Phase 2 testing: Original HFLIC loss - Charbonnier + LPIPS + Style + GAN + BPP."""
    model.eval()
    device = next(model.parameters()).device
    gan_loss = GANLoss('hinge', loss_weight=2.0, real_label_val=1.0, fake_label_val=0.0)

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    charbonnier = AverageMeter()
    lpips = AverageMeter()
    style_loss = AverageMeter()
    adv_loss = AverageMeter()
    aux_loss = AverageMeter()
    psnr = AverageMeter()
    ms_ssim = AverageMeter()

    with torch.no_grad():
        for i, d in enumerate(test_dataloader):
            d = d.to(device)
            
            B, C, H, W = d.shape
            pad_h = 0
            pad_w = 0
            if H % 64 != 0:
                pad_h = 64 * (H // 64 + 1) - H
            if W % 64 != 0:
                pad_w = 64 * (W // 64 + 1) - W
            
            if pad_h > 0 or pad_w > 0:
                d_pad = F.pad(d, (0, pad_w, 0, pad_h), mode='constant', value=0)
            else:
                d_pad = d
            
            out_net = model(d_pad)
            
            if pad_h > 0 or pad_w > 0:
                out_net['x_hat'] = out_net['x_hat'][:, :, :H, :W]
            
            out_criterion = criterion(out_net, d)

            pred_fake = model_disc(out_net["x_hat"])
            loss_G_fake = gan_loss(pred_fake, False, is_disc=False)
            
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

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"].item())
            loss.update(loss_G_total.item())
            charbonnier.update(out_criterion["charbonnier"].item())
            lpips.update(out_criterion["lpips"].item())
            style_loss.update(out_criterion["style_loss"].item())
            adv_loss.update(loss_G_fake.item())

            rec = torch2img(out_net['x_hat'])
            img = torch2img(d)
            p, m = compute_metrics(rec, img)
            psnr.update(p)
            ms_ssim.update(m)

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            rec.save(os.path.join(save_dir, '%03d_rec.png' % i))
            img.save(os.path.join(save_dir, '%03d_gt.png' % i))

    tb_logger.add_scalar('[val]: loss', loss.avg, epoch + 1)
    tb_logger.add_scalar('[val]: bpp_loss', bpp_loss.avg, epoch + 1)
    tb_logger.add_scalar('[val]: psnr', psnr.avg, epoch + 1)
    tb_logger.add_scalar('[val]: ms-ssim', ms_ssim.avg, epoch + 1)
    tb_logger.add_scalar('[val]: charbonnier', charbonnier.avg, epoch + 1)
    tb_logger.add_scalar('[val]: lpips', lpips.avg, epoch + 1)

    logger_val.info(
        f"Test epoch {epoch + 1}: Average losses: "
        f"Loss: {loss.avg:.4f} | "
        f"Char: {charbonnier.avg:.4f} | "
        f"LPIPS: {lpips.avg:.4f} | "
        f"Style: {style_loss.avg:.4f} | "
        f"Adv: {adv_loss.avg:.4f} | "
        f"Bpp: {bpp_loss.avg:.4f} | "
        f"Aux: {aux_loss.avg:.2f} | "
        f"PSNR: {psnr.avg:.6f} dB | "
        f"MS-SSIM: {ms_ssim.avg:.6f} dB"
    )
    
    return lpips.avg + bpp_loss.avg


def test_one_epoch_gan_face(epoch, test_dataloader, model, model_disc,criterion, save_dir, logger_val, tb_logger, config):
    model.eval()
    device = next(model.parameters()).device
    gan_loss = GANLoss('hinge', loss_weight=2.0, real_label_val=1.0, fake_label_val=0.0)

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    charbonnier = AverageMeter()
    dists = AverageMeter()
    style_loss = AverageMeter() 
    adv_loss = AverageMeter() 
    aux_loss = AverageMeter()
    face_loss = AverageMeter()
    psnr = AverageMeter()
    ms_ssim = AverageMeter()

    with torch.no_grad():
        for i, d in enumerate(test_dataloader):
            d = d.to(device)
            mask = d[:, 3:, :, :]   # /255.0     mask_roi
            img = d[:, :3, :, :]    # /255.0
            
            # Handle padding for images not divisible by 64
            B, C, H, W = img.shape
            pad_h = 0
            pad_w = 0
            if H % 64 != 0:
                pad_h = 64 * (H // 64 + 1) - H
            if W % 64 != 0:
                pad_w = 64 * (W // 64 + 1) - W
            
            # Pad image if needed
            if pad_h > 0 or pad_w > 0:
                img_pad = F.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0)
            else:
                img_pad = img
            
            # Forward pass with padded image
            out_net = model(img_pad)
            
            # Crop output back to original size before computing loss
            if pad_h > 0 or pad_w > 0:
                out_net['x_hat'] = out_net['x_hat'][:, :, :H, :W]
            
            # Now compute loss with correctly sized tensors
            out_criterion = criterion(out_net, img, mask)

            pred_fake = model_disc(out_criterion["x_tidle"])
            loss_G_fake = gan_loss(pred_fake, False, is_disc=False)
            loss_G_total = (config["lambda_char"]* out_criterion["charbonnier"] + config.get("lambda_dists", 1.0) * out_criterion["dists"] + config["lambda_style"] * out_criterion["style_loss"] + config["lambda_gan"] * loss_G_fake +  out_criterion["bpp_loss"] + config["lambda_face"] * out_criterion["face_loss"])
            
            out_criterion["loss"] =  torch.mean(loss_G_total)
            out_criterion["dists"] = torch.mean(out_criterion["dists"])
            out_criterion["face_loss"] = torch.mean(out_criterion["face_loss"])

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"].item())
            loss.update(loss_G_total.item())
            if isinstance(out_criterion["dists"], torch.Tensor):
                dists.update(out_criterion["dists"].item())
            style_loss.update(out_criterion["style_loss"].item())
            adv_loss.update(loss_G_fake.item())
            face_loss.update(out_criterion["face_loss"].item())
            if out_criterion["charbonnier"] is not None:
                charbonnier.update(out_criterion["charbonnier"].item())

            rec = torch2img(out_net['x_hat'])
            img = torch2img(img)
            p, m = compute_metrics(rec, img)
            psnr.update(p)
            ms_ssim.update(m)

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            rec.save(os.path.join(save_dir, '%03d_rec.png' % i))
            img.save(os.path.join(save_dir, '%03d_gt.png' % i))

    tb_logger.add_scalar('{}'.format('[val]: loss'), loss.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format('[val]: bpp_loss'), bpp_loss.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format('[val]: psnr'), psnr.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format('[val]: ms-ssim'), ms_ssim.avg, epoch + 1)
    if dists.count > 0:
        tb_logger.add_scalar('{}'.format('[val]: dists'), dists.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format('[val]: style loss'), style_loss.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format('[val]: face loss'), face_loss.avg, epoch + 1)

    dists_str = f"{dists.avg:.6f}" if dists.count > 0 else "N/A"
    logger_val.info(
        f"Test epoch {epoch + 1}: Average losses: "
        f"Loss: {loss.avg:.4f} | "
        f"Charbonnier loss: {charbonnier.avg:.4f} | "
        f"DISTS loss: {dists_str} | "
        f"Style loss: {style_loss.avg:.4f} | "
        f"Adv loss: {adv_loss.avg:.4f} | "
        f"Face loss: {face_loss.avg:.6f} | "
        f"Bpp rate loss: {bpp_loss.avg:.4f} | "
        f"Aux loss: {aux_loss.avg:.2f} | "
        f"PSNR: {psnr.avg:.4f} dB | "
        f"MS-SSIM: {ms_ssim.avg:.6f} dB"

    )
    tb_logger.add_scalar('{}'.format('[val]: charbonnier loss'), charbonnier.avg, epoch + 1)
    
    return loss.avg
