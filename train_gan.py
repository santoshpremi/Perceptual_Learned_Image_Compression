import os
import random
import logging
import gc
from PIL import ImageFile, Image
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from compressai.datasets import ImageFolder
from utils.logger import setup_logger
from utils.utils import CustomDataParallel, save_checkpoint
from utils.optimizers import configure_optimizers
from utils.training import train_one_epoch_gan
from utils.testing import test_one_epoch_gan
from loss.rd_loss import RateDistortionPOELICLossPhase2
from utils.args import train_options
from config.config_5group import model_config
from models.models import ELIC
from models.disc import Discriminator, MultiScaleDiscriminator, init_weights
from datasets.open_images import OpenImagesDataset
import random
import numpy as np

def setup_seed(seed=3407):
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enable = False
    return seed 

def main():
    torch.backends.cudnn.benchmark = True
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    Image.MAX_IMAGE_PIXELS = None
    seed =  setup_seed()
    args = train_options()
    config = model_config()

    os.environ['CUDA_VISIBLE_DEVICES'] =  ', '.join(str(id) for id in args.gpu_id)
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    if not os.path.exists(os.path.join('./experiments', args.experiment)):
        os.makedirs(os.path.join('./experiments', args.experiment))

    setup_logger('train', os.path.join('./experiments', args.experiment), 'train_' + args.experiment, level=logging.INFO,
                        screen=True, tofile=True)
    setup_logger('val', os.path.join('./experiments', args.experiment), 'val_' + args.experiment, level=logging.INFO,
                        screen=True, tofile=True)

    logger_train = logging.getLogger('train')
    logger_val = logging.getLogger('val')
    tb_logger = SummaryWriter(log_dir='./tb_logger/' + args.experiment)

    if not os.path.exists(os.path.join('./experiments', args.experiment, 'checkpoints')):
        os.makedirs(os.path.join('./experiments', args.experiment, 'checkpoints'))

    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )
    test_transforms = transforms.Compose(
        [transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
    )

    train_dataset = OpenImagesDataset(
        root=args.train_root,
        split=args.train_split,
        transform=train_transforms,
    )
    # Use Open Images for evaluation if eval_split is a valid Open Images split, otherwise use ImageFolder (e.g., for Kodak, CLIC2022)
    if args.eval_split in ["train", "validation", "test"]:
        # For Open Images splits, use the same root as training (open_images directory)
        test_dataset = OpenImagesDataset(
            root=args.train_root,
            split=args.eval_split,
            transform=test_transforms,
        )
    else:
        # For other datasets like Kodak or CLIC2022, use ImageFolder
        # ImageFolder expects root/split to be the directory containing images
        # e.g., root="data", split="kodak" → looks for "data/kodak/"
        # e.g., root="data", split="clic2022" → looks for "data/clic2022/"
        test_dataset = ImageFolder(args.eval_root, split=args.eval_split, transform=test_transforms)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    net = ELIC(config=config)
    
    # Use Multi-Scale Discriminator for improved GAN stability (if enabled in config)
    use_multi_scale = config.get("use_multi_scale_disc", True)
    if use_multi_scale:
        net_disc = MultiScaleDiscriminator(num_scales=3)
        print("Using Multi-Scale Discriminator (3 scales) for GAN stability")
    else:
        net_disc = Discriminator()
        print("Using single-scale Discriminator")

    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)
        net_disc = CustomDataParallel(net_disc)
    
    net = net.to(device)
    net_disc.to(device)
    
    init_weights(net_disc, init_type='normal', init_gain=0.02)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 100], gamma=0.1)

    # TTUR: Two-Timescale Update Rule - Discriminator learns faster than Generator
    # This helps maintain balance between G and D during training
    ttur_ratio = config.get("ttur_ratio", 4.0)
    lr_D = args.lr_D * ttur_ratio
    optimizer_D = torch.optim.Adam(net_disc.parameters(), lr=lr_D)
    lr_scheduler_D = optim.lr_scheduler.MultiStepLR(optimizer_D, milestones=[80, 100], gamma=0.1)
    print(f"TTUR enabled: Generator LR={args.learning_rate}, Discriminator LR={lr_D} (ratio={ttur_ratio}x)")

    # Phase 2 loss: Enhanced perceptual loss with TOPIQ-FR
    # Use Rate-Distortion (MSE/MS-SSIM) instead of Charbonnier for consistency with Phase 1
    # Use metrics='mse' for MSE-based or 'ms-ssim' for MS-SSIM based rate-distortion
    criterion = RateDistortionPOELICLossPhase2(lmbda=args.lmbda, device=device, gpu_id=args.gpu_id, metrics='mse')
    
    if args.checkpoint != None:
        checkpoint = torch.load(args.checkpoint)
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint['optimizer'])
        aux_optimizer.load_state_dict(checkpoint['aux_optimizer'])
        # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[450,550], gamma=0.1)
        lr_scheduler._step_count = checkpoint['lr_scheduler']['_step_count']
        lr_scheduler.last_epoch = checkpoint['lr_scheduler']['last_epoch']
        # print(lr_scheduler.state_dict())
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['loss']
        current_step = start_epoch * math.ceil(len(train_dataloader.dataset) / args.batch_size)
        checkpoint = None
    else:
        start_epoch = 0
        best_loss = 1e10
        current_step = 0

    logger_train.info(f"Seed: {seed}")
    logger_train.info(args)
    # logger_train.info(net)
    optimizer.param_groups[0]['lr'] = args.learning_rate
    for epoch in range(start_epoch, args.epochs):
        logger_train.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        current_step = train_one_epoch_gan(
            net,
            net_disc,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            optimizer_D,
            epoch,
            args.clip_max_norm,
            logger_train,
            tb_logger,
            current_step,
            config
        )

        # Clear memory before validation
        torch.cuda.empty_cache()
        gc.collect()

        save_dir = os.path.join('./experiments', args.experiment, 'val_images', '%03d' % (epoch + 1))
        loss = test_one_epoch_gan(epoch, test_dataloader, net, net_disc, criterion, save_dir, logger_val, tb_logger, config)
        # lr_scheduler.step(loss)
        lr_scheduler.step()
        lr_scheduler_D.step()

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        # Update entropy models - use force=False to only update when safe
        try:
            updated = net.update(force=False)
            if not updated:
                logger_train.debug(f"Entropy model update skipped at epoch {epoch + 1}: not yet ready.")
        except Exception as e:
            logger_train.warning(f"Entropy model update failed at epoch {epoch + 1}: {e}")
        
        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best,
                os.path.join('./experiments', args.experiment, 'checkpoints', "checkpoint_%03d.pth.tar" % (epoch + 1))
            )
            if is_best:
                logger_val.info('best checkpoint saved.')

if __name__ == '__main__':
    main()
