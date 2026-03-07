"""
Phase 2 BASELINE Training: Original HFLIC with GAN loss.

This is the ORIGINAL HFLIC Phase 2 for fair comparison.
Loss = 3e-4 × Charbonnier(sum) + 2.0 × LPIPS + Style + GAN + BPP

NO MSE, NO VSI - just the original HFLIC components.
"""

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
from utils.training import train_one_epoch_gan_baseline
from utils.testing import test_one_epoch_gan_baseline
from loss.rd_loss import RateDistortionPOELICLossBaseline
from utils.args import train_options
from config.config_5group import model_config
from models.models import ELIC
from models.disc import Discriminator, init_weights
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
    seed = setup_seed()
    args = train_options()
    config = model_config()

    os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join(str(id) for id in args.gpu_id)
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
        [transforms.RandomCrop(args.patch_size), transforms.RandomHorizontalFlip(), transforms.ToTensor()]
    )
    test_transforms = transforms.Compose(
        [transforms.ToTensor()]
    )

    train_dataset = OpenImagesDataset(
        root=args.train_root,
        split=args.train_split,
        transform=train_transforms,
    )
    if args.eval_split in ["train", "validation", "test"]:
        test_dataset = OpenImagesDataset(
            root=args.train_root,
            split=args.eval_split,
            transform=test_transforms,
        )
    else:
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
    net_disc = Discriminator()

    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)
        net_disc = CustomDataParallel(net_disc)
    
    net = net.to(device)
    net_disc.to(device)
    
    init_weights(net_disc, init_type='normal', init_gain=0.02)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 100], gamma=0.1)

    optimizer_D = torch.optim.Adam(net_disc.parameters(), lr=args.lr_D)
    lr_scheduler_D = optim.lr_scheduler.MultiStepLR(optimizer_D, milestones=[80, 100], gamma=0.1)

    criterion = RateDistortionPOELICLossBaseline(lmbda=args.lmbda, device=device, gpu_id=args.gpu_id, metrics='mse')
    
    if args.checkpoint != None:
        checkpoint = torch.load(args.checkpoint)
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint['optimizer'])
        aux_optimizer.load_state_dict(checkpoint['aux_optimizer'])
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[450,550], gamma=0.1)
        lr_scheduler._step_count = checkpoint['lr_scheduler']['_step_count']
        lr_scheduler.last_epoch = checkpoint['lr_scheduler']['last_epoch']
        start_epoch = checkpoint['epoch']
        best_loss = 1e10  # Reset for Phase 2 - different loss metric than Phase 1
        current_step = start_epoch * math.ceil(len(train_dataloader.dataset) / args.batch_size)
        checkpoint = None
    else:
        start_epoch = 0
        best_loss = 1e10
        current_step = 0

    logger_train.info("=" * 80)
    logger_train.info("PHASE 2 BASELINE: Original HFLIC with GAN")
    logger_train.info("Loss = 3e-4 × Charbonnier + 2.0 × LPIPS + Style + GAN + BPP")
    logger_train.info("NO MSE, NO VSI - Original HFLIC for fair comparison")
    logger_train.info("=" * 80)
    logger_train.info(f"Seed: {seed}")
    logger_train.info(args)
    optimizer.param_groups[0]['lr'] = args.learning_rate
    
    for epoch in range(start_epoch, args.epochs):
        logger_train.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        current_step = train_one_epoch_gan_baseline(
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

        torch.cuda.empty_cache()
        gc.collect()

        save_dir = os.path.join('./experiments', args.experiment, 'val_images', '%03d' % (epoch + 1))
        loss = test_one_epoch_gan_baseline(epoch, test_dataloader, net, net_disc, criterion, save_dir, logger_val, tb_logger, config)
        lr_scheduler.step()
        lr_scheduler_D.step()

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        net.update(force=True)
        
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
                logger_val.info('Best checkpoint saved.')
                logger_train.info(f"Best checkpoint saved at epoch {epoch + 1} with loss {best_loss:.4f}")

    logger_train.info("=" * 80)
    logger_train.info("PHASE 2 BASELINE TRAINING COMPLETE")
    logger_train.info(f"Best loss: {best_loss:.4f}")
    logger_train.info("=" * 80)

if __name__ == '__main__':
    main()
