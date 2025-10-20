import os
import json
import random
import shutil
import wandb
import argparse
import datetime
import importlib
import logging as logger
import numpy as np
import albumentations as A
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
from torch import optim
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pytorch_warmup import LinearWarmup
from PIL import Image
from sklearn.metrics import (accuracy_score, average_precision_score, precision_recall_curve, roc_auc_score, roc_curve)
from tabulate import tabulate
from tqdm import tqdm
from dataset import ChunkedIterableDataset
from pytorch_metric_learning.losses import NTXentLoss
from albumentations.pytorch import ToTensorV2
import warnings
warnings.filterwarnings("ignore", message="Failed to load image Python extension:")

wandb.login(key="6193f2d9d42be82d9ae54f86340762df5c141087")

logger.basicConfig(level=logger.INFO,
                   format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')

class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class FakeWriter:
    def __init__(self):
        pass

    def add_scalar(self, p1, p2, p3):
        pass

def normalize_tensor(x):
    return (x - 0.5) * 2

def search_best_acc(gt_labels, pred_probs):
    best_acc = -1
    best_threshold = -1
    for thresh in sorted(pred_probs.tolist()):
        pred_binary = np.where(np.array(pred_probs) > thresh, 1, 0)
        acc = accuracy_score(gt_labels, pred_binary)
        if acc > best_acc:
            best_acc = acc
            best_threshold = thresh
    return best_acc, best_threshold

def save_checkpoint(model, optimizer, scheduler, epoch, best_acc, best_auc, output_dir, filename):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_acc': best_acc,
        'best_auc': best_auc
    }, f"{output_dir}/{filename}")

def train_one_epoch(data_loader, model, optimizer, cur_epoch, loss_meter, args, device, writer, ngpus_per_node, wandbrun, scaler=None):
    """Train one epoch."""
    model.train()
    loss_meter.reset()
    total_train_loss, total_ce_loss, total_contrastive_loss = 0, 0, 0

    for images, labels, latent_seqs, loss_maps in tqdm(data_loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device).flatten()
        features = latent_seqs.to(device)

        optimizer.zero_grad()

        with autocast(device_type='cuda', enabled=args.is_amp):
            # CE loss
            logits, embeds = model(features) if args.latent_only else model(images, features)
            loss_ce = args.criterion_ce(logits, labels)

            # Contrastive loss
            loss_contrast = args.criterion_contrast(embeds, labels)

            # Total loss
            loss = (1 - args.lambda_contrast) * loss_ce + args.lambda_contrast * loss_contrast

        total_ce_loss += loss_ce.item()
        total_contrastive_loss += loss_contrast.item()
        total_train_loss += loss.item()

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        loss_meter.update(loss.item(), images.shape[0])

    avg_ce_loss = total_ce_loss / len(data_loader.dataset)
    avg_contrastive_loss = total_contrastive_loss / len(data_loader.dataset)
    avg_train_loss = total_train_loss / len(data_loader.dataset)
    wandbrun.log({"epoch": cur_epoch, f"CE_loss": avg_ce_loss, f"Contrastive_loss": avg_contrastive_loss, f"Train_loss": avg_train_loss})
    logger.info('End Training')
    return avg_train_loss


def validation(data_loader, model, args, device, ngpus_per_node, wandbrun, epoch=-1):
    """Run model validation."""
    logger.info('Starting validation')
    model.eval()
    total_ce_loss, total_contrastive_loss, total_validation_loss = 0, 0, 0
    gt_labels_list, prob_labels_list = [], []

    with torch.no_grad():
       for images, labels, latent_seqs, loss_maps in tqdm(data_loader, desc="Validation", leave=False):
            images, labels = images.to(device), labels.to(device).flatten()
            features = latent_seqs.to(device)

            with autocast(device_type='cuda', enabled=args.is_amp):
                # CE loss
                logits, embeds = model(features) if args.latent_only else model(images, features)
                loss_ce = args.criterion_ce(logits, labels)

                # Contrastive loss
                loss_contrast = args.criterion_contrast(embeds, labels)

                # Total loss
                loss = (1 - args.lambda_contrast) * loss_ce + args.lambda_contrast * loss_contrast

            total_ce_loss += loss_ce.item()
            total_contrastive_loss += loss_contrast.item()
            total_validation_loss += loss.item()
            prob = torch.softmax(logits, dim=-1)[:, 1]

            if torch.isnan(prob).any():
                logger.warning("Found NaNs in softmax outputs.")
                logger.warning(f"Logits: {logits}")
                continue  # Skip this batch

            gt_labels_list.append(labels.cpu())
            prob_labels_list.append(prob.cpu())

    avg_ce_loss = total_ce_loss / len(data_loader.dataset)
    avg_contrastive_loss = total_contrastive_loss / len(data_loader.dataset)
    avg_valid_loss = total_validation_loss / len(data_loader.dataset)
    wandbrun.log({"epoch": epoch, f"Validation_CE_loss": avg_ce_loss, f"Validation_Contrastive_loss": avg_contrastive_loss, "Validation_loss": avg_valid_loss})
    
    gt_labels_list = torch.cat(gt_labels_list, dim=0).numpy()
    prob_labels_list = torch.cat(prob_labels_list, dim=0).numpy()

    fpr, tpr, thres = roc_curve(gt_labels_list, prob_labels_list)
    thresh = thres[len(thres) // 2]
    
    precision, recall, thresholds = precision_recall_curve(gt_labels_list, prob_labels_list)
    f_score = (2 * precision * recall) / (precision + recall + 1e-10)
    thresh = thresholds[np.argmax(f_score)]

    pred_labels_list = np.where(prob_labels_list > thresh, 1, 0)
    auc = roc_auc_score(gt_labels_list, prob_labels_list)
    accuracy = accuracy_score(gt_labels_list, pred_labels_list)
    ap = average_precision_score(gt_labels_list, prob_labels_list)

    best_acc, best_thresh = search_best_acc(gt_labels_list, prob_labels_list)
    logger.info(f'Search ACC: {best_acc}')

    r_acc = accuracy_score(gt_labels_list[gt_labels_list == 0], prob_labels_list[gt_labels_list == 0] > 0.5)
    f_acc = accuracy_score(gt_labels_list[gt_labels_list == 1], prob_labels_list[gt_labels_list == 1] > 0.5)
    raw_acc = accuracy_score(gt_labels_list, prob_labels_list > 0.5)

    wandbrun.log({
        "Val/AUC": auc,
        "Val/Accuracy real": r_acc,
        "Val/Accuracy fake": f_acc,
        "Val/Raw_Accuracy": raw_acc,
        "Val/Best_Accuracy": best_acc,
        "Val/AP": ap
    })

    return auc, best_acc, ap, raw_acc, r_acc, f_acc, avg_valid_loss

def main_worker(local_rank, args, output_dir):
    """
    Main function for distributed multi-node training using torchrun.
    This version is refined based on the PyTorch multinode DDP tutorial.
    """
    local_rank = int(os.environ["LOCAL_RANK"])      # local GPU index on this node
    global_rank = int(os.environ["RANK"])           # unique rank across all nodes
    # world_size can either be passed in or taken from the process group after initialization. Here we retrieve it after group initialization for consistency.
    
    # Set the current CUDA device using local_rank
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    
    # Initialize the process group using environment variables (torchrun automatically sets WORLD_SIZE and RANK in the environment)
    dist.init_process_group(backend="nccl", init_method="env://")
    
    # For extra clarity, get the actual values from the process group
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()

    print(f"Global rank: {global_rank}, Local rank: {local_rank}, using device: {torch.cuda.get_device_name(local_rank)}")

    wandbrun = wandb.init(
        project="lare",
        name=os.environ.get("WANDB_NAME", None),
        dir=os.environ.get("WANDB_DIR", "./wandb_logs"),
        id=os.environ.get("WANDB_RUN_ID", None),
        resume="allow" if os.environ.get("WANDB_RUN_ID") else None,
        config=vars(args),
    ) if global_rank == 0 else wandb.init(mode="disabled")
        
    # Image preprocessing
    mean_std = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) if args.imagenet_norm else ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=mean_std[0], std=mean_std[1]),
        ToTensorV2()
    ])
    
    # Create the distributed training and validation datasets.
    train_dataset = ChunkedIterableDataset(
        cache_dir=args.latent_dir_train, 
        transform=transform,
        augment=args.augment,
        multi_view_mode=args.multi_view_mode,
        shuffle_chunks=True,     
        shuffle_within_chunk=True,
        selected_t_indices=args.selected_t_indices,
        max_real=args.max_real_train,
        max_fake=args.max_fake_train,
        rank=global_rank,
        world_size=world_size
    )
    logger.info(f'Train dataset size: {len(train_dataset)}')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers)
    
    val_dataset = ChunkedIterableDataset(
        cache_dir=args.latent_dir_validation, 
        transform=transform,
        augment=False,
        multi_view_mode=False,
        shuffle_chunks=False,
        shuffle_within_chunk=False,
        selected_t_indices=args.selected_t_indices,
        max_real=args.max_real_valid,
        max_fake=args.max_fake_valid,
        rank=global_rank,
        world_size=world_size
    )
    logger.info(f'Valid/Test dataset size: {len(val_dataset)}')
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.workers)
    
    # ============================
    # Build and wrap the model in DDP
    # ============================
    if args.latent_only:
        model = getattr(importlib.import_module('model'), "LatentOnlyClassifier")(
            num_class=args.num_class, 
            num_timesteps=len(args.selected_t_indices), 
        )
    else:
        if args.model_type == "LatentTrajectoryClassifier":
            model = getattr(importlib.import_module('model'), args.model_type)(
                num_class=args.num_class, 
                clip_type=args.clip_type, 
                num_timesteps=len(args.selected_t_indices), 
                num_heads=8, 
                self_attention_latents=args.self_attention_latents,
                weighted_average=args.weighted_average,
                single_decoder=args.single_decoder,
                process_latents_separately=args.process_latents_separately,
                decoders_per_timestep=args.decoders_per_timestep,
                use_cls_token=args.use_cls_token,
                positional_embeddings=args.positional_embeddings,
                xavier_kaiming_init=args.xavier_kaiming_init,
                return_clip_global_feats=args.return_clip_global_feats,
                latent_visual_refinement=not args.no_latent_visual_refinement,
                prompt_tuning=args.prompt_tuning,
                qkNorm=args.qkNorm
            )
        elif args.model_type == "LatentTrajectoryClassifierSingleCLS":
            model = getattr(importlib.import_module('model'), args.model_type)(
                num_class=args.num_class, 
                clip_type=args.clip_type, 
                num_timesteps=len(args.selected_t_indices), 
                num_heads=8, 
                cls_self_attn_layers=2, 
                cls_self_attn_heads=4,
                positional_embeddings=args.positional_embeddings,
                xavier_kaiming_init=args.xavier_kaiming_init,
                return_clip_global_feats=args.return_clip_global_feats,
                use_cls_token=args.use_cls_token,
                qkNorm=args.qkNorm
            )
        
    model = model.to(device)

    # Freeze the backbone
    if args.freeze_backbone:
        cm = model.clip_model

        # --- ConvNeXt backbone ---
        if hasattr(cm, "stages"):
            # Freeze everything
            for name, param in cm.named_parameters():
                param.requires_grad = False

            # Optionally unfreeze some final layers
            # for name, param in cm.named_parameters():
            #     if (name.startswith("stages.3") or "proj_reduction" in name or "proj_global" in name):
            #         param.requires_grad = True

        # --- ViT backbone ---
        elif hasattr(cm.visual, "transformer"):
            # Freeze everything
            for name, param in cm.named_parameters():
                param.requires_grad = False

            # Optionally unfreeze the last transformer block
            # n_layers = len(cm.visual.transformer.resblocks)
            # last_idx = n_layers - 1
            # prefix = f"visual.transformer.resblocks.{last_idx}"

            # for name, param in cm.named_parameters():
            #     if name.startswith(prefix):
            #         param.requires_grad = True
        else:
            # other backbones—either leave them all trainable or add more cases
            pass

    # Wrap with DistributedDataParallel using the local device
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    # Set up the loss criterion (with optional label smoothing)
    args.criterion_ce = nn.CrossEntropyLoss().to(device)
    args.criterion_contrast = NTXentLoss(temperature=args.temperature)
    
    # Create a TensorBoard writer only on global rank 0
    writer = SummaryWriter(log_dir=os.path.join(output_dir)) if global_rank == 0 else FakeWriter()
    
    # Set up optimizer and scheduler
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(parameters, lr=args.lr, weight_decay=4e-5)
    
    if not args.is_warmup:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    else:
        warmup_epochs = max(1, int(0.1 * args.epochs))
        # 1) Cosine decay over (total_epochs – warmup_epochs)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - warmup_epochs, eta_min=1e-6)
        # 2) Linear warmup over warmup_epochs
        warmup_scheduler = LinearWarmup(optimizer, warmup_period=warmup_epochs)
    
    loss_meter = AverageMeter()
    scaler = GradScaler() if args.is_amp else None
    
    start_epoch = 0
    best_acc = -1
    best_auc = -1
    
    # Optionally resume training from a checkpoint
    if args.resume_training:
        logger.info(f"Resuming training from checkpoint: {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        best_auc = checkpoint['best_auc']
        logger.info(f"Resumed from epoch {start_epoch}, best accuracy: {best_acc}, best AUC: {best_auc}")
    
    # Main training loop
    for epoch in range(start_epoch, args.epochs):
        train_one_epoch(train_loader, model, optimizer, epoch, loss_meter, args, device, writer, world_size, wandbrun, scaler)
        
        # Save checkpoint only on the master process (global rank 0)
        if global_rank == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, best_acc, best_auc, output_dir, "latest.pth")
        
        # Run validation
        val_auc, val_acc, val_ap, val_raw_acc, val_r_acc, val_f_acc, val_loss = validation(val_loader, model, args, device, world_size, wandbrun, epoch)
        logger.info(f'Score: Validation AUC: {val_auc:.4f}, Acc: {val_acc:.4f}, AP: {val_ap:.4f}')

        if global_rank == 0:
            if val_acc > best_acc:
                best_acc = val_acc
                save_checkpoint(model, optimizer, scheduler, epoch, best_acc, best_auc, output_dir, "ACC_best.pth")
            if val_auc > best_auc:
                best_auc = val_auc
                save_checkpoint(model, optimizer, scheduler, epoch, best_acc, best_auc, output_dir, "AUC_best.pth")
        
        # Update learning rate scheduler
        if args.is_warmup:
            with warmup_scheduler.dampening():
                scheduler.step()
        else:
            scheduler.step()

    # Clean up distributed resources
    dist.destroy_process_group()


if __name__ == '__main__':
    conf = argparse.ArgumentParser()

    # === Experiment Settings ===
    conf.add_argument("--exp_name", type=str, default='', help="Experiment name for logging and saving")
    conf.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    conf.add_argument("--out_dir", type=str, default='checkpoints', help="Output directory for checkpoints")

    # === Model Configuration ===
    conf.add_argument("--model_type", type=str, default='LatentTrajectoryClassifier', help="Model type")
    conf.add_argument("--clip_type", type=str, default='convnext_base_in22k', help="CLIP variant")
    conf.add_argument("--num_class", type=int, default=2, help='Number of output classes')
    conf.add_argument("--latent_only", action='store_true', help="Use only latent inputs")
    conf.add_argument("--freeze_backbone", action='store_true', help="Freeze all backbone layers except last")
    conf.add_argument("--use_cls_token", action='store_true', help="Use CLS token in model")
    conf.add_argument("--return_clip_global_feats", action='store_true', help="Return global CLIP features")
    conf.add_argument("--no_latent_visual_refinement", action='store_true', help="Disable latent refinement")
    conf.add_argument("--prompt_tuning", action='store_true', help="Enable prompt tuning")
    
    # === Architecture Flags ===
    conf.add_argument("--self_attention_latents", action='store_true')
    conf.add_argument("--weighted_average", action='store_true')
    conf.add_argument("--single_decoder", action='store_true')
    conf.add_argument("--process_latents_separately", action='store_true')
    conf.add_argument("--xavier_kaiming_init", action='store_true')
    conf.add_argument("--positional_embeddings", action='store_true')
    conf.add_argument("--qkNorm", action='store_true')
    conf.add_argument("--decoders_per_timestep", type=int, default=1)

    # === Latent Timesteps Selection ===
    conf.add_argument(
        "--tracked_timesteps", 
        type=json.loads, 
        default="[981, 741, 521, 261, 1]",
        help="List of diffusion timesteps to track (as JSON-formatted list)."
    )
    conf.add_argument(
        "--selected_t_indices", 
        type=json.loads, 
        default="[0, 1, 2, 3, 4]",
        help="Indices of selected tracked timesteps (e.g., [0, 1, 2, 3, 4])."
    )

    # === Dataset & Dataloader ===
    conf.add_argument("--latent_dir_train", type=str, required=True, help="Folder containing train .pt chunk files.")
    conf.add_argument("--latent_dir_validation", type=str, required=True, help="Folder containing validation .pt chunk files.")
    conf.add_argument('--max_real_train', type=int, default=None)
    conf.add_argument('--max_fake_train', type=int, default=None)
    conf.add_argument('--max_real_valid', type=int, default=None)
    conf.add_argument('--max_fake_valid', type=int, default=None)
    conf.add_argument("--augment", action='store_true', default=False)
    conf.add_argument("--multi_view_mode", action='store_true', default=False)
    conf.add_argument("--imagenet_norm", action='store_true', default=False)
    conf.add_argument("--data_size", type=int, nargs=2, default=(224, 224), help="Image dimensions")

    # === Optimization ===
    conf.add_argument('--lr', type=float, default=1e-4)
    conf.add_argument('--epochs', type=int, default=20)
    conf.add_argument('--batch_size', type=int, default=48)
    conf.add_argument('--test_batch_size', type=int, default=48)
    conf.add_argument('--temperature', type=float, default=0.5)
    conf.add_argument('--lambda_contrast', type=float, default=0)
    conf.add_argument('--smoothing', type=float, default=0.1)
    conf.add_argument('--is_amp', action='store_true', help='Use mixed-precision training')
    conf.add_argument('--is_warmup', action='store_true', help='Enable warmup schedule')

    # === Resume Checkpoint ===
    conf.add_argument("--resume_training", action='store_true')
    conf.add_argument("--checkpoint_path", type=str, default="")

    # === System ===
    conf.add_argument('--rank', default=-1, type=int, help='Node rank')
    conf.add_argument('-j', '--workers', default=4, type=int)
    conf.add_argument("--local_rank", type=int, default=0, help="Local rank assigned by torchrun")
    
    args = conf.parse_args()
    logger.info(args)

    ngpus_per_node = torch.cuda.device_count() if torch.cuda.is_available() else 1

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    # Device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    logger.info('Train Mode')
    date_now = datetime.datetime.now()
    date_now = args.exp_name + '_Log_v%02d%02d%02d%02d' % (
        date_now.month, date_now.day, date_now.hour, date_now.minute)
    args.time = date_now

    # 1. Create directory
    out_dir = os.path.join(args.out_dir, args.time)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)

    # 2. Train model
    main_worker(args.local_rank, args, out_dir)
    