import warnings
warnings.filterwarnings("ignore", message="Failed to load image Python extension:")

import os
import glob
import random
import numpy as np
import warnings
import torch
from torch.utils.data import IterableDataset
from torchvision import transforms
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from collections import defaultdict

# ------------------------------
# Iterable Dataset to Load One Chunk at a Time
# ------------------------------
class ChunkedIterableDataset(IterableDataset):
    def __init__(self, cache_dir, 
        transform=None, 
        augment=False, 
        multi_view_mode=False,
        shuffle_chunks=True, 
        shuffle_within_chunk=True, 
        selected_t_indices=None,
        max_real=None, 
        max_fake=None, 
        rank=0, 
        world_size=1, 
        image_size=(224, 224)):
        """
        Args:
            cache_dir (str): Directory containing chunk .pt files.
            transform (callable, optional): Transformation to apply to images.
            shuffle_chunks (bool): Shuffle chunk order each epoch.
            shuffle_within_chunk (bool): Shuffle samples within each chunk.
        """
        self.cache_dir = cache_dir

        all_files = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))
        if max_real is not None or max_fake is not None:
            real_files = [f for f in all_files if os.path.basename(f).startswith("real_chunk_")]
            fake_files = [f for f in all_files if os.path.basename(f).startswith("fake_chunk_")]
            if max_real is not None:
                real_files = real_files[:max_real]
            if max_fake is not None:
                fake_files = fake_files[:max_fake]
            self.chunk_files = real_files + fake_files
        else:
            self.chunk_files = all_files

        self.rank = rank
        self.world_size = world_size
        self.transform = transform
        self.image_size = image_size
        self.shuffle_chunks = shuffle_chunks
        self.shuffle_within_chunk = shuffle_within_chunk
        self.selected_t_indices = selected_t_indices
        self.total_length = self.compute_total_length()

        # Augmentation
        self.augment = augment
        self.multi_view_mode = multi_view_mode
        
        self.multi_view_augs = [
            A.HorizontalFlip(p=1.0),
            A.RandomRotate90(p=1.0),
            A.OneOf([
                A.OneOf([A.GaussNoise(p=0.1), A.GaussianBlur(p=0.1)], p=0.2),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.8)
            ], p=1.0),
            A.OneOf([A.CoarseDropout(), A.GridDropout()], p=1.0)
        ]
        self.multi_view_pipelines = [self._wrap_pipeline(aug) for aug in self.multi_view_augs]

        self.one_view_aug = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.3),
            A.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0), ratio=(0.95, 1.05), p=0.3),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.ImageCompression(quality_lower=85, quality_upper=95, p=1.0),
            ], p=0.3),
            A.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.01, p=0.2),

            A.Resize(224,224),
            A.Normalize(mean=(0.5,)*3, std=(0.5,)*3),
            ToTensorV2(),
        ])

        self.replay_pipeline = A.ReplayCompose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.3),
            A.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0), ratio=(0.95, 1.05), p=0.3),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.ImageCompression(quality_lower=85, quality_upper=95, p=1.0),
            ], p=0.3),
            A.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.01, p=0.2),

            A.Resize(224, 224),
            A.Normalize(mean=(0.5,)*3, std=(0.5,)*3),
            ToTensorV2(),
        ])

    def _wrap_pipeline(self, aug):
        return A.Compose([
            aug,
            A.Resize(*self.image_size),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ])

    def __iter__(self):
        chunk_files = self.chunk_files.copy()
        if self.shuffle_chunks:
            random.shuffle(chunk_files)
        # Shard the files using the provided rank/world_size
        chunk_files = chunk_files[self.rank::self.world_size]

        for chunk_file in chunk_files:
            chunk_data = torch.load(chunk_file, weights_only=False)
            samples = list(chunk_data.values())
            if self.shuffle_within_chunk:
                random.shuffle(samples)
            for sample in samples:
                file_path = sample["file_path"]

                try:
                    image = Image.open(file_path).convert("RGB")
                except Exception as e:
                    print(f"Failed to load image {file_path}: {e}")
                    continue

                label = sample["label"]
                latent_seq = sample["latent_seq"]

                if self.selected_t_indices is not None:
                    latent_seq = latent_seq[self.selected_t_indices]
                
                loss_map = sample["loss_map"] if "loss_map" in sample and sample["loss_map"] is not None else torch.zeros((4, 28, 28))

                image_np = np.array(image) 

                # 1) If replay exists, apply it exclusively
                replay = sample.get("replay", None)
                if replay is not None:
                    try:
                        resize_first = A.Resize(224, 224)
                        image_resized = resize_first(image=image_np)["image"]

                        out = self.replay_pipeline.replay(replay, image=image_resized)["image"]
                        yield out, label, latent_seq, loss_map
                        continue
                    except Exception as e:
                        print(f"Replay failed on {file_path}: {e}")

                if self.augment:
                    if self.multi_view_mode:
                        for pipeline in self.multi_view_pipelines:
                            try:
                                out = pipeline(image=image_np)["image"]
                                yield out, label, latent_seq, loss_map
                            except Exception as e:
                                print(f"Augmentation failed on {file_path}: {e}")
                    else:
                        try:
                            out = self.one_view_aug(image=image_np)["image"]
                            yield out, label, latent_seq, loss_map
                        except Exception as e:
                            print(f"LaRE-style augmentation failed on {file_path}: {e}")
                else:
                    if self.transform:
                        out = self.transform(image=image_np)["image"]
                    else:
                        out = (transforms.ToTensor()(image) - 0.5) * 2
                    yield out, label, latent_seq, loss_map

    def compute_total_length(self):
        total = 0
        for chunk_file in self.chunk_files:
            chunk_data = torch.load(chunk_file, weights_only=False)
            total += len(chunk_data)
        return total

    def __len__(self):
        # Only consider the chunk files assigned to this process
        sharded_files = self.chunk_files[self.rank::self.world_size]
        total = 0
        for chunk_file in sharded_files:
            chunk_data = torch.load(chunk_file, weights_only=False)
            total += len(chunk_data)
        return total