import os
import glob
import torch
import json
import argparse
import numpy as np
import albumentations as A
from PIL import Image
from tqdm import tqdm
from albumentations.pytorch import ToTensorV2
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, DDIMScheduler

def extract_latent_trajectorylatents(
    image_paths,
    label,
    image_size,
    device,
    batch_size,
    tracked_timesteps,
    use_prompt_template,
    prompt,
    prompt_template,
    vae, 
    unet,
    tokenizer,
    text_encoder,
    noise_scheduler,
    extra_transform=None,
):
    """
    Extracts a sequence of latents for each image across tracked timesteps
    using a pre-trained diffusion model pipeline.
    """
    results_for_all_images = {}

    # Process in mini-batches
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing batches"):
        batch_paths = image_paths[i: i + batch_size]
        batch_images, batch_file_paths, batch_replay_infos = [], [], []

        for path in batch_paths:
            try:
                img = Image.open(path).convert("RGB").resize(image_size)
                img_np = np.array(img)
            except Exception as e:
                print(f"Failed to load/resize {path}: {e}")
                continue

            if extra_transform is not None:
                augmented = extra_transform(image=img_np)
                img_tensor = augmented["image"] 
                replay = augmented["replay"] 
            else:
                default_transform = A.Compose([
                    A.Resize(224, 224),
                    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                    ToTensorV2()
                ])
                img_tensor = default_transform(image=img_np)["image"]
                replay = None

            batch_images.append(img_tensor)
            batch_file_paths.append(path)
            batch_replay_infos.append(replay)

        if not batch_images:
            continue

        images_tensor = torch.stack(batch_images, dim=0).to(device)
        latents = vae.encode(images_tensor).latent_dist.sample() * vae.config.scaling_factor

        # --- Track latents at specific timesteps ---
        latent_sequences = []
        for t in tracked_timesteps:
            noise = torch.randn_like(latents)
            timesteps = torch.full((latents.shape[0],), t, device=latents.device, dtype=torch.long)

            # Apply noise and UNet prediction
            t_tensor_scheduler = torch.tensor(t, device=device, dtype=torch.long)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            prompts = [prompt_template.replace('[CLS]', str(label)) if use_prompt_template else prompt
                       for _ in batch_file_paths]

            text_inputs = tokenizer(prompts,
                                    max_length=tokenizer.model_max_length,
                                    padding="max_length",
                                    truncation=True,
                                    return_tensors="pt").to(device)
            encoder_hidden_states = text_encoder(text_inputs["input_ids"])[0]
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            step_output = noise_scheduler.step(model_pred, t_tensor_scheduler, noisy_latents)
            current_latents = step_output.prev_sample
            latent_sequences.append(current_latents)

        # [B, T, 4, 32, 32]
        latent_sequences = torch.stack(latent_sequences, dim=1)  

        for j, file_path in enumerate(batch_file_paths):
            results_for_all_images[file_path] = {
                "label": label,
                "latent_seq": latent_sequences[j].cpu(),
                "file_path": file_path,  # For lazy loading later
                "replay": batch_replay_infos[j] # Save the replay info if available
            }

    return results_for_all_images

def process_all_categories(args, device, global_rank, world_size):
    """
    Processes real and fake image datasets and extracts latent representations per chunk.
    Saves each result as a .pt file.
    """
    extensions = ["jpg", "JPG", "jpeg", "JPEG", "png", "PNG"]

    # Augmentations
    if args.augmentations:
        aug = A.ReplayCompose([
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
    else:
        aug = None

    # Load diffusion model components
    vae = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae").to(device).eval()
    unet = UNet2DConditionModel.from_pretrained(args.model_id, subfolder="unet").to(device).eval()
    tokenizer = CLIPTokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.model_id, subfolder="text_encoder").to(device).eval()
    
    # Scheduler
    noise_scheduler = DDIMScheduler.from_pretrained(args.model_id, subfolder="scheduler") if args.ddim \
        else DDPMScheduler.from_pretrained(args.model_id, subfolder="scheduler")
    if args.ddim:
        noise_scheduler.set_timesteps(100)

    # Disable gradient computation for inference
    for model in (vae, unet, text_encoder):
        model.requires_grad_(False)

    for idx in range(len(args.real_folders)):
        cache_dir = args.cache_dirs[idx]
        os.makedirs(cache_dir, exist_ok=True)

        real_image_paths = sorted([p for ext in extensions for p in glob.glob(os.path.join(args.real_folders[idx], f"*.{ext}"))])
        fake_image_paths = sorted([p for ext in extensions for p in glob.glob(os.path.join(args.fake_folders[idx], f"*.{ext}"))])

        # Shard among processes
        real_image_paths = real_image_paths[global_rank::world_size]
        fake_image_paths = fake_image_paths[global_rank::world_size]

        # Subsampling for start/end chunk control
        if args.start != -1:
            real_image_paths = real_image_paths[args.start * 1000:]
            fake_image_paths = fake_image_paths[args.start * 1000:]
        if args.end != -1:
            real_image_paths = real_image_paths[:args.end * 1000]
            fake_image_paths = fake_image_paths[:args.end * 1000]

        # Process and save in chunks
        for split, label, image_paths in [("REAL", 0, real_image_paths), ("FAKE", 1, fake_image_paths)]:
            num_real_chunks = (len(image_paths) + args.chunk_size - 1) // args.chunk_size
            for i in range(num_real_chunks):
                chunk_paths = image_paths[i * args.chunk_size: (i + 1) * args.chunk_size]

                chunk_results = extract_latent_trajectorylatents(
                    image_paths=chunk_paths,
                    label=label,
                    image_size=args.data_size,
                    device=device,
                    batch_size=args.batch_size,
                    tracked_timesteps=args.tracked_timesteps,
                    use_prompt_template=args.use_prompt_template,
                    prompt=args.prompt,
                    prompt_template=args.prompt_template,
                    vae=vae,
                    unet=unet,
                    tokenizer=tokenizer,
                    text_encoder=text_encoder,
                    noise_scheduler=noise_scheduler,
                    extra_transform=aug,
                )
                # Save the replay information (which records the random augmentation parameters)
                for key, sample in chunk_results.items():
                    sample["replay"] = sample.get("replay", None)

                save_idx = i + args.start if args.start != -1 else i
                torch.save(chunk_results, os.path.join(cache_dir, f"{split.lower()}_chunk_{save_idx:04d}.pt"))
                print(f"[Rank {global_rank}] Saved {split} chunk {save_idx} with {len(chunk_results)} samples.")


def main_worker(args):
    """
    Entry point for processing. Handles a single process/device (no distributed logic).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_rank = 0
    world_size = 1

    print(f"[Global Rank {global_rank}] Using device: {device}")
    process_all_categories(args, device, global_rank, world_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latent trajectory extraction.")

    # === Input/Output paths ===
    parser.add_argument("--real_folders", type=str, nargs='+', required=True, help="Path(s) to folder(s) containing real images.")
    parser.add_argument("--fake_folders", type=str, nargs='+', required=True, help="Path(s) to folder(s) containing fake images.")
    parser.add_argument("--cache_dirs", type=str, nargs='+', required=True, help="Path(s) to output folders for saving latent features (real and fake chunks).")
    
    # === Image and preprocessing options ===
    parser.add_argument("--data_size", type=int, nargs=2, default=(224, 224), help="Image size (height, width).")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size used for processing.")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Number of images per output file (chunk).")
    parser.add_argument("--augmentations", action="store_true", help="Enable data augmentation using Albumentations.")

    # === Latent extraction logic ===
    parser.add_argument(
        "--tracked_timesteps",
        type=json.loads,
        default="[981, 741, 521, 261, 1]",
        help="Timesteps at which to extract intermediate latents, passed as a JSON list string."
    )
    parser.add_argument(
        "--start",
        type=int,
        default=-1,
        help="Starting chunk index (in tens)."
    )
    parser.add_argument(
        "--end",
        type=int,
        default=-1,
        help="Ending chunk index (in tens)."
    )

    # === Prompt and conditioning ===
    parser.add_argument("--use_prompt_template", action="store_true", help="Use prompt template with class replacement like 'a photo of a [CLS]'.")
    parser.add_argument("--prompt", type=str, default="a photo", help="Base prompt string if not using prompt template.")
    parser.add_argument("--prompt_template", type=str, default="a photo of a [CLS]", help="Prompt template to use if --use_prompt_template is enabled.")

    # === Model and scheduler config ===
    parser.add_argument(
        "--model_id",
        type=str,
        default="stabilityai/stable-diffusion-2-1",
        choices=[
            "stabilityai/stable-diffusion-2-1",
            "stabilityai/stable-diffusion-v1-5"
        ],
        help="Model ID to use (either SD v1.5 or v2.1)"
    )
    parser.add_argument("--ddim", action="store_true", help="Use DDIM scheduler instead of default DDPM.")

    # === Device ===
    parser.add_argument("--device_id", default='0', help="Setting the GPU id, multi gpu split by ',', such as '0,1,2,3'", type=str)
    
    args = parser.parse_args()
    main_worker(args)