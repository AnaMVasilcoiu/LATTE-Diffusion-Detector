import json
import argparse
import importlib
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision import transforms
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, precision_recall_curve
from dataset import ChunkedIterableDataset
from sklearn.manifold import TSNE
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# JPEG compression
def apply_alb_jpeg(quality):
    return A.Compose([
        A.JpegCompression(quality=quality, p=1.0),
        A.Resize(224, 224),
        A.Normalize(mean=0.5, std=0.5),
        ToTensorV2()
    ])

# Crop
def apply_alb_crop(factor):
    crop_size = int(224 * factor)
    margin = (224 - crop_size) // 2
    return A.Compose([
        A.CenterCrop(height=crop_size, width=crop_size, p=1.0),
        A.Resize(224, 224),
        A.Normalize(mean=0.5, std=0.5),
        ToTensorV2()
    ])

# Gaussian blur
def apply_alb_blur(sigma):
    return A.Compose([
        A.GaussianBlur(blur_limit=(int(sigma), int(sigma)), sigma_limit=(sigma, sigma), p=1.0),
        A.Resize(224, 224),
        A.Normalize(mean=0.5, std=0.5),
        ToTensorV2()
    ])

# Gaussian noise
def apply_alb_noise(sigma):
    return A.Compose([
        A.GaussNoise(var_limit=(sigma * 255)**2, mean=0, p=1.0),
        A.Resize(224, 224),
        A.Normalize(mean=0.5, std=0.5),
        ToTensorV2()
    ])

def transform_image_with(img, alb_transform):
    img_np = np.array(img.convert("RGB"))
    augmented = alb_transform(image=img_np)
    return augmented['image']

def search_best_acc(gt_labels, pred_probs):
    best_acc, best_thresh = -1, -1
    for thresh in sorted(pred_probs):
        preds = (pred_probs > thresh).astype(int)
        acc = accuracy_score(gt_labels, preds)
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
    return best_acc, best_thresh

# Evaluate one perturbation config
def evaluate_perturbation(model, dataset, alb_transform_fn, level_values, device):
    results = []
    for val in level_values:
        model.eval()
        preds, labels = [], []
        transform_fn = alb_transform_fn(val)
        with torch.no_grad():
            for img, label, latent, loss_map in tqdm(dataset, desc=f"Evaluating val={val}"):
                img_tensor = transform_image_with(transforms.ToPILImage()(img), transform_fn).unsqueeze(0).to(device)
                label = torch.tensor([label]).to(device)
                latent = latent.unsqueeze(0).to(device)
                logits, _ = model(img_tensor, latent)
                prob = torch.softmax(logits, dim=-1)[:, 1]
                preds.append(prob.item())
                labels.append(label.item())
        ap = average_precision_score(np.array(labels), np.array(preds))
        best_acc, _ = search_best_acc(np.array(labels), np.array(preds))
        results.append((ap, best_acc))
    return results

def evaluate_all_perturbations(
    model, dataset, device, 
    jpeg_levels=[100, 90, 80, 70, 60, 50], 
    crop_levels=[1.0, 0.9, 0.8, 0.7, 0.6, 0.5], 
    blur_levels=[0, 1, 2, 3, 4, 5], 
    noise_levels=[0, 0.05, 0.1, 0.15, 0.2, 0.25]
):
    jpeg_results = evaluate_perturbation(model, dataset, apply_alb_jpeg, jpeg_levels, device)
    crop_results = evaluate_perturbation(model, dataset, apply_alb_crop, crop_levels, device)
    blur_results = evaluate_perturbation(model, dataset, apply_alb_blur, blur_levels, device)
    noise_results = evaluate_perturbation(model, dataset, apply_alb_noise, noise_levels, device)

    # Extract AP and Accuracy from results
    def unpack(results):
        ap, acc = zip(*results)
        return list(ap), list(acc)

    return {
        "jpeg": unpack(jpeg_results),
        "crop": unpack(crop_results),
        "blur": unpack(blur_results),
        "noise": unpack(noise_results)
    }

# Evaluate Conv-B model
def evaluate_convnext(model, dataset, alb_transform_fn, level_values, device):
    model.eval()
    results = []
    for val in level_values:
        preds, labels = [], []
        transform_fn = alb_transform_fn(val)
        with torch.no_grad():
            for img, label, _, _ in tqdm(dataset, desc=f"ConvNeXt val={val}"):
                img_tensor = transform_image_with(transforms.ToPILImage()(img), transform_fn).unsqueeze(0).to(device)
                logits = model(img_tensor)
                prob = torch.softmax(logits, dim=-1)[:, 1]
                preds.append(prob.item())
                labels.append(label.item())
        ap = average_precision_score(labels, preds)
        acc = accuracy_score(labels, [int(p > 0.5) for p in preds])
        results.append((ap, acc))
    return results

def main():
    conf = argparse.ArgumentParser()
    conf.add_argument('--checkpoint', type=str, required=True)
    conf.add_argument('--latent_dir', type=str, required=True)
    conf.add_argument("--exp_name", type=str, default="", help="Name of experiment.")
    conf.add_argument("--num_class", type=int, default=2, help="Number of classes in the dataset")
    conf.add_argument("--batch_size", type=int, default=32, help="Batch size for testing")
    conf.add_argument("--data_size", type=int, nargs=2, default=(224, 224), help="Image size (height, width)")
    conf.add_argument("--gpu", type=int, default=None, help="GPU device ID (if available)")
    conf.add_argument("--clip_type", type=str, default='RN50', help="CLIP model type")
    conf.add_argument("--model_type", type=str, required=True, help="Model type (e.g., LatentTrajectoryClassifier or CLipClassifierWMapV6)")
    conf.add_argument(
        "--tracked_timesteps",
        type=json.loads,
        default="[981, 741, 521, 261, 1]",
        help="Selected timesteps for training"
    )
    conf.add_argument('--one_t_index', type=int, default=-1, help='Chose one timestep from the saved 5.')
    conf.add_argument("--self_attention_latents", action='store_true', default=False)
    conf.add_argument("--weighted_average", action='store_true', default=False)
    conf.add_argument("--single_decoder", action='store_true', default=False)
    conf.add_argument("--process_latents_separately", action='store_true', default=False)
    conf.add_argument("--use_cls_token", action='store_true', default=False)
    conf.add_argument("--positional_embedding_type", type=str, default="none", help="Positional embedding type.")
    conf.add_argument("--plot_tsne", action='store_true', default=False)
    conf.add_argument("--return_clip_global_feats", action='store_true', default=False)
 
    args = conf.parse_args()

    # Device setup
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu is not None else "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if args.model_type == "LatentTrajectoryClassifier":
        model_latte = getattr(importlib.import_module('model'), args.model_type)(
            num_class=args.num_class, 
            clip_type=args.clip_type, 
            num_timesteps=len(args.tracked_timesteps), 
            num_heads=8, 
            self_attention_latents=args.self_attention_latents,
            weighted_average=args.weighted_average,
            single_decoder=args.single_decoder,
            process_latents_separately=args.process_latents_separately,
            use_cls_token=args.use_cls_token,
            positional_embedding_type=args.positional_embedding_type,
            return_clip_global_feats=args.return_clip_global_feats
        )
    elif args.model_type == "LatentTrajectoryClassifierSingleCLS":
        model_latte = getattr(importlib.import_module('model'), args.model_type)(
            num_class=args.num_class, 
            clip_type=args.clip_type, 
            num_timesteps=len(args.tracked_timesteps), 
            num_heads=8, 
        )

    model_latte = torch.nn.DataParallel(model_latte)
    model_latte.load_state_dict(checkpoint["model_state_dict"])
    model_latte.to(device)
    print(f"Loaded model from {args.checkpoint}")

    # Load dataset
    dataset = ChunkedIterableDataset(cache_dir=args.latent_dir, shuffle_chunks=False, shuffle_within_chunk=False)

    # Experiments
    jpeg_levels = [100, 90, 80, 70, 60, 50]
    crop_levels = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    blur_levels = [0, 1, 2, 3, 4, 5]
    noise_levels = [0, 0.05, 0.1, 0.15, 0.2, 0.25]

    # Evaluate LaTTE model
    results_latte = evaluate_all_perturbations(model_latte, dataset, device, jpeg_levels, crop_levels, blur_levels, noise_levels)

    # Load ConvNeXt-B model
    model_convb = timm.create_model("convnext_base_in22k", pretrained=True)
    model_convb.head = nn.Linear(model_convb.head.in_features, args.num_class)
    model_convb = torch.nn.DataParallel(model_convb)
    model_convb.to(device)

    results_convb = {
        "jpeg": list(zip(*evaluate_convnext(model_convb, dataset, apply_alb_jpeg, jpeg_levels, device))),
        "crop": list(zip(*evaluate_convnext(model_convb, dataset, apply_alb_crop, crop_levels, device))),
        "blur": list(zip(*evaluate_convnext(model_convb, dataset, apply_alb_blur, blur_levels, device))),
        "noise": list(zip(*evaluate_convnext(model_convb, dataset, apply_alb_noise, noise_levels, device)))
    }

    def plot_curve(ax, levels, results_a, results_b, title, xlabel):
        ap_a, acc_a = results_a
        ap_b, acc_b = results_b
        ax.plot(levels, ap_a, 'ro-', label='LaTTE AP')
        ax.plot(levels, acc_a, 'rs--', label='LaTTE Acc')
        ax.plot(levels, ap_b, 'bo-', label='Conv-B AP')
        ax.plot(levels, acc_b, 'bs--', label='Conv-B Acc')
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Score")
        ax.grid(True)
        ax.set_ylim(0.4, 1.0)
        ax.legend()
        if xlabel == "q":
            ax.invert_xaxis()

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    plot_curve(axs[0, 0], jpeg_levels, results_latte["jpeg"], results_convb["jpeg"], "(a) JPEG", "q")
    plot_curve(axs[0, 1], crop_levels, results_latte["crop"], results_convb["crop"], "(b) Crop", "f")
    plot_curve(axs[1, 0], blur_levels, results_latte["blur"], results_convb["blur"], "(c) Blur", "σ")
    plot_curve(axs[1, 1], noise_levels, results_latte["noise"], results_convb["noise"], "(d) Noise", "σ")

    plt.tight_layout()
    plt.savefig('perturbation_ap_acc_latte_vs_convb.png')
    print("Saved robustness plots as perturbation_ap_acc_latte_vs_convb.png")

if __name__ == '__main__':
    main()