import json
import os
import argparse
import importlib
import numpy as np
import logging as logger
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset import ChunkedIterableDataset
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, precision_recall_curve


logger.basicConfig(level=logger.INFO,
                   format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')

# Utility methods
def build_method_map(method_names):
    """
    Assigns a unique class label to each fake generation method. Real is always 0.
    Returns a dictionary like {'ADM': 1, 'BigGAN': 2, ...}.
    """
    unique_methods = sorted(set(method_names))
    return {m: i + 1 for i, m in enumerate(unique_methods)}

def get_device(gpu_id=None):
    """Returns the appropriate torch.device."""
    if torch.cuda.is_available():
        return torch.device(f'cuda:{gpu_id}' if gpu_id is not None else "cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def plot_tsne(embeddings, labels, title, save_path=None, real_fake_labels=False):
    """
    Plots a 2D t-SNE visualization with multiple classes (0=real, others=fake_X).
    """
    print(f"Running t-SNE on {embeddings.shape[0]} embeddings...")
    tsne = TSNE(n_components=2, random_state=42)
    emb_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    unique_labels = np.unique(labels)
    for lab in unique_labels:
        idx = (labels == lab)
        if real_fake_labels:
            label = "Real" if lab == 0 else "Fake"
        else:
            label = f"Class {lab}"
        plt.scatter(emb_2d[idx, 0], emb_2d[idx, 1], label=label, alpha=0.7)

    plt.title(title)
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.legend()
    if save_path:
        plt.savefig(save_path)
        print(f"t-SNE plot saved to {save_path}")
    else:
        plt.show()

def search_best_acc(gt_labels, pred_probs):
    """Finds the best threshold that maximizes accuracy."""
    best_acc = -1
    best_threshold = -1
    
    for thresh in sorted(pred_probs.tolist()):
        pred_probs_copy = np.array(pred_probs)
        pred_probs_copy[pred_probs_copy > thresh] = 1
        pred_probs_copy[pred_probs_copy <= thresh] = 0
        acc = accuracy_score(gt_labels, pred_probs_copy)
        
        if acc > best_acc:
            best_acc = acc
            best_threshold = thresh
    
    return best_acc, best_threshold

def evaluate_model(data_loader, model, model_type, args, device, dataset_name, method_name, exp_name, method_map):
    """Evaluates the model on a test dataset."""
    print(f'Start Testing on {dataset_name} (Method: {method_name})')
    model.eval()

    gt_labels_list, prob_labels_list = [], []
    emb_list, clip_global_list, plot_labels_list = [], [], []

    with torch.no_grad():
        for iter, (images, labels, latent_seqs, loss_maps) in enumerate(data_loader):
            images = images.to(device)
            labels = labels.flatten().squeeze().to(device)
            features = loss_maps.to(device) if args.lare else latent_seqs.to(device)

            if args.return_global_feats:
                logits, embeddings, clip_global = model(images, features)
                clip_global_list.append(clip_global)
            else:
                if args.lare:
                    logits = model(images, features)
                    embeddings = None
                elif args.latent_only:
                    logits, embeddings = model(features)
                else:
                    logits, embeddings = model(images, features)

            prob = torch.softmax(logits, dim=1)  # Convert logits to probabilities

            gt_labels_list.append(labels)
            prob_labels_list.append(prob[:, 1])  # Keep probability for class 1 (fake)

            if not args.lare and embeddings is not None:
                emb_list.append(embeddings)

            # Build multi-class plot labels: real => 0, fake => method_name
            labels_np = labels.cpu().numpy()
            plot_label_batch = np.array([0 if lbl == 0 else method_map[method_name] for lbl in labels_np], dtype=np.int32)
            plot_labels_list.append(plot_label_batch)

            if iter % 50 == 0:
                print(f'Testing Progress on {dataset_name}: {iter}/{len(data_loader)}')

    # Concatenate results from all batches
    gt_labels = torch.cat(gt_labels_list, dim=0).cpu().numpy()
    prob_labels = torch.cat(prob_labels_list, dim=0).cpu().numpy()
    embeddings_all = torch.cat(emb_list, dim=0).cpu().numpy() if not args.lare and emb_list else np.zeros((len(gt_labels), 1))
    clip_global_all = torch.cat(clip_global_list, dim=0).cpu().numpy() if args.return_global_feats else np.zeros((len(gt_labels), 1))
    plot_labels = np.concatenate(plot_labels_list, axis=0)

    # Compute AUC, Accuracy, and other metrics
    auc = roc_auc_score(gt_labels, prob_labels)
    precision, recall, thresholds = precision_recall_curve(gt_labels, prob_labels)
    ap = average_precision_score(gt_labels, prob_labels)

    # Method 1: Find best threshold using f_score from Precision-Recall Curve
    f_score = (2 * precision * recall) / (precision + recall + 1e-10)  # Avoid division by zero
    best_f_score_idx = np.argmax(f_score)
    best_threshold_f_score = thresholds[best_f_score_idx]

    # Apply thresholding (f_score method)
    pred_labels_f_score = (prob_labels > best_threshold_f_score).astype(int)
    acc_f_score = accuracy_score(gt_labels, pred_labels_f_score)

    # Method 2: Find best threshold using search_best_acc()
    best_acc_search, best_threshold_search = search_best_acc(gt_labels, prob_labels)

    print(f'[{dataset_name}] ({method_name}) Test AUC: {auc:.4f}')
    print(f'[{dataset_name}] ({method_name}) Best Accuracy (f_score): {acc_f_score}')
    print(f'[{dataset_name}] ({method_name}) Best Accuracy (search_best_acc): {best_acc_search}')
    print(f'[{dataset_name}] ({method_name}) Test AP: {ap:.4f}')

    return {
        "dataset": dataset_name,
        "method": method_name,
        "AUC": auc,
        "Best Accuracy (f_score)": acc_f_score,
        "Threshold (f_score)": best_threshold_f_score,
        "Best Accuracy (search_best_acc)": best_acc_search,
        "Threshold (search_best_acc)": best_threshold_search,
        "AP": ap,
        "gt_labels": gt_labels,
        "pred_probs": prob_labels,
        "embeddings_all": embeddings_all,
        "clip_global_all": clip_global_all,
        "plot_labels": plot_labels
    }


if __name__ == '__main__':
    conf = argparse.ArgumentParser(description="Evaluate pretrained model on chunked latent datasets with optional t-SNE visualization.")

    # === Dataset & Input Configuration ===
    conf.add_argument(
        "--latent_dirs_test", type=str, nargs='+', required=True,
        help="List of paths to folders containing test .pt chunk files."
    )
    conf.add_argument(
        "--method_names", type=str, nargs='+', required=True,
        help="List of method names (e.g., 'ADM', 'BigGAN') corresponding to each test folder."
    )

    # === Model Configuration ===
    conf.add_argument("--model_type", type=str, required=True, help="Model class name to instantiate (e.g., 'LatentTrajectoryClassifier', 'CLipClassifierWMapV6').")
    conf.add_argument("--clip_type", type=str, default='RN50', help="CLIP backbone type to use (e.g., 'RN50', 'ViT-L/14').")
    conf.add_argument("--checkpoint", type=str, required=True, help="Path to the pretrained model checkpoint (.pth file).")
    conf.add_argument("--num_class", type=int, default=2, help="Number of output classes (default: 2 for binary classification).")
    conf.add_argument("--gpu", type=int, default=None, help="ID of the GPU to use (if available). If not set, uses the default CUDA device.")
    
    # === Data Loading & Preprocessing ===
    conf.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation.")
    conf.add_argument("--data_size", type=int, nargs=2, default=(224, 224), help="Image resolution as (height, width).")
    conf.add_argument("--imagenet_norm", action='store_true', default=False, help="Use ImageNet normalization instead of default 0.5 mean/std.")

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

    # === Model Behavior Flags ===
    conf.add_argument("--latent_only", action='store_true', help="Use latent-only model without image input.")
    conf.add_argument("--self_attention_latents", action='store_true', help="Enable self-attention over latent sequences.")
    conf.add_argument("--weighted_average", action='store_true', help="Use learned weighted averaging across timesteps.")
    conf.add_argument("--single_decoder", action='store_true', help="Use a single shared decoder for all timesteps.")
    conf.add_argument("--process_latents_separately", action='store_true', help="Process each latent timestep separately.")
    conf.add_argument("--decoders_per_timestep", type=int, default=1, help="Number of decoders per timestep (default: 1).")
    conf.add_argument("--use_cls_token", action='store_true', help="Use a class token in the transformer model.")
    conf.add_argument("--positional_embeddings", action='store_true', help="Enable positional embeddings in transformer.")
    conf.add_argument("--xavier_kaiming_init", action='store_true', help="Use Xavier/Kaiming initialization in model.")
    conf.add_argument("--no_latent_visual_refinement", action='store_true', help="Disable visual refinement of latents.")
    conf.add_argument("--return_global_feats", action='store_true', help="Return global CLIP features from model.")
    conf.add_argument("--qkNorm", action='store_true', help="Apply QK-normalization in transformer layers.")

    # === Logging & Visualization ===
    conf.add_argument("--exp_name", type=str, default="", help="Optional experiment name for saving results and plots.")
    conf.add_argument("--plot_tsne", action='store_true', help="Generate t-SNE plots of refined/CLIP embeddings.")
    conf.add_argument("--real_fake_labels", action='store_true', help="Use binary real/fake labels in t-SNE plots.")

    args = conf.parse_args()

    # Ensure the number of test folders matches the number of method names
    if len(args.latent_dirs_test) != len(args.method_names):
        raise ValueError("The number of test folders must match the number of method names.")
    
    method_map = build_method_map(args.method_names)
    device = get_device(args.gpu)
 
    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
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
                return_clip_global_feats=args.return_global_feats,
                latent_visual_refinement=not args.no_latent_visual_refinement,
                qkNorm=args.qkNorm
            )
        elif args.model_type == "LatentTrajectoryClassifierSingleCLS":
            model = getattr(importlib.import_module('model'), args.model_type)(
                num_class=args.num_class, 
                clip_type=args.clip_type, 
                num_timesteps=len(args.selected_t_indices), 
                num_heads=8, 
                positional_embeddings=args.positional_embeddings,
                return_clip_global_feats=args.return_global_feats
            )
        elif args.model_type == "CLipClassifierWMapV6":
            model = getattr(importlib.import_module('model_lare'), args.model_type)(
                num_class=args.num_class, 
                clip_type=args.clip_type
            )

    model = torch.nn.DataParallel(model, device_ids=[0])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    print(f"Loaded model from {args.checkpoint}")

    # Image preprocessing
    mean_std = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) if args.imagenet_norm else ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=mean_std[0], std=mean_std[1]),
        ToTensorV2()
    ])

    results = []
    all_refined, all_clip, all_plot_labels = [], [], []

    # Run testing for each dataset and corresponding method name
    for test_folder, method_name in zip(args.latent_dirs_test, args.method_names):
        dataset_name = os.path.basename(test_folder)

        # Load test dataset
        test_dataset = ChunkedIterableDataset(
            cache_dir=test_folder,
            transform=transform,
            shuffle_chunks=False,
            shuffle_within_chunk=False,
            selected_t_indices=args.selected_t_indices,
        )

        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        print(f'Test dataset ({dataset_name}) size: {len(test_dataset)}')

        # Run evaluation
        result = evaluate_model(test_loader, model, args.model_type, args, device, dataset_name, method_name, args.exp_name, method_map)
        results.append(result)

        all_refined.append(result["embeddings_all"])
        all_clip.append(result["clip_global_all"])
        all_plot_labels.append(result["plot_labels"])

        if args.plot_tsne:
            os.makedirs("tsne_plots", exist_ok=True)

            label_subset_idx = np.where((result["plot_labels"] == 0) | (result["plot_labels"] == method_map[method_name]))[0]
            refined_subset = result["embeddings_all"][label_subset_idx]
            clip_subset = result["clip_global_all"][label_subset_idx]
            label_subset = result["plot_labels"][label_subset_idx]

            plot_tsne(
                embeddings=refined_subset,
                labels=label_subset,
                title=f"t-SNE: Real vs {method_name}",
                save_path=f"tsne_plots/{args.exp_name}_tsne_real_vs_{method_name}.png",
                real_fake_labels=args.real_fake_labels
            )

            if args.return_global_feats:
                plot_tsne(
                    embeddings=clip_subset,
                    labels=label_subset,
                    title=f"t-SNE: Real vs {method_name}",
                    save_path=f"tsne_plots/{args.exp_name}_tsne_real_vs_{method_name}.png",
                    real_fake_labels=args.real_fake_labels
                )

    print("Final Test Results:")
    for res in results:
        print(f"[{res['dataset']}] Method: {res['method']} -> AUC: {res['AUC']:.4f}, "
                    f"Best Acc (f_score): {res['Best Accuracy (f_score)']:.4f}, "
                    f"Best Acc (search_best_acc): {res['Best Accuracy (search_best_acc)']:.4f}, "
                    f"AP: {res['AP']:.4f}")

    # Plot t-SNE across all subsets
    refined_all = np.concatenate(all_refined, axis=0)
    clip_all = np.concatenate(all_clip, axis=0)
    label_all = np.concatenate(all_plot_labels, axis=0)