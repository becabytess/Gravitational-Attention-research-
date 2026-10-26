import os
import gc
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset

# Ensure results directory exists
os.makedirs("results", exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on device: {device}")

# ==========================================
# 1. ULTRA-FAST GPU TENSOR PIPELINE FOR CIFAR-100
# ==========================================
print("Downloading uoft-cs/cifar100 from Hugging Face...")
hf_data = load_dataset("uoft-cs/cifar100")
classes = hf_data["train"].features["fine_label"].names

def pack_to_gpu(hf_split):
    imgs = np.stack([np.array(img, dtype=np.uint8) for img in hf_split["img"]])
    imgs_t = torch.from_numpy(imgs).permute(0, 3, 1, 2).to(device)
    labels_t = torch.tensor(hf_split["fine_label"], dtype=torch.long, device=device)
    return imgs_t, labels_t

print("Moving CIFAR-100 dataset directly into VRAM (150 MB)...")
train_x_u8, train_y = pack_to_gpu(hf_data["train"])
val_x_u8, val_y     = pack_to_gpu(hf_data["test"])
del hf_data
gc.collect()

cifar_mean = torch.tensor([0.5071, 0.4867, 0.4408], device=device).view(1, 3, 1, 1)
cifar_std  = torch.tensor([0.2675, 0.2565, 0.2761], device=device).view(1, 3, 1, 1)

def preprocess_gpu_batch(imgs_u8, is_train=True):
    x = imgs_u8.float().div_(255.0)
    x = (x - cifar_mean) / cifar_std
    if is_train:
        flip_mask = torch.rand(x.size(0), 1, 1, 1, device=device) < 0.5
        x = torch.where(flip_mask, x.flip(-1), x)
        x_pad = F.pad(x, (4, 4, 4, 4), mode='reflect')
        dy = torch.randint(0, 9, (1,)).item()
        dx = torch.randint(0, 9, (1,)).item()
        x = x_pad[:, :, dy:dy+32, dx:dx+32]
    return x


# ==========================================
# 2. PURE MULTI-CENTROID + TRAJECTORY CNN BLOCK
# ==========================================
class PureCentroidTrajectoryBlock(nn.Module):
    """
    Physical Multi-Centroid Dynamics + Trajectory CNN:
      - 4 Orthogonal Subspaces (G=4), each on S^47
      - K=5 dynamic centroids per subspace (no single-attractor tug-of-war)
      - Weight-tied rounds with Hooke springs to preserve input identity
      - 2D Trajectory CNN over [Batch, Channels, Rounds, Tokens] to integrate
        the full velocity & momentum history across the 6 physical rounds
    """
    def __init__(self, dim=192, num_groups=4, num_centroids=5, num_rounds=6):
        super().__init__()
        self.dim = dim
        self.G = num_groups
        self.d_g = dim // num_groups # 48
        self.K = num_centroids # 5
        self.R = num_rounds
        
        # Base queries for the K=5 attractors in each group: (1, G, K, d_g)
        self.query_base = nn.Parameter(torch.randn(1, self.G, self.K, self.d_g) * 0.2)
        # Lightweight modulation from group mean (48 -> 5*48 = 240)
        self.query_mod = nn.Linear(self.d_g, self.K * self.d_g)
        
        self.scale = nn.Parameter(torch.ones(1, self.G, 1, 1) * 8.0)
        self.eta_e = nn.Parameter(torch.full((self.G,), 0.40))
        
        # Trajectory CNN across (Rounds, Tokens)
        # Depthwise 2D convs keep parameters strictly fair (~5k params)
        self.traj_cnn = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=(3, 3), padding=(1, 1), groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=(num_rounds, 3), padding=(0, 1), groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        self.traj_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x, u, kappa, charge, return_physics=False):
        B, G, N, d_g = x.shape
        K, R = self.K, self.R
        
        transitions = []
        cluster_maps = []
        
        for r in range(R):
            # 1. Subspace group mean modulates the K=5 centroid queries
            group_mean = x.mean(dim=2) # (B, G, d_g)
            q_mod = self.query_mod(group_mean).view(B, G, K, d_g)
            q = F.normalize(self.query_base + q_mod, dim=-1) # (B, G, K, d_g)
            
            # 2. Gather N tokens into K=5 centroids on S^(d_g-1)
            scale = torch.clamp(self.scale, min=1.0, max=25.0)
            gather_scores = scale * torch.matmul(q, x.transpose(-1, -2)) # (B, G, K, N)
            centroid_weights = F.softmax(gather_scores, dim=-1)
            centroids = F.normalize(torch.matmul(centroid_weights, x), dim=-1) # (B, G, K, d_g)
            
            # 3. Pull tokens toward their closest matching centroid
            pull_scores = scale * torch.matmul(x, q.transpose(-1, -2)) # (B, G, N, K)
            pull_weights = F.softmax(pull_scores, dim=-1)
            
            if return_physics:
                cluster_maps.append(pull_weights.argmax(dim=-1)) # (B, G, N)
                
            assigned_centroid = torch.matmul(pull_weights, centroids) # (B, G, N, d_g)
            
            # 4. Coulomb Force + Hooke Spring Force
            # Coulomb pull toward centroid modulated by signed charge
            force_coulomb = charge * (assigned_centroid - x)
            force_spring  = kappa * (u - x)
            
            eta = self.eta_e.view(1, G, 1, 1)
            delta_x = eta * force_coulomb + force_spring
            x_moved = F.normalize(x + delta_x, dim=-1)
            
            # Record velocity transition vector in flat space
            delta_flat = (x_moved - x).permute(0, 2, 1, 3).reshape(B, N, self.dim)
            transitions.append(delta_flat)
            x = x_moved
            
        # 5. Trajectory CNN across velocity path: (B, D, R, N)
        traj = torch.stack(transitions, dim=2).permute(0, 3, 2, 1).contiguous()
        traj_features = self.traj_cnn(traj).squeeze(2).transpose(1, 2) # (B, N, D)
        
        # 6. Cross-subspace residual MLP mixing
        x_flat = x.permute(0, 2, 1, 3).reshape(B, N, self.dim)
        x_out_flat = x_flat + self.traj_proj(traj_features)
        x_mlp = x_out_flat + 0.25 * self.mlp(self.norm(x_out_flat))
        x_out = F.normalize(x_mlp.view(B, N, G, d_g).permute(0, 2, 1, 3), dim=-1)
        
        if return_physics:
            return x_out, transitions, cluster_maps
        return x_out


# ==========================================
# 3. FULL CENTROID-TRAJECTORY VISION NETWORK
# ==========================================
class CentroidTrajectoryVisionNet(nn.Module):
    def __init__(self, num_classes=100, dim=192, num_groups=4, num_centroids=5, num_rounds=6):
        super().__init__()
        self.dim = dim
        self.G = num_groups
        self.d_g = dim // num_groups
        self.patch_stem = nn.Sequential(
            nn.Conv2d(3, dim // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim // 2),
            nn.GELU(),
            nn.Conv2d(dim // 2, dim, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(dim)
        )
        self.pos_emb = nn.Parameter(torch.randn(1, 256, dim) * 0.02)
        self.kappa_proj = nn.Linear(dim, num_groups)
        self.charge_proj = nn.Linear(dim, num_groups)
        
        self.block = PureCentroidTrajectoryBlock(dim, num_groups, num_centroids, num_rounds)
        self.head_norm = nn.LayerNorm(dim * 2)
        self.classifier = nn.Linear(dim * 2, num_classes)

    def forward(self, img, return_physics=False):
        B = img.size(0)
        raw = self.patch_stem(img).flatten(2).transpose(1, 2) + self.pos_emb
        u = F.normalize(raw.view(B, 256, self.G, self.d_g).permute(0, 2, 1, 3), dim=-1)
        x = u.clone()
        
        raw_flat = raw
        kappa = torch.sigmoid(8.0 * self.kappa_proj(raw_flat)).permute(0, 2, 1).unsqueeze(-1)
        charge = torch.tanh(4.0 * self.charge_proj(raw_flat)).permute(0, 2, 1).unsqueeze(-1)
        
        if return_physics:
            x_final, trans, clusters = self.block(x, u, kappa, charge, return_physics=True)
        else:
            x_final = self.block(x, u, kappa, charge, return_physics=False)
            
        x_flat = x_final.permute(0, 2, 1, 3).reshape(B, 256, self.dim)
        com = F.normalize(x_flat.mean(dim=1), dim=-1)
        peak = x_flat.max(dim=1).values
        logits = self.classifier(self.head_norm(torch.cat([com, peak], dim=-1)))
        
        if return_physics:
            return logits, trans, clusters
        return logits


# ==========================================
# 4. TRAIN FOR 10 EPOCHS (RAPID BENCHMARK & VISUALIZATION)
# ==========================================
if __name__ == "__main__":
    model = CentroidTrajectoryVisionNet(num_classes=100, dim=192, num_groups=4, num_centroids=5, num_rounds=6).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters : {total_params:,} | Centroid-Trajectory Hybrid (4 Spheres S^47, K=5 Centroids)")
    print(f"Benchmark Target : Beat Baseline at Epoch 10 | Fast 10-Epoch Run (~2.5 mins)\n")

    epochs = 10
    batch_size = 128
    val_batch_size = 500
    num_train = train_x_u8.size(0)
    num_val   = val_x_u8.size(0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3.5e-3, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))

    telemetry = {
        "experiment": "Centroid Trajectory Coulomb Network (CIFAR-100, 10 Epochs, K=5 Centroids)",
        "parameters": total_params,
        "epochs": epochs,
        "epoch_metrics": []
    }

    t_start = time.time()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        perm = torch.randperm(num_train, device=device)
        train_loss, num_batches = 0.0, 0

        for i in range(0, num_train, batch_size):
            idx = perm[i:i+batch_size]
            imgs = preprocess_gpu_batch(train_x_u8[idx], is_train=True)
            labels = train_y[idx]

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                logits = model(imgs)
                loss = F.cross_entropy(logits, labels, label_smoothing=0.1)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            num_batches += 1

        scheduler.step()

        # Validation every epoch
        model.eval()
        correct_1, correct_5 = 0, 0
        v_loss = 0.0
        with torch.no_grad():
            for i in range(0, num_val, val_batch_size):
                imgs = preprocess_gpu_batch(val_x_u8[i:i+val_batch_size], is_train=False)
                labels = val_y[i:i+val_batch_size]
                with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                    logits = model(imgs)
                    v_loss += F.cross_entropy(logits, labels).item()
                _, pred = logits.topk(5, dim=-1, largest=True, sorted=True)
                correct = pred.eq(labels.view(-1, 1).expand_as(pred))
                correct_1 += correct[:, :1].sum().item()
                correct_5 += correct[:, :5].sum().item()

        top1 = 100.0 * correct_1 / num_val
        top5 = 100.0 * correct_5 / num_val
        current_lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        telemetry["epoch_metrics"].append({
            "epoch": epoch,
            "train_loss": round(train_loss / num_batches, 3),
            "val_loss": round(v_loss / (num_val / val_batch_size), 3),
            "top1_acc": round(top1, 2),
            "top5_acc": round(top5, 2),
            "lr": round(current_lr, 6),
            "elapsed_s": round(elapsed, 1)
        })

        print(f"Epoch {epoch:02d}/10 ({elapsed:.1f}s) | Loss: {train_loss/num_batches:.3f} | "
              f"Val Loss: {v_loss/(num_val/val_batch_size):.3f} | Top-1: {top1:.2f}% | Top-5: {top5:.2f}% | LR: {current_lr:.6f}")

    total_time = time.time() - t_start
    print(f"\nTraining completed in {total_time:.1f}s!")

    # Save JSON telemetry
    json_path = "results/centroid_trajectory_cifar100_telemetry.json"
    with open(json_path, "w") as f:
        json.dump(telemetry, f, indent=2)
    print(f"Saved raw telemetry to {json_path}")

    # ==========================================
    # 5. DIAGNOSTIC TRAJECTORY & CENTROID VISUALIZATION
    # ==========================================
    print("\nGenerating Diagnostic Trajectory & Centroid Visualizations...")
    model.eval()
    num_samples = 4
    sample_u8 = val_x_u8[:num_samples]
    sample_imgs = preprocess_gpu_batch(sample_u8, is_train=False)
    sample_labels = val_y[:num_samples]

    with torch.no_grad():
        logits, transitions, cluster_maps = model(sample_imgs, return_physics=True)
        preds = logits.argmax(dim=-1)

    # Setup 9-column figure matching the Gemini App inspection
    fig, axes = plt.subplots(num_samples, 9, figsize=(22, 3.0 * num_samples))
    plt.subplots_adjust(wspace=0.3, hspace=0.35)

    for i in range(num_samples):
        orig_img = sample_u8[i].cpu().permute(1, 2, 0).numpy() / 255.0
        true_cls = classes[sample_labels[i].item()]
        pred_cls = classes[preds[i].item()]
        title_color = "green" if sample_labels[i] == preds[i] else "red"

        # Col 0: Original Image
        axes[i, 0].imshow(orig_img)
        axes[i, 0].set_title(f"True: {true_cls}\nPred: {pred_cls}", color=title_color, fontsize=9, fontweight='bold')
        axes[i, 0].axis("off")

        # Col 1: Total Trajectory Focus (Cumulative velocity drift across all 6 rounds)
        # transitions is list of 6 tensors of shape (B, N, D)
        total_drift = sum(trans[i].norm(dim=-1) for trans in transitions).view(16, 16).cpu().numpy()
        axes[i, 1].imshow(orig_img)
        axes[i, 1].imshow(total_drift, cmap="inferno", alpha=0.55, interpolation="bicubic", extent=[0, 32, 32, 0])
        axes[i, 1].set_title("Total Trajectory\n(Attn Focus)", fontsize=9, fontweight='bold')
        axes[i, 1].axis("off")

        # Cols 2 to 7: Per-Round Velocity Shifts (Round 1 to 6)
        for r in range(6):
            r_shift = transitions[r][i].norm(dim=-1).view(16, 16).cpu().numpy()
            axes[i, 2 + r].imshow(r_shift, cmap="magma")
            axes[i, 2 + r].set_title(f"Round {r+1}\nShift", fontsize=8)
            axes[i, 2 + r].axis("off")

        # Col 8: Centroid Cluster Map (Final Round, Group 0, K=5 attractors)
        # cluster_maps[-1] has shape (B, G, N)
        cluster_map = cluster_maps[-1][i, 0].view(16, 16).cpu().numpy()
        axes[i, 8].imshow(cluster_map, cmap="tab10", vmin=0, vmax=4)
        axes[i, 8].set_title("Centroid Clusters\n(Group 0, K=5)", fontsize=9, fontweight='bold')
        axes[i, 8].axis("off")

    plt.suptitle("Centroid-Trajectory Coulomb Network (4 Spheres S^47, K=5 Centroids, 6 Rounds, 10 Epochs)", fontsize=13, fontweight='bold', y=0.99)
    plt.tight_layout()
    plot_path = "results/centroid_trajectory_cifar100_10epochs.png"
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    print(f"Saved visualization to {plot_path}")
    plt.show()
