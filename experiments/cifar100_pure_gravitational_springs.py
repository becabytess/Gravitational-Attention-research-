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
# 2. GRAVITATIONAL CENTROID BLOCK WITH HOOKE SPRINGS
# ==========================================
class GravitationalCentroidBlock(nn.Module):
    """
    Pure Gravitational Attraction + Hooke Springs + Trajectory CNN:
      - 4 Orthogonal Subspaces (G=4), d_g = 48
      - K=5 dynamic centroids per subspace
      - Hooke Springs: F_spring = kappa * (u - x) anchors tokens to initial coordinates
      - Trajectory CNN over [Batch, Channels, Rounds, Tokens] captures velocity momentum
    """
    def __init__(self, dim=192, num_groups=4, num_centroids=5, num_rounds=4):
        super().__init__()
        self.dim = dim
        self.G = num_groups
        self.d_g = dim // num_groups # 48
        self.K = num_centroids # 5
        self.R = num_rounds

        self.round_norm = nn.LayerNorm(dim)
        
        # Base queries for the K=5 centroids in each group
        self.query_base = nn.Parameter(torch.randn(1, self.G, self.K, self.d_g) * 0.2)
        self.query_mod = nn.Linear(self.d_g, self.K * self.d_g)
        
        self.scale = nn.Parameter(torch.ones(1, self.G, 1, 1) * 8.0)
        self.step_size = nn.Parameter(torch.ones(self.G) * 0.5)
        
        # Trajectory CNN across (Rounds, Tokens)
        self.traj_cnn = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=(3, 3), padding=(1, 1), groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=(num_rounds, 3), padding=(0, 1), groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        self.traj_proj = nn.Linear(dim, dim)
        
        # Block FFN
        self.norm_final = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x, u, kappa, return_physics=False):
        B, N, D = x.shape
        G, K, d_g, R = self.G, self.K, self.d_g, self.R
        
        transitions = []
        cluster_maps = []
        
        for r in range(R):
            x_norm = self.round_norm(x)
            
            # 1. Subspace group mean modulates the K=5 centroid queries
            x_g = x_norm.view(B, N, G, d_g).transpose(1, 2) # (B, G, N, d_g)
            group_mean = x_g.mean(dim=2) # (B, G, d_g)
            q = F.normalize(self.query_base + self.query_mod(group_mean).view(B, G, K, d_g), dim=-1)
            
            # 2. Gather tokens into K=5 centroids
            k_tok = F.normalize(x_g, dim=-1)
            scale = torch.clamp(self.scale, min=1.0, max=25.0)
            gather_scores = scale * torch.matmul(q, k_tok.transpose(-1, -2)) # (B, G, K, N)
            centroids = torch.matmul(F.softmax(gather_scores, dim=-1), x_g) # (B, G, K, d_g)
            
            # 3. Pull tokens toward matching centroids (Pure Gravitation)
            pull_scores = scale * torch.matmul(k_tok, q.transpose(-1, -2)) # (B, G, N, K)
            pull_weights = F.softmax(pull_scores, dim=-1)
            
            if return_physics:
                cluster_maps.append(pull_weights.argmax(dim=-1)) # (B, G, N)
                
            pulled = torch.matmul(pull_weights, centroids) - x_g # (B, G, N, d_g)
            
            # 4. Pure gravitational displacement + Hooke Spring coordinate anchor
            eta = torch.sigmoid(self.step_size).view(1, G, 1, 1)
            drift_g = eta * pulled
            drift = drift_g.transpose(1, 2).contiguous().view(B, N, D)
            
            # Hooke spring anchoring tokens back to initial coordinate u
            spring_force = kappa * (u - x)
            delta_x = drift + spring_force
            x = x + delta_x
            transitions.append(delta_x)
            
        # 5. Trajectory CNN reads velocity momentum across all R rounds
        traj = torch.stack(transitions, dim=2).permute(0, 3, 2, 1).contiguous()
        traj_features = self.traj_cnn(traj).squeeze(2).transpose(1, 2)
        x = x + self.traj_proj(traj_features)
        
        # 6. Block FFN
        x = x + 0.25 * self.mlp(self.norm_final(x))
        
        if return_physics:
            return x, transitions, cluster_maps
        return x


# ==========================================
# 3. GRAVITATIONAL SPRINGS 2-STAGE VISION NETWORK
# ==========================================
class GravitationalSpringsVisionNet(nn.Module):
    """
    Two-Stage Hierarchical Gravitational Architecture WITH Hooke Springs:
      - Reuses the EXACT same block for Stage 1 -> Stage 2
      - Springs preserve token coordinate identity across multi-stage evolution
      - Parameter count: strictly ~369k parameters
    """
    def __init__(self, num_classes=100, dim=192, num_groups=4, num_centroids=5, num_rounds=4):
        super().__init__()
        self.dim = dim
        self.patch_stem = nn.Sequential(
            nn.Conv2d(3, dim // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim // 2),
            nn.GELU(),
            nn.Conv2d(dim // 2, dim, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(dim)
        )
        self.pos_emb = nn.Parameter(torch.randn(1, 256, dim) * 0.02)
        self.kappa_proj = nn.Linear(dim, num_groups)
        
        # One shared gravitational block with Hooke springs reused for both stages
        self.block = GravitationalCentroidBlock(dim, num_groups, num_centroids, num_rounds)
        
        self.head_norm = nn.LayerNorm(dim * 2)
        self.classifier = nn.Linear(dim * 2, num_classes)

    def forward(self, img, return_physics=False):
        B = img.size(0)
        u = self.patch_stem(img).flatten(2).transpose(1, 2) + self.pos_emb
        x = u.clone()
        kappa = torch.sigmoid(8.0 * self.kappa_proj(u)).repeat_interleave(self.block.d_g, dim=-1)
        
        if return_physics:
            # Stage 1: Local part clustering (anchored to u by springs)
            x, trans1, clusters1 = self.block(x, u, kappa, return_physics=True)
            # Stage 2: Global semantic object binding (reusing exact same block)
            x, trans2, clusters2 = self.block(x, u, kappa, return_physics=True)
            com = x.mean(dim=1)
            peak = x.max(dim=1).values
            logits = self.classifier(self.head_norm(torch.cat([com, peak], dim=-1)))
            return logits, trans1, trans2, clusters1, clusters2
        else:
            x = self.block(x, u, kappa) # Stage 1
            x = self.block(x, u, kappa) # Stage 2
            com = x.mean(dim=1)
            peak = x.max(dim=1).values
            return self.classifier(self.head_norm(torch.cat([com, peak], dim=-1)))


# ==========================================
# 4. TRAIN FOR 10 EPOCHS (PARALLEL BENCHMARK & VISUALIZATION)
# ==========================================
if __name__ == "__main__":
    model = GravitationalSpringsVisionNet(num_classes=100, dim=192, num_groups=4, num_centroids=5, num_rounds=4).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters : {total_params:,} | Gravitational 2-Stage WITH Springs (K=5 Centroids, R=4x2)")
    print(f"Physics Config   : Pure Gravitation + Hooke Springs | Reused Block (Stage 1 -> Stage 2)")
    print(f"Benchmark Target : 10 Epochs (~2.5 mins)\n")

    epochs = 10
    batch_size = 128
    val_batch_size = 500
    num_train = train_x_u8.size(0)
    num_val   = val_x_u8.size(0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3.5e-3, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))

    telemetry = {
        "experiment": "Gravitational 2-Stage WITH Hooke Springs (CIFAR-100, 10 Epochs, K=5 Centroids)",
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
    json_path = "results/pure_gravitational_springs_telemetry.json"
    with open(json_path, "w") as f:
        json.dump(telemetry, f, indent=2)
    print(f"Saved raw telemetry to {json_path}")

    # ==========================================
    # 5. DIAGNOSTIC STAGE 1 VS STAGE 2 VISUALIZATION
    # ==========================================
    print("\nGenerating Diagnostic Stage 1 vs Stage 2 Visualizations...")
    model.eval()
    num_samples = 4
    sample_u8 = val_x_u8[:num_samples]
    sample_imgs = preprocess_gpu_batch(sample_u8, is_train=False)
    sample_labels = val_y[:num_samples]

    with torch.no_grad():
        logits, trans1, trans2, clusters1, clusters2 = model(sample_imgs, return_physics=True)
        preds = logits.argmax(dim=-1)

    fig, axes = plt.subplots(num_samples, 7, figsize=(18, 2.8 * num_samples))
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

        # Col 1: Stage 1 Total Drift
        drift1 = sum(t[i].norm(dim=-1) for t in trans1).view(16, 16).cpu().numpy()
        axes[i, 1].imshow(orig_img)
        axes[i, 1].imshow(drift1, cmap="inferno", alpha=0.55, interpolation="bicubic", extent=[0, 32, 32, 0])
        axes[i, 1].set_title("Stage 1 Drift\n(Part Focus)", fontsize=9, fontweight='bold')
        axes[i, 1].axis("off")

        # Col 2: Stage 1 Centroid Clusters (Group 0)
        c1_map = clusters1[-1][i, 0].view(16, 16).cpu().numpy()
        axes[i, 2].imshow(c1_map, cmap="tab10", vmin=0, vmax=4)
        axes[i, 2].set_title("Stage 1 Clusters\n(K=5 Centroids)", fontsize=9, fontweight='bold')
        axes[i, 2].axis("off")

        # Col 3: Stage 2 Total Drift
        drift2 = sum(t[i].norm(dim=-1) for t in trans2).view(16, 16).cpu().numpy()
        axes[i, 3].imshow(orig_img)
        axes[i, 3].imshow(drift2, cmap="inferno", alpha=0.55, interpolation="bicubic", extent=[0, 32, 32, 0])
        axes[i, 3].set_title("Stage 2 Drift\n(Semantic Focus)", fontsize=9, fontweight='bold')
        axes[i, 3].axis("off")

        # Col 4: Stage 2 Centroid Clusters (Group 0)
        c2_map = clusters2[-1][i, 0].view(16, 16).cpu().numpy()
        axes[i, 4].imshow(c2_map, cmap="tab10", vmin=0, vmax=4)
        axes[i, 4].set_title("Stage 2 Clusters\n(Object Binding)", fontsize=9, fontweight='bold')
        axes[i, 4].axis("off")

        # Col 5: Stage 1 Group 1 Clusters
        c1_g1 = clusters1[-1][i, 1].view(16, 16).cpu().numpy()
        axes[i, 5].imshow(c1_g1, cmap="tab10", vmin=0, vmax=4)
        axes[i, 5].set_title("Stage 1 G1 Clusters", fontsize=8)
        axes[i, 5].axis("off")

        # Col 6: Stage 2 Group 1 Clusters
        c2_g1 = clusters2[-1][i, 1].view(16, 16).cpu().numpy()
        axes[i, 6].imshow(c2_g1, cmap="tab10", vmin=0, vmax=4)
        axes[i, 6].set_title("Stage 2 G1 Clusters", fontsize=8)
        axes[i, 6].axis("off")

    plt.suptitle("Gravitational 2-Stage Network WITH Hooke Springs (K=5 Centroids, 10 Epochs)", fontsize=13, fontweight='bold', y=0.99)
    plt.tight_layout()
    plot_path = "results/pure_gravitational_springs_10epochs.png"
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    print(f"Saved visualization to {plot_path}")
    plt.show()
