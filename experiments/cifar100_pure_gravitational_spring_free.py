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

def augment_gpu_batch(imgs_u8, is_train=True):
    # Keep images in [0, 1] for direct pixel space dynamics
    x = imgs_u8.float().div_(255.0)
    if is_train:
        flip_mask = torch.rand(x.size(0), 1, 1, 1, device=device) < 0.5
        x = torch.where(flip_mask, x.flip(-1), x)
        x_pad = F.pad(x, (4, 4, 4, 4), mode='reflect')
        dy = torch.randint(0, 9, (1,)).item()
        dx = torch.randint(0, 9, (1,)).item()
        x = x_pad[:, :, dy:dy+32, dx:dx+32]
    return x

def img_to_patches(img):
    # (B, 3, 32, 32) -> (B, 256, 12)
    B = img.size(0)
    return img.view(B, 3, 16, 2, 16, 2).permute(0, 2, 4, 1, 3, 5).reshape(B, 256, 12)

def patches_to_img(patches):
    # (B, 256, 12) -> (B, 3, 32, 32)
    B = patches.size(0)
    return patches.view(B, 16, 16, 3, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(B, 3, 32, 32)


# ==========================================
# 2. PURE SPRING-FREE GRAVITATIONAL ATTENTION BLOCK
# ==========================================
class PureSpringFreeGravitationalBlock(nn.Module):
    """
    Pure Gravitational Dynamics — ZERO SPRINGS, ZERO CHARGES:
      - Tokens and pixels move completely unconstrained
      - Semantics in 4 Orthogonal Spheres S^47 compute affinities & centroid routing
      - Centroid Prototype Colors: C_k = WeightedAverage(Pixels in centroid k)
      - Pixels blend toward their centroid color via natural step-size absorption:
          P <- (1 - eta) * P + eta * P_target
      - Semantic particles move along gravitational displacement:
          x <- Normalize(x + eta * (C_assigned - x) + pixel_feedback)
      - 2D Trajectory CNN across (Rounds, Tokens) captures the full relaxation path!
    """
    def __init__(self, dim=192, num_groups=4, num_centroids=5, num_rounds=6):
        super().__init__()
        self.dim = dim
        self.G = num_groups
        self.d_g = dim // num_groups # 48
        self.K = num_centroids # 5
        self.R = num_rounds

        # Base attractor queries per group: (1, G, K, d_g)
        self.query_base = nn.Parameter(torch.randn(1, self.G, self.K, self.d_g) * 0.2)
        self.query_mod  = nn.Linear(self.d_g, self.K * self.d_g)
        self.scale = nn.Parameter(torch.ones(1, self.G, 1, 1) * 8.0)

        # Learnable physical step rates (initialized to stable values)
        self.eta_pixel = nn.Parameter(torch.tensor(0.35))
        self.eta_sem   = nn.Parameter(torch.full((self.G,), 0.35))

        # Learned Token Mass prior (saliency gate)
        self.token_mass = nn.Linear(dim, self.G)
        nn.init.zeros_(self.token_mass.weight)
        nn.init.constant_(self.token_mass.bias, 1.5)

        # Pixel feedback projection: 12-dim patch RGB -> dim
        self.pixel_encoder = nn.Linear(12, dim)

        # Trajectory CNN across velocity path
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

    def forward(self, x, P_0, return_telemetry=False):
        B, G, N, d_g = x.shape
        P = P_0.clone()
        transitions = []
        image_history = [patches_to_img(P_0)]
        cluster_maps = []

        for r in range(self.R):
            # 1. Modulate centroid queries by group mean
            group_mean = x.mean(dim=2) # (B, G, d_g)
            q_mod = self.query_mod(group_mean).view(B, G, self.K, d_g)
            q = F.normalize(self.query_base + q_mod, dim=-1)

            # 2. Compute Learned Token Mass prior (saliency gate)
            x_flat_r = x.permute(0, 2, 1, 3).reshape(B, N, self.dim)
            mass_logits = self.token_mass(x_flat_r)
            mass = F.logsigmoid(mass_logits).permute(0, 2, 1).unsqueeze(2)

            # 3. Gather Step (Semantic + Pixel Space)
            scale = torch.clamp(self.scale, min=1.0, max=25.0)
            gather_scores = scale * torch.matmul(q, x.transpose(-1, -2)) + mass # (B, G, K, N)
            A = F.softmax(gather_scores, dim=-1)

            # Semantic Centroids: (B, G, K, d_g)
            C_sem = F.normalize(torch.matmul(A, x), dim=-1)

            # Centroid RGB Appearance: (B, G, K, 12)
            P_exp = P.unsqueeze(1).expand(-1, self.G, -1, -1)
            C_rgb = torch.matmul(A, P_exp)

            # 4. Pull Step
            pull_scores = scale * torch.matmul(x, q.transpose(-1, -2)) # (B, G, N, K)
            W = F.softmax(pull_scores, dim=-1)

            if return_telemetry:
                cluster_maps.append(W.argmax(dim=-1))

            assigned_C_sem = torch.matmul(W, C_sem) # (B, G, N, d_g)
            assigned_P_rgb = torch.matmul(W, C_rgb).mean(dim=1) # (B, N, 12)

            # 5. Pure Spring-Free Pixel Migration:
            # Pixels absorb their centroid color directly without artificial springs
            eta_p = torch.sigmoid(self.eta_pixel)
            P = torch.clamp((1.0 - eta_p) * P + eta_p * assigned_P_rgb, 0.0, 1.0)
            image_history.append(patches_to_img(P))

            # 6. Pure Spring-Free Semantic Migration:
            eta_s = torch.sigmoid(self.eta_sem).view(1, G, 1, 1)
            delta_sem = eta_s * (assigned_C_sem - x)
            # Differentiable pixel feedback:
            pix_delta = self.pixel_encoder(P - P_0).view(B, N, G, d_g).permute(0, 2, 1, 3)
            x_moved = F.normalize(x + delta_sem + 0.15 * pix_delta, dim=-1)

            delta_flat = (x_moved - x).permute(0, 2, 1, 3).reshape(B, N, self.dim)
            transitions.append(delta_flat)
            x = x_moved

        # 7. Trajectory CNN across velocity transitions
        traj = torch.stack(transitions, dim=2).permute(0, 3, 2, 1).contiguous()
        traj_features = self.traj_cnn(traj).squeeze(2).transpose(1, 2)
        x_flat = x.permute(0, 2, 1, 3).reshape(B, N, self.dim)
        x_out_flat = x_flat + self.traj_proj(traj_features)
        x_mlp = x_out_flat + 0.25 * self.mlp(self.norm(x_out_flat))
        x_out = F.normalize(x_mlp.view(B, N, G, d_g).permute(0, 2, 1, 3), dim=-1)

        if return_telemetry:
            return x_out, image_history, cluster_maps
        return x_out


# ==========================================
# 3. FULL VISION NETWORK (NO SPRINGS)
# ==========================================
class PureSpringFreeVisionNet(nn.Module):
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

        self.block = PureSpringFreeGravitationalBlock(dim, num_groups, num_centroids, num_rounds)
        self.head_norm = nn.LayerNorm(dim * 2)
        self.classifier = nn.Linear(dim * 2, num_classes)

    def forward(self, img, return_telemetry=False):
        B = img.size(0)
        P_0 = img_to_patches(img)

        raw = self.patch_stem(img).flatten(2).transpose(1, 2) + self.pos_emb
        u = F.normalize(raw.view(B, 256, self.G, self.d_g).permute(0, 2, 1, 3), dim=-1)

        if return_telemetry:
            x_final, img_hist, clusters = self.block(u.clone(), P_0, return_telemetry=True)
            x_flat = x_final.permute(0, 2, 1, 3).reshape(B, 256, self.dim)
            com = F.normalize(x_flat.mean(dim=1), dim=-1)
            peak = x_flat.max(dim=1).values
            logits = self.classifier(self.head_norm(torch.cat([com, peak], dim=-1)))
            return logits, img_hist, clusters

        x_final = self.block(u.clone(), P_0, return_telemetry=False)
        x_flat = x_final.permute(0, 2, 1, 3).reshape(B, 256, self.dim)
        com = F.normalize(x_flat.mean(dim=1), dim=-1)
        peak = x_flat.max(dim=1).values
        return self.classifier(self.head_norm(torch.cat([com, peak], dim=-1)))


# ==========================================
# 4. TRAIN FOR 10 EPOCHS & VISUALIZE SPRING-FREE CANVASES
# ==========================================
if __name__ == "__main__":
    model = PureSpringFreeVisionNet(num_classes=100, dim=192, num_groups=4, num_centroids=5, num_rounds=6).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters : {total_params:,} | Pure Spring-Free Gravitational Attention (4 Spheres S^47, K=5 Centroids)")
    print(f"Physics Config   : NO Springs, NO Anchors, NO Charges | Pure Centroid Drift | Epochs: 10 (~2.5 mins)\n")

    epochs = 10
    batch_size = 128
    val_batch_size = 500
    num_train = train_x_u8.size(0)
    num_val   = val_x_u8.size(0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3.5e-3, weight_decay=0.05)
    total_steps = epochs * (num_train // batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))

    telemetry = {
        "experiment": "Pure Spring-Free Gravitational Attention (CIFAR-100)",
        "parameters": total_params,
        "epochs": epochs,
        "epoch_metrics": []
    }

    t_start = time.time()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        perm = torch.randperm(num_train, device=device)
        train_loss = 0.0
        num_batches = 0

        for i in range(0, num_train, batch_size):
            idx = perm[i:i+batch_size]
            if len(idx) < batch_size:
                continue
            imgs = augment_gpu_batch(train_x_u8[idx], is_train=True)
            targets = train_y[idx]

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                logits = model(imgs)
                loss = F.cross_entropy(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item()
            num_batches += 1

        # Validation
        model.eval()
        v_loss, correct1, correct5 = 0.0, 0, 0
        with torch.no_grad():
            for i in range(0, num_val, val_batch_size):
                imgs = augment_gpu_batch(val_x_u8[i:i+val_batch_size], is_train=False)
                targets = val_y[i:i+val_batch_size]
                with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                    logits = model(imgs)
                    loss = F.cross_entropy(logits, targets)

                v_loss += loss.item()
                _, p1 = logits.topk(1, dim=-1)
                _, p5 = logits.topk(5, dim=-1)
                correct1 += p1.eq(targets.view(-1, 1)).sum().item()
                correct5 += p5.eq(targets.view(-1, 1)).sum().item()

        top1 = 100.0 * correct1 / num_val
        top5 = 100.0 * correct5 / num_val
        elapsed = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]

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
    json_path = "results/pure_spring_free_cifar100_telemetry.json"
    with open(json_path, "w") as f:
        json.dump(telemetry, f, indent=2)
    print(f"Saved raw telemetry to {json_path}")

    # ==========================================
    # 5. DIAGNOSTIC SPRING-FREE CANVASES VISUALIZATION
    # ==========================================
    print("\nGenerating Diagnostic Spring-Free Visualizations...")
    model.eval()
    num_samples = 4
    sample_imgs = augment_gpu_batch(val_x_u8[:num_samples], is_train=False)
    sample_labels = val_y[:num_samples]

    with torch.no_grad():
        logits, img_hist, clusters = model(sample_imgs, return_telemetry=True)
        preds = logits.argmax(dim=-1)

    fig, axes = plt.subplots(num_samples, 9, figsize=(22, 2.7 * num_samples))
    plt.subplots_adjust(wspace=0.35, hspace=0.35)

    for i in range(num_samples):
        orig_img = img_hist[0][i].cpu().permute(1, 2, 0).numpy()
        true_cls = classes[sample_labels[i].item()]
        pred_cls = classes[preds[i].item()]
        title_color = "green" if sample_labels[i] == preds[i] else "red"

        # Col 0: Original Photo
        axes[i, 0].imshow(orig_img)
        axes[i, 0].set_title(f"True: {true_cls}\nPred: {pred_cls}", color=title_color, fontsize=8, fontweight='bold')
        axes[i, 0].axis("off")

        # Cols 1 to 6: Evolving RGB Canvas (Round 1 to Round 6)
        for r in range(1, 7):
            r_img = img_hist[r][i].cpu().permute(1, 2, 0).numpy()
            axes[i, r].imshow(r_img)
            axes[i, r].set_title(f"Round {r}\nCanvas", fontsize=8)
            axes[i, r].axis("off")

        # Col 7: Total Pixel Drift Map (|Final - Initial|)
        final_img = img_hist[6][i].cpu().permute(1, 2, 0).numpy()
        drift_map = np.abs(final_img - orig_img).sum(axis=-1)
        axes[i, 7].imshow(drift_map, cmap="inferno")
        axes[i, 7].set_title("Total Pixel\nDrift Map", fontsize=8, fontweight='bold')
        axes[i, 7].axis("off")

        # Col 8: Centroid Clusters (Final Round, Group 0)
        c_map = clusters[-1][i, 0].view(16, 16).cpu().numpy()
        axes[i, 8].imshow(c_map, cmap="tab10", vmin=0, vmax=4)
        axes[i, 8].set_title("Centroid Clusters\n(K=5)", fontsize=8, fontweight='bold')
        axes[i, 8].axis("off")

    plt.suptitle("Pure Spring-Free Gravitational Attention (NO Tethers, Natural Centroid Drift)", fontsize=13, fontweight='bold', y=0.99)
    plt.tight_layout()
    plot_path = "results/pure_spring_free_evolution.png"
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    print(f"Saved visualization to {plot_path}")
    plt.show()
