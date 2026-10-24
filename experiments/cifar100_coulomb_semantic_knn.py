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
# 2. SEMANTIC k-NN MANIFOLD QUERY CNN
# ==========================================
class SemanticKNNQueryCNN(nn.Module):
    """
    Semantic Space Convolution:
      - Exploits SEMANTIC locality (cosine proximity in representation space)
      - Identifies k-nearest semantic neighbors on the hypersphere
      - 1D Depthwise Conv filters along the ordered neighbor sequence from
        densest semantic center to sparse perimeter
      - Extracts the semantic density gradient (grad_sem rho) and dominant bulge direction
    """
    def __init__(self, dim=192, num_subspaces=4, k_neighbors=9):
        super().__init__()
        self.dim = dim
        self.K = num_subspaces
        self.d_k = dim // num_subspaces
        self.k = k_neighbors
        
        # 1D Depthwise Convolution across the k-nearest semantic neighbors
        # Filters the density gradient along the semantic similarity ordering
        self.density_conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.BatchNorm1d(dim),
            nn.GELU()
        )
        
        # Subspace query projection
        self.query_proj = nn.Linear(dim, dim)

    def forward(self, x_flat):
        # x_flat: (B, N, D)
        B, N, D = x_flat.shape
        x_norm = F.normalize(x_flat, dim=-1)
        
        # 1. Pairwise semantic similarity matrix S_ij = x_i . x_j
        sim = torch.bmm(x_norm, x_norm.transpose(1, 2)) # (B, N, N)
        
        # 2. Find top-k nearest semantic neighbors on the hypersphere
        _, topk_idx = torch.topk(sim, k=self.k, dim=-1, largest=True, sorted=True) # (B, N, k)
        
        # 3. Memory-efficient neighbor gather: (B, N, k, D)
        batch_idx = torch.arange(B, device=x_flat.device).view(B, 1, 1).expand(B, N, self.k)
        neighbors = x_flat[batch_idx, topk_idx] # (B, N, k, D)
        
        # 4. Convolve across the k-neighbor dimension (from closest to farthest neighbor)
        # Input to conv1d: (B * N, D, k)
        neighbors_flat = neighbors.view(B * N, self.k, D).permute(0, 2, 1)
        density_filtered = self.density_conv(neighbors_flat) # (B * N, D, k)
        
        # 5. Local semantic density state per token (pooled across k neighbors)
        token_semantic_density = density_filtered.mean(dim=-1).view(B, N, D) # (B, N, D)
        
        # 6. Form subspace queries from density-weighted global pool
        global_semantic_summary = token_semantic_density.mean(dim=1) # (B, D)
        q = self.query_proj(global_semantic_summary).view(B, self.K, 1, self.d_k)
        q = F.normalize(q, dim=-1)
        
        return q, token_semantic_density


# ==========================================
# 3. ORTHOGONAL COULOMB ROUND WITH SEMANTIC k-NN QUERIES
# ==========================================
class OrthogonalCoulombSemanticRound(nn.Module):
    """
    Coulomb Electrostatics + Hooke Springs on K Orthogonal Hyperspheres
    guided by Semantic-Space k-NN Density Queries:
      - K = 4 orthogonal spheres (dim = 192, d_k = 48 on S^47)
      - Query generation powered by semantic manifold density & bulge detection
      - Signed Coulomb Force:
          q_i,k * Q_k > 0: Attracts toward central attractor T*_k
          q_i,k * Q_k < 0: Repels away from central attractor T*_k
      - Hooke Springs: F_spring = kappa_k * (u_k - x_k) anchors tokens back to initial patch u_k
    """
    def __init__(self, dim=192, num_subspaces=4, k_neighbors=9, temp=10.0, eps=0.1):
        super().__init__()
        assert dim % num_subspaces == 0
        self.dim = dim
        self.K = num_subspaces
        self.d_k = dim // num_subspaces
        self.temp = temp
        self.eps = eps

        # Semantic k-NN Query CNN
        self.semantic_query_cnn = SemanticKNNQueryCNN(dim=dim, num_subspaces=num_subspaces, k_neighbors=k_neighbors)

        # Attractor central charge Q_k (learnable scalar per subspace)
        self.attractor_charge = nn.Parameter(torch.ones(self.K))

        # Electric step size per subspace
        self.eta_e = nn.Parameter(torch.full((self.K,), 0.40))

        # Cross-subspace manifold mixer
        self.norm = nn.LayerNorm(dim)
        self.mlp  = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x, u, kappa, charge, eta_override=None, kappa_override=None):
        # x, u: (B, K, N, d_k) on S^(d_k-1)
        # kappa: (B, K, N, 1) in (0, 1)
        # charge: (B, K, N, 1) in (-1, 1)
        B, K, N, d_k = x.shape

        # 1. Flatten across subspaces to form token representation in semantic space
        x_flat = x.permute(0, 2, 1, 3).reshape(B, N, self.dim)
        
        # 2. Semantic Space Convolution: extracts density gradients & bulge direction
        q, _ = self.semantic_query_cnn(x_flat) # q: (B, K, 1, d_k)

        # 3. Retrieve central attractor T*_k guided by semantic density queries
        match_scores  = torch.matmul(q, x.transpose(-1, -2)) * self.temp
        match_weights = F.softmax(match_scores, dim=-1)
        t_star = F.normalize(torch.matmul(match_weights, x), dim=-1)
        best_idx = match_scores.squeeze(2).argmax(dim=-1)

        # 4. Coulomb Electrostatic Force (Signed attraction or repulsion)
        cos_sim = (x * t_star).sum(dim=-1, keepdim=True)
        dist_sq = (2.0 - 2.0 * cos_sim).clamp(min=0.0) + self.eps
        Q_k = self.attractor_charge.view(1, K, 1, 1)
        coulomb_coupling = (charge * Q_k) / dist_sq
        force_coulomb = coulomb_coupling * (t_star - x)

        # 5. Hooke's Law Spring Force
        k_val = kappa if kappa_override is None else kappa_override
        force_spring = k_val * (u - x)

        # 6. Combined Physical Step (Coulomb + Springs, Zero Magnetic)
        if eta_override is not None:
            eta_k = torch.full((1, K, 1, 1), eta_override, device=x.device, dtype=x.dtype)
        else:
            eta_k = self.eta_e.view(1, K, 1, 1)

        f_net = eta_k * force_coulomb + force_spring
        x_moved = F.normalize(x + f_net, dim=-1)

        # 7. Cross-subspace residual MLP
        x_flat_moved = x_moved.permute(0, 2, 1, 3).reshape(B, N, self.dim)
        x_mlp  = x_flat_moved + 0.25 * self.mlp(self.norm(x_flat_moved))
        x_out  = F.normalize(x_mlp.view(B, N, K, d_k).permute(0, 2, 1, 3), dim=-1)

        return x_out, coulomb_coupling.squeeze(-1), best_idx


# ==========================================
# 4. ORTHOGONAL COULOMB + SEMANTIC k-NN VISION NETWORK
# ==========================================
class OrthogonalCoulombSemanticVisionNet(nn.Module):
    def __init__(self, num_classes=100, dim=192, num_subspaces=4, k_neighbors=9, default_rounds=6):
        super().__init__()
        self.dim = dim
        self.K = num_subspaces
        self.d_k = dim // num_subspaces
        self.default_rounds = default_rounds

        self.patch_stem = nn.Sequential(
            nn.Conv2d(3, dim // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim // 2),
            nn.GELU(),
            nn.Conv2d(dim // 2, dim, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(dim)
        )
        self.pos_emb = nn.Parameter(torch.randn(1, 256, dim) * 0.02)

        # Predicts per-token spring stiffness kappa in (0, 1) per subspace
        self.kappa_proj = nn.Linear(dim, num_subspaces)
        self.kappa_scale = 8.0

        # Predicts per-token signed electric charge q_i in (-1, 1) per subspace
        self.charge_proj = nn.Linear(dim, num_subspaces)
        self.charge_scale = 4.0

        self.round = OrthogonalCoulombSemanticRound(dim, num_subspaces=num_subspaces, k_neighbors=k_neighbors)
        self.head_norm = nn.LayerNorm(dim * 2)
        self.classifier = nn.Linear(dim * 2, num_classes)

    def forward(self, img, num_rounds=None, return_physics=False, eta_override=None, kappa_override=None):
        if num_rounds is None:
            num_rounds = self.default_rounds
        B = img.size(0)

        raw = self.patch_stem(img).flatten(2).transpose(1, 2) + self.pos_emb
        u = F.normalize(raw.view(B, 256, self.K, self.d_k).permute(0, 2, 1, 3), dim=-1)
        x = u.clone()

        # Predict per-token stiffness kappa in (0, 1) and charge q_i in (-1, 1)
        if kappa_override is None:
            kappa = torch.sigmoid(self.kappa_scale * self.kappa_proj(raw)).permute(0, 2, 1).unsqueeze(-1)
        else:
            kappa = torch.full((B, self.K, 256, 1), kappa_override, device=img.device, dtype=u.dtype)

        charge = torch.tanh(self.charge_scale * self.charge_proj(raw)).permute(0, 2, 1).unsqueeze(-1)

        trajectories = [x]
        coulomb_maps = []
        chosen_indices = []

        for _ in range(num_rounds):
            x, c_pull, best_idx = self.round(x, u, kappa, charge, eta_override=eta_override, kappa_override=kappa_override)
            if return_physics:
                trajectories.append(x)
                coulomb_maps.append(c_pull)
                chosen_indices.append(best_idx)

        x_final = x.permute(0, 2, 1, 3).reshape(B, 256, self.dim)
        com = F.normalize(x_final.mean(dim=1), dim=-1)
        peak = x_final.max(dim=1).values
        logits = self.classifier(self.head_norm(torch.cat([com, peak], dim=-1)))

        if return_physics:
            return logits, u, kappa.squeeze(-1), charge.squeeze(-1), trajectories, coulomb_maps, chosen_indices
        return logits


# ==========================================
# 5. TRAIN FOR 20 EPOCHS (HEAD-TO-HEAD BENCHMARK)
# ==========================================
if __name__ == "__main__":
    model = OrthogonalCoulombSemanticVisionNet(num_classes=100, dim=192, num_subspaces=4, k_neighbors=9, default_rounds=6).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters : {total_params:,} | Orthogonal Coulomb + Semantic k-NN Conv Queries (4 Spheres S^47)")
    print(f"Benchmark Targets: Pure Gravity: 53.26% | Coulomb (Flat Mean): 54.31% Top-1 | Epochs: 20\n")

    epochs = 20
    batch_size = 128
    val_batch_size = 500
    num_train = train_x_u8.size(0)
    num_val   = val_x_u8.size(0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3.5e-3, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))

    telemetry = {
        "experiment": "Orthogonal Coulomb Attention + Semantic k-NN Queries (CIFAR-100, 20 Epochs)",
        "parameters": total_params,
        "epochs": epochs,
        "benchmark_targets": {
            "pure_gravity_epoch20_top1": 53.26,
            "coulomb_flat_mean_epoch20_top1": 54.31
        },
        "epoch_metrics": []
    }

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
                logits = model(imgs, num_rounds=6)
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
                    logits = model(imgs, num_rounds=6)
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

        print(f"Epoch {epoch:02d}/20 ({elapsed:.1f}s) | Loss: {train_loss/num_batches:.3f} | "
              f"Val Loss: {v_loss/(num_val/val_batch_size):.3f} | Top-1: {top1:.2f}% | Top-5: {top5:.2f}% | LR: {current_lr:.6f}")

    # ==========================================
    # 6. PHYSICS ABLATIONS AT EPOCH 20
    # ==========================================
    @torch.no_grad()
    def eval_ablation_cifar(eta_override=None, kappa_override=None):
        model.eval()
        c1, c5 = 0, 0
        for i in range(0, val_x_u8.size(0), 500):
            imgs = preprocess_gpu_batch(val_x_u8[i:i+500], is_train=False)
            labels = val_y[i:i+500]
            with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                logits = model(imgs, num_rounds=6, eta_override=eta_override, kappa_override=kappa_override)
            _, pred = logits.topk(5, dim=-1, largest=True, sorted=True)
            correct = pred.eq(labels.view(-1, 1).expand_as(pred))
            c1 += correct[:, :1].sum().item()
            c5 += correct[:, :5].sum().item()
        return 100.0 * c1 / val_x_u8.size(0), 100.0 * c5 / val_x_u8.size(0)

    full_t1, full_t5 = eval_ablation_cifar()
    noe_t1, _        = eval_ablation_cifar(eta_override=0.0)
    nospr_t1, _      = eval_ablation_cifar(kappa_override=0.0)
    flatspr_t1, _    = eval_ablation_cifar(kappa_override=0.5)

    q_attract = model.round.attractor_charge.detach().cpu().numpy()
    etas      = model.round.eta_e.detach().cpu().numpy()

    telemetry["ablations"] = {
        "full_model_top1": round(full_t1, 2),
        "full_model_top5": round(full_t5, 2),
        "electric_off_eta_0": round(noe_t1, 2),
        "springs_off_kappa_0": round(nospr_t1, 2),
        "constant_spring_kappa_0.5": round(flatspr_t1, 2),
        "subspace_attractor_charges": [round(float(q), 4) for q in q_attract],
        "subspace_electric_etas": [round(float(e), 4) for e in etas]
    }

    # Save JSON telemetry
    json_path = "results/cifar100_coulomb_semantic_knn_telemetry.json"
    with open(json_path, "w") as f:
        json.dump(telemetry, f, indent=2)
    print(f"\nSaved raw telemetry to {json_path}")

    print("\n=== EPOCH 20 COULOMB + SEMANTIC k-NN RESULTS ===")
    for k in range(4):
        print(f"Subspace {k+1} (48-D) | Attractor Q_{k+1}: {q_attract[k]:.4f} | Electric eta_{k+1}: {etas[k]:.4f}")

    print("-" * 65)
    print(f"1. Full Coulomb + Semantic k-NN (4 Spheres)  : {full_t1:.2f}% Top-1 ({full_t5:.2f}% Top-5)")
    print(f"   [Comparison vs. Flat Mean Coulomb         : 54.31% Top-1]")
    print(f"   [Comparison vs. Pure Gravity              : 53.26% Top-1]")
    print(f"2. Electric Turned Off (eta = 0)              : {noe_t1:.2f}% Top-1 (Drop: -{full_t1 - noe_t1:.2f}%)")
    print(f"3. Spring Turned Off (kappa = 0)              : {nospr_t1:.2f}% Top-1 (Drop: -{full_t1 - nospr_t1:.2f}%)")
    print(f"4. Constant Spring (kappa = 0.5)              : {flatspr_t1:.2f}% Top-1 (Drop: -{full_t1 - flatspr_t1:.2f}%)")

    # ==========================================
    # 7. DIAGNOSTIC VISUALIZATION (COULOMB + SEMANTIC k-NN)
    # ==========================================
    model.eval()
    sample_u8 = val_x_u8[:4]
    sample_imgs = preprocess_gpu_batch(sample_u8, is_train=False)
    sample_labels = val_y[:4]

    with torch.no_grad():
        logits, u, kappa_maps, charge_maps, trajs, c_maps, best_idxs = model(sample_imgs, num_rounds=6, return_physics=True)
        probs = F.softmax(logits, dim=-1)

    fig, axes = plt.subplots(4, 5, figsize=(16, 12))
    for i in range(4):
        img_np = sample_u8[i].cpu().permute(1, 2, 0).numpy() / 255.0
        true_cls = classes[sample_labels[i].item()]
        top_p, top_i = probs[i].topk(1)
        pred_str = f"Pred: {classes[top_i[0].item()]} ({top_p[0].item()*100:.0f}%)\nTrue: {true_cls}"

        # Col 0: Image
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title(pred_str, fontsize=9, fontweight='bold')
        axes[i, 0].axis("off")

        # Col 1: S1 Electric Charge q_i
        q_img = charge_maps[i, 0].view(16, 16).cpu().numpy()
        im_q = axes[i, 1].imshow(q_img, cmap="coolwarm", vmin=-1.0, vmax=1.0)
        axes[i, 1].set_title("S1 Charge q_i\n[Red:Attract | Blue:Repel]", fontsize=9)
        axes[i, 1].axis("off")
        plt.colorbar(im_q, ax=axes[i, 1], fraction=0.046, pad=0.04)

        # Col 2: S1 Elasticity kappa
        k_img = kappa_maps[i, 0].view(16, 16).cpu().numpy()
        im_k = axes[i, 2].imshow(k_img, cmap="viridis", vmin=0.0, vmax=1.0)
        axes[i, 2].set_title(f"S1 Elasticity kappa\n[min:{k_img.min():.2f}, max:{k_img.max():.2f}]", fontsize=9)
        axes[i, 2].axis("off")
        plt.colorbar(im_k, ax=axes[i, 2], fraction=0.046, pad=0.04)

        # Col 3: S2 Electric Charge q_i
        q2_img = charge_maps[i, 1].view(16, 16).cpu().numpy()
        im_q2 = axes[i, 3].imshow(q2_img, cmap="coolwarm", vmin=-1.0, vmax=1.0)
        axes[i, 3].set_title("S2 Charge q_i\n[Red:Attract | Blue:Repel]", fontsize=9)
        axes[i, 3].axis("off")
        plt.colorbar(im_q2, ax=axes[i, 3], fraction=0.046, pad=0.04)

        # Col 4: S1 Displacement Stretch
        stretch = (1.0 - (trajs[-1][i, 0] * u[i, 0]).sum(dim=-1)).view(16, 16).cpu().numpy()
        im_s = axes[i, 4].imshow(stretch, cmap="magma")
        axes[i, 4].set_title("S1 Displacement (1 - x_6 . u)", fontsize=9)
        axes[i, 4].axis("off")
        plt.colorbar(im_s, ax=axes[i, 4], fraction=0.046, pad=0.04)

    plt.suptitle("Orthogonal Coulomb + Semantic k-NN Conv Network (CIFAR-100, Epoch 20)", fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig("results/orthogonal_coulomb_semantic_knn_cifar100_20epochs.png", dpi=200, bbox_inches='tight')
    print("Saved results/orthogonal_coulomb_semantic_knn_cifar100_20epochs.png")
    plt.show()
