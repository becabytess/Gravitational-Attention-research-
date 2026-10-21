import gc
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset

# ==========================================
# 1. ULTRA-FAST GPU TENSOR PIPELINE FOR CIFAR-100
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on device: {device}")

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
# 2. ORTHOGONAL SUBSPACE VISION ROUND (VECTORIZED)
# ==========================================
class OrthogonalVisionRound(nn.Module):
    def __init__(self, dim=192, num_subspaces=4, temp=10.0, eps=0.1):
        super().__init__()
        assert dim % num_subspaces == 0
        self.dim = dim
        self.K = num_subspaces
        self.d_k = dim // num_subspaces
        self.temp = temp
        self.eps = eps

        self.query_gen = nn.Linear(dim, dim)
        self.log_mass  = nn.Parameter(torch.zeros(self.K))
        self.eta       = nn.Parameter(torch.full((self.K,), 0.40))

        self.norm = nn.LayerNorm(dim)
        self.mlp  = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x, u, kappa_per_subspace):
        # x, u: (B, K, N, d_k), kappa: (B, K, N, 1)
        B, K, N, d_k = x.shape

        # 1. Global Space Average per subspace: (B, K, 1, d_k)
        space_avg = F.normalize(x.mean(dim=2, keepdim=True), dim=-1)
        probe_in  = F.normalize(space_avg + x, dim=-1)
        probe_flat = probe_in.permute(0, 2, 1, 3).reshape(B, N, self.dim)

        # 2. Generate K Subspace Queries: (B, K, 1, d_k)
        q = self.query_gen(probe_flat.mean(dim=1)).view(B, K, 1, d_k)
        q = F.normalize(q, dim=-1)

        # 3. Retrieve Best-Matching T*_k in each orthogonal sphere: (B, K, 1, d_k)
        match_scores  = torch.matmul(q, x.transpose(-1, -2)) * self.temp  # (B, K, 1, N)
        match_weights = F.softmax(match_scores, dim=-1)                   # (B, K, 1, N)
        t_star = F.normalize(torch.matmul(match_weights, x), dim=-1)      # (B, K, 1, d_k)

        # 4. Inverse-Square Gravitational Pull toward T*_k (Zero vector collision across subspaces)
        cos_sim = (x * t_star).sum(dim=-1, keepdim=True)                  # (B, K, N, 1)
        dist_sq = (2.0 - 2.0 * cos_sim).clamp(min=0.0) + self.eps
        mass = F.softplus(self.log_mass).view(1, K, 1, 1)
        gravity_pull = mass / dist_sq                                     # (B, K, N, 1)
        force_grav = gravity_pull * (t_star - x)                          # (B, K, N, d_k)

        # 5. Per-Subspace Hooke's Law Spring Force
        force_spring = kappa_per_subspace * (u - x)                       # (B, K, N, d_k)

        # 6. Geodesic Movement + Cross-Subspace Residual MLP Mixing
        eta_k = self.eta.view(1, K, 1, 1)
        x_moved = F.normalize(x + eta_k * force_grav + force_spring, dim=-1)

        x_flat = x_moved.permute(0, 2, 1, 3).reshape(B, N, self.dim)
        x_mlp  = x_flat + 0.25 * self.mlp(self.norm(x_flat))
        x_out  = F.normalize(x_mlp.view(B, N, K, d_k).permute(0, 2, 1, 3), dim=-1)

        best_idx = match_scores.squeeze(2).argmax(dim=-1)                 # (B, K)
        return x_out, gravity_pull.squeeze(-1), best_idx


# ==========================================
# 3. ORTHOGONAL SUBSPACE VISION NET
# ==========================================
class OrthogonalVisionNet(nn.Module):
    def __init__(self, num_classes=100, dim=192, num_subspaces=4, default_rounds=6):
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
        self.kappa_proj = nn.Linear(dim, num_subspaces)
        self.kappa_scale = 8.0

        self.round = OrthogonalVisionRound(dim, num_subspaces=num_subspaces)
        self.head_norm = nn.LayerNorm(dim * 2)
        self.classifier = nn.Linear(dim * 2, num_classes)

    def forward(self, img, num_rounds=None, return_physics=False, eta_override=None, kappa_override=None):
        if num_rounds is None:
            num_rounds = self.default_rounds
        B = img.size(0)

        raw = self.patch_stem(img).flatten(2).transpose(1, 2) + self.pos_emb # (B, 256, D)
        u = F.normalize(raw.view(B, 256, self.K, self.d_k).permute(0, 2, 1, 3), dim=-1) # (B, K, 256, d_k)
        x = u.clone()

        if kappa_override is None:
            kappa = torch.sigmoid(self.kappa_scale * self.kappa_proj(raw)).permute(0, 2, 1).unsqueeze(-1) # (B, K, 256, 1)
        else:
            kappa = torch.full((B, self.K, 256, 1), kappa_override, device=img.device, dtype=u.dtype)

        saved_eta = self.round.eta.data.clone()
        if eta_override is not None:
            self.round.eta.data.fill_(eta_override)

        trajectories = [x]
        gravity_maps = []
        chosen_indices = []

        for _ in range(num_rounds):
            x, g_pull, best_idx = self.round(x, u, kappa)
            if return_physics:
                trajectories.append(x)
                gravity_maps.append(g_pull)
                chosen_indices.append(best_idx)

        if eta_override is not None:
            self.round.eta.data.copy_(saved_eta)

        x_final = x.permute(0, 2, 1, 3).reshape(B, 256, self.dim)
        com = F.normalize(x_final.mean(dim=1), dim=-1)
        peak = x_final.max(dim=1).values
        logits = self.classifier(self.head_norm(torch.cat([com, peak], dim=-1)))

        if return_physics:
            return logits, u, kappa.squeeze(-1), trajectories, gravity_maps, chosen_indices
        return logits


# ==========================================
# 4. TRAIN FOR 40 EPOCHS (~2.5s / EPOCH)
# ==========================================
if __name__ == "__main__":
    model = OrthogonalVisionNet(num_classes=100, dim=192, num_subspaces=4, default_rounds=6).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters: {total_params:,} | 4 Orthogonal 48-D Spheres | Epochs: 40\n")

    epochs = 40
    batch_size = 128
    val_batch_size = 500
    num_train = train_x_u8.size(0)
    num_val   = val_x_u8.size(0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3.5e-3, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))

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

        # Validation (Every 5 epochs + Epoch 40)
        if epoch % 5 == 0 or epoch == epochs:
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
            print(f"Epoch {epoch:02d}/40 ({time.time()-t0:.1f}s) | Loss: {train_loss/num_batches:.3f} | "
                  f"Val Loss: {v_loss/(num_val/val_batch_size):.3f} | Top-1: {top1:.2f}% | Top-5: {top5:.2f}%")
        else:
            print(f"Epoch {epoch:02d}/40 ({time.time()-t0:.1f}s) | Loss: {train_loss/num_batches:.3f}")


    # ==========================================
    # 5. DIAGNOSTICS & VISUALIZATION (4 IMAGES)
    # ==========================================
    model.eval()
    sample_u8 = val_x_u8[:4]
    sample_imgs = preprocess_gpu_batch(sample_u8, is_train=False)
    sample_labels = val_y[:4]

    with torch.no_grad():
        logits, u, kappa_maps, trajs, g_maps, best_idxs = model(sample_imgs, num_rounds=6, return_physics=True)
        probs = F.softmax(logits, dim=-1)

    fig, axes = plt.subplots(4, 4, figsize=(13, 12))
    for i in range(4):
        img_np = sample_u8[i].cpu().permute(1, 2, 0).numpy() / 255.0
        true_cls = classes[sample_labels[i].item()]
        top_p, top_i = probs[i].topk(1)
        pred_str = f"Pred: {classes[top_i[0].item()]} ({top_p[0].item()*100:.0f}%)\nTrue: {true_cls}"

        # 1. Image
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title(pred_str, fontsize=9, fontweight='bold')
        axes[i, 0].axis("off")

        # 2. Subspace 1 Elasticity Map (kappa_i,1)
        k_img = kappa_maps[i, 0].view(16, 16).cpu().numpy()
        im_k = axes[i, 1].imshow(k_img, cmap="viridis", vmin=0.0, vmax=1.0)
        axes[i, 1].set_title(f"Subspace 1 Elasticity kappa\n[min:{k_img.min():.2f}, max:{k_img.max():.2f}]", fontsize=9)
        axes[i, 1].axis("off")
        plt.colorbar(im_k, ax=axes[i, 1], fraction=0.046, pad=0.04)

        # 3. Round 6 Gravitational Pull (Subspace 1)
        axes[i, 2].imshow(g_maps[-1][i, 0].view(16, 16).cpu().numpy(), cmap="inferno")
        axes[i, 2].set_title("Round 6 Pull to T* (Subspace 1)", fontsize=10)
        axes[i, 2].axis("off")

        # 4. Final Angular Stretch from Anchor U (Subspace 1)
        stretch_r6 = (1.0 - (trajs[-1][i, 0] * u[i, 0]).sum(dim=-1)).view(16, 16).cpu().numpy()
        axes[i, 3].imshow(stretch_r6, cmap="magma")
        axes[i, 3].set_title("Final Subspace 1 Stretch", fontsize=10)
        axes[i, 3].axis("off")

    plt.suptitle("Orthogonal 4-Subspace Vision Model (40 Epochs): Emergent Object Shadows", fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig("results/emergent_object_shadows.png", dpi=200, bbox_inches='tight')
    print("Saved results/emergent_object_shadows.png")
    plt.show()


    # ==========================================
    # 6. LEARNED CONSTANTS & PHYSICS ABLATION AT EPOCH 40
    # ==========================================
    @torch.no_grad()
    def eval_ablation_cifar(eta_override=None, kappa_override=None, num_rounds=6):
        model.eval()
        c1, c5 = 0, 0
        for i in range(0, val_x_u8.size(0), 500):
            imgs = preprocess_gpu_batch(val_x_u8[i:i+500], is_train=False)
            labels = val_y[i:i+500]
            with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                logits = model(imgs, num_rounds=num_rounds, eta_override=eta_override, kappa_override=kappa_override)
            _, pred = logits.topk(5, dim=-1, largest=True, sorted=True)
            correct = pred.eq(labels.view(-1, 1).expand_as(pred))
            c1 += correct[:, :1].sum().item()
            c5 += correct[:, :5].sum().item()
        return 100.0 * c1 / val_x_u8.size(0), 100.0 * c5 / val_x_u8.size(0)

    masses = F.softplus(model.round.log_mass).detach().cpu().numpy()
    etas   = model.round.eta.detach().cpu().numpy()

    print("\n=== EPOCH 40 ORTHOGONAL SUBSPACE CONSTANTS (CIFAR-100) ===")
    for k in range(4):
        print(f"Subspace {k+1} (48-D) | Mass M_{k+1}: {masses[k]:.4f} | Pull Strength eta_{k+1}: {etas[k]:.4f}")

    full_t1, full_t5     = eval_ablation_cifar()
    nograv_t1, _         = eval_ablation_cifar(eta_override=0.0)
    nospr_t1, _          = eval_ablation_cifar(kappa_override=0.0)
    flatspr_t1, _        = eval_ablation_cifar(kappa_override=0.5)

    print("-" * 65)
    print(f"1. Full Orthogonal Model (40 Ep)         : {full_t1:.2f}% Top-1 ({full_t5:.2f}% Top-5)")
    print(f"2. Gravity Turned Off (eta = 0)          : {nograv_t1:.2f}% Top-1 (Drop: -{full_t1 - nograv_t1:.2f}%)")
    print(f"3. Spring Turned Off (kappa = 0)         : {nospr_t1:.2f}% Top-1 (Drop: -{full_t1 - nospr_t1:.2f}%)")
    print(f"4. Constant Spring (kappa = 0.5)         : {flatspr_t1:.2f}% Top-1 (Drop: -{full_t1 - flatspr_t1:.2f}%)")


    # ==========================================
    # 7. ELASTICITY & STRETCH ACROSS ALL 4 ORTHOGONAL SUBSPACES
    # ==========================================
    fig, axes = plt.subplots(4, 5, figsize=(15, 11.5))
    subspace_colors = ['#00ffff', '#ff00ff', '#00ff00', '#ffff00']

    for i in range(4):
        img_np = sample_u8[i].cpu().permute(1, 2, 0).numpy() / 255.0
        true_cls = classes[sample_labels[i].item()]
        top_p, top_i = probs[i].topk(1)
        pred_str = f"Pred: {classes[top_i[0].item()]} ({top_p[0].item()*100:.0f}%)\nTrue: {true_cls}"

        # Col 0: Image + Round 6 T* location for each of the 4 subspaces
        axes[i, 0].imshow(img_np)
        for k in range(4):
            idx_1d = best_idxs[-1][i, k].item()
            ry, rx = divmod(idx_1d, 16)
            axes[i, 0].scatter(rx * 2 + 1, ry * 2 + 1, s=75, c=subspace_colors[k], edgecolors='black', label=f"S{k+1} T*" if i==0 else "")
        axes[i, 0].set_title(pred_str + "\n(Dots = R6 T* per Subspace)", fontsize=9, fontweight='bold')
        axes[i, 0].axis("off")

        # Cols 1..4: Elasticity Map (kappa) for Subspace 1, 2, 3, 4
        for k in range(4):
            k_img = kappa_maps[i, k].view(16, 16).cpu().numpy()
            im_k = axes[i, k+1].imshow(k_img, cmap="viridis", vmin=0.0, vmax=1.0)
            axes[i, k+1].set_title(f"Subspace {k+1} Kappa\n[min:{k_img.min():.2f}, max:{k_img.max():.2f}]", fontsize=9)
            axes[i, k+1].axis("off")

    axes[0, 0].legend(loc="lower right", fontsize=7.5, framealpha=0.8)
    plt.suptitle("How the 4 Orthogonal Subspaces Partition Elasticity (kappa_{i,k}) Across the Image", fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig("results/subspace_elasticity_partitioning.png", dpi=200, bbox_inches='tight')
    print("Saved results/subspace_elasticity_partitioning.png")
    plt.show()
