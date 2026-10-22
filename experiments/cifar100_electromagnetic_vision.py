import os
import gc
import time
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
# 2. ORTHOGONAL ELECTROMAGNETIC ROUND (COULOMB + LORENTZ)
# ==========================================
class OrthogonalElectromagneticRound(nn.Module):
    """
    Electromagnetic Field Dynamics on K Orthogonal Hyperspheres:
      1. Coulomb Electrostatics: Signed charges q_i in [-1, 1]
         - q_i * Q_k > 0: Attractive force toward T*
         - q_i * Q_k < 0: Repulsive force away from T* (Anti-collapse / Dipole separation)
      2. Magnetic Lorentz Force: F_mag = v x B
         - Velocity v = x^(r) - x^(r-1)
         - Skew-symmetric magnetic field tensor B = W - W^T (B = -B^T)
         - Zero work: purely induces orbital rotation along the hypersphere
      3. Elastic Restoring Springs: Hooke's Law tethers back to origin anchor U
    """
    def __init__(self, dim=192, num_subspaces=4, temp=10.0, eps=0.1):
        super().__init__()
        assert dim % num_subspaces == 0
        self.dim = dim
        self.K = num_subspaces
        self.d_k = dim // num_subspaces
        self.temp = temp
        self.eps = eps

        self.query_gen = nn.Linear(dim, dim)

        # Attractor central charge Q_k (learnable scalar per subspace)
        self.attractor_charge = nn.Parameter(torch.ones(self.K))
        # Electric step size (Coulomb coupling)
        self.eta_e = nn.Parameter(torch.full((self.K,), 0.40))
        # Magnetic step size (Lorentz rotation)
        self.mu_b  = nn.Parameter(torch.full((self.K,), 0.20))

        # Learnable anti-symmetric magnetic field tensor per subspace (B = W - W^T)
        self.raw_B = nn.Parameter(torch.randn(self.K, self.d_k, self.d_k) * 0.02)

        # Cross-subspace manifold mixer
        self.norm = nn.LayerNorm(dim)
        self.mlp  = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

    def get_magnetic_field(self):
        # Strictly anti-symmetric: B = -B^T guarantees v @ B . v = 0 (perpendicular deflection)
        return self.raw_B - self.raw_B.transpose(-1, -2)

    def forward(self, x, u, kappa, charge, v_prev):
        # x, u, v_prev: (B, K, N, d_k)
        # kappa: (B, K, N, 1) in (0, 1)
        # charge: (B, K, N, 1) in (-1, 1)
        B, K, N, d_k = x.shape

        # 1. Global Space Average per subspace
        space_avg = F.normalize(x.mean(dim=2, keepdim=True), dim=-1)
        probe_in  = F.normalize(space_avg + x, dim=-1)
        probe_flat = probe_in.permute(0, 2, 1, 3).reshape(B, N, self.dim)

        # 2. Subspace queries
        q = self.query_gen(probe_flat.mean(dim=1)).view(B, K, 1, d_k)
        q = F.normalize(q, dim=-1)

        # 3. Retrieve central attractor T*_k
        match_scores  = torch.matmul(q, x.transpose(-1, -2)) * self.temp
        match_weights = F.softmax(match_scores, dim=-1)
        t_star = F.normalize(torch.matmul(match_weights, x), dim=-1)
        best_idx = match_scores.squeeze(2).argmax(dim=-1)

        # 4. Coulomb Electrostatic Force (Attractive or Repulsive based on sign!)
        # F_coulomb = (q_i * Q_k / d^2) * (t_star - x)
        cos_sim = (x * t_star).sum(dim=-1, keepdim=True)
        dist_sq = (2.0 - 2.0 * cos_sim).clamp(min=0.0) + self.eps
        Q_k = self.attractor_charge.view(1, K, 1, 1)
        
        coulomb_coupling = (charge * Q_k) / dist_sq
        force_coulomb = coulomb_coupling * (t_star - x)

        # 5. Magnetic Lorentz Force (v x B): Perpendicular orbital deflection
        B_mat = self.get_magnetic_field() # (K, d_k, d_k)
        force_lorentz = torch.matmul(v_prev, B_mat.unsqueeze(0)) # (B, K, N, d_k)

        # 6. Hooke's Law Spring Force
        force_spring = kappa * (u - x)

        # 7. Total Electromagnetic Update
        eta_k = self.eta_e.view(1, K, 1, 1)
        mu_k  = self.mu_b.view(1, K, 1, 1)
        f_net = eta_k * force_coulomb + mu_k * force_lorentz + force_spring

        x_moved = F.normalize(x + f_net, dim=-1)
        v_next  = x_moved - x # token velocity for next round

        # Cross-subspace residual MLP
        x_flat = x_moved.permute(0, 2, 1, 3).reshape(B, N, self.dim)
        x_mlp  = x_flat + 0.25 * self.mlp(self.norm(x_flat))
        x_out  = F.normalize(x_mlp.view(B, N, K, d_k).permute(0, 2, 1, 3), dim=-1)

        return x_out, v_next, coulomb_coupling.squeeze(-1), force_lorentz.norm(dim=-1), best_idx


# ==========================================
# 3. ELECTROMAGNETIC VISION NETWORK
# ==========================================
class ElectromagneticVisionNet(nn.Module):
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

        # Predicts per-token spring stiffness kappa in (0, 1)
        self.kappa_proj = nn.Linear(dim, num_subspaces)
        self.kappa_scale = 8.0

        # Predicts per-token signed electric charge q_i in (-1, 1) via tanh
        self.charge_proj = nn.Linear(dim, num_subspaces)
        self.charge_scale = 4.0

        self.round = OrthogonalElectromagneticRound(dim, num_subspaces=num_subspaces)
        self.head_norm = nn.LayerNorm(dim * 2)
        self.classifier = nn.Linear(dim * 2, num_classes)

    def forward(self, img, num_rounds=None, return_physics=False, eta_override=None, mu_override=None, kappa_override=None):
        if num_rounds is None:
            num_rounds = self.default_rounds
        B = img.size(0)

        raw = self.patch_stem(img).flatten(2).transpose(1, 2) + self.pos_emb
        u = F.normalize(raw.view(B, 256, self.K, self.d_k).permute(0, 2, 1, 3), dim=-1)
        x = u.clone()

        # Initial velocity v^(0) = 0
        v = torch.zeros_like(x)

        # Predict per-token stiffness kappa in (0, 1) and charge q_i in (-1, 1)
        if kappa_override is None:
            kappa = torch.sigmoid(self.kappa_scale * self.kappa_proj(raw)).permute(0, 2, 1).unsqueeze(-1)
        else:
            kappa = torch.full((B, self.K, 256, 1), kappa_override, device=img.device, dtype=u.dtype)

        charge = torch.tanh(self.charge_scale * self.charge_proj(raw)).permute(0, 2, 1).unsqueeze(-1)

        saved_eta = self.round.eta_e.data.clone()
        saved_mu  = self.round.mu_b.data.clone()
        if eta_override is not None:
            self.round.eta_e.data.fill_(eta_override)
        if mu_override is not None:
            self.round.mu_b.data.fill_(mu_override)

        trajectories = [x]
        coulomb_maps = []
        lorentz_maps = []
        chosen_indices = []

        for _ in range(num_rounds):
            x, v, c_pull, l_mag, best_idx = self.round(x, u, kappa, charge, v)
            if return_physics:
                trajectories.append(x)
                coulomb_maps.append(c_pull)
                lorentz_maps.append(l_mag)
                chosen_indices.append(best_idx)

        if eta_override is not None:
            self.round.eta_e.data.copy_(saved_eta)
        if mu_override is not None:
            self.round.mu_b.data.copy_(saved_mu)

        x_final = x.permute(0, 2, 1, 3).reshape(B, 256, self.dim)
        com = F.normalize(x_final.mean(dim=1), dim=-1)
        peak = x_final.max(dim=1).values
        logits = self.classifier(self.head_norm(torch.cat([com, peak], dim=-1)))

        if return_physics:
            return logits, u, kappa.squeeze(-1), charge.squeeze(-1), trajectories, coulomb_maps, lorentz_maps, chosen_indices
        return logits


# ==========================================
# 4. TRAIN FOR 10 EPOCHS (FAST TEST SCHEDULE)
# ==========================================
if __name__ == "__main__":
    model = ElectromagneticVisionNet(num_classes=100, dim=192, num_subspaces=4, default_rounds=6).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters : {total_params:,} | Electromagnetic Field (Coulomb + Lorentz on S^47)")
    print(f"Subspaces        : 4 Orthogonal 48-D Spheres | Epochs: 10\n")

    epochs = 10
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
        print(f"Epoch {epoch:02d}/10 ({time.time()-t0:.1f}s) | Loss: {train_loss/num_batches:.3f} | "
              f"Val Loss: {v_loss/(num_val/val_batch_size):.3f} | Top-1: {top1:.2f}% | Top-5: {top5:.2f}%")


    # ==========================================
    # 5. PHYSICS ABLATIONS AT EPOCH 10
    # ==========================================
    @torch.no_grad()
    def eval_ablation_cifar(eta_override=None, mu_override=None, kappa_override=None):
        model.eval()
        c1, c5 = 0, 0
        for i in range(0, val_x_u8.size(0), 500):
            imgs = preprocess_gpu_batch(val_x_u8[i:i+500], is_train=False)
            labels = val_y[i:i+500]
            with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                logits = model(
                    imgs, num_rounds=6,
                    eta_override=eta_override,
                    mu_override=mu_override,
                    kappa_override=kappa_override
                )
            _, pred = logits.topk(5, dim=-1, largest=True, sorted=True)
            correct = pred.eq(labels.view(-1, 1).expand_as(pred))
            c1 += correct[:, :1].sum().item()
            c5 += correct[:, :5].sum().item()
        return 100.0 * c1 / val_x_u8.size(0), 100.0 * c5 / val_x_u8.size(0)

    full_t1, full_t5  = eval_ablation_cifar()
    noe_t1, _         = eval_ablation_cifar(eta_override=0.0)
    nob_t1, _         = eval_ablation_cifar(mu_override=0.0)
    nospr_t1, _       = eval_ablation_cifar(kappa_override=0.0)
    flatspr_t1, _     = eval_ablation_cifar(kappa_override=0.5)

    q_attract = model.round.attractor_charge.detach().cpu().numpy()
    etas      = model.round.eta_e.detach().cpu().numpy()
    mus       = model.round.mu_b.detach().cpu().numpy()

    print("\n=== EPOCH 10 ELECTROMAGNETIC CONSTANTS ===")
    for k in range(4):
        print(f"Subspace {k+1} (48-D) | Attractor Q_{k+1}: {q_attract[k]:.4f} | Electric eta_{k+1}: {etas[k]:.4f} | Magnetic mu_{k+1}: {mus[k]:.4f}")

    print("-" * 68)
    print(f"1. Full Electromagnetic Model            : {full_t1:.2f}% Top-1 ({full_t5:.2f}% Top-5)")
    print(f"2. Electric Turned Off (eta = 0)         : {noe_t1:.2f}% Top-1 (Drop: -{full_t1 - noe_t1:.2f}%)")
    print(f"3. Magnetic Turned Off (mu = 0)          : {nob_t1:.2f}% Top-1 (Drop: -{full_t1 - nob_t1:.2f}%)")
    print(f"4. Spring Turned Off (kappa = 0)         : {nospr_t1:.2f}% Top-1 (Drop: -{full_t1 - nospr_t1:.2f}%)")
    print(f"5. Constant Spring (kappa = 0.5)         : {flatspr_t1:.2f}% Top-1 (Drop: -{full_t1 - flatspr_t1:.2f}%)")


    # ==========================================
    # 6. DIAGNOSTIC VISUALIZATION (CHARGES, CURVATURE, SHADOWS)
    # ==========================================
    model.eval()
    sample_u8 = val_x_u8[:4]
    sample_imgs = preprocess_gpu_batch(sample_u8, is_train=False)
    sample_labels = val_y[:4]

    with torch.no_grad():
        logits, u, kappa_maps, charge_maps, trajs, c_maps, l_maps, best_idxs = model(sample_imgs, num_rounds=6, return_physics=True)
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

        # Col 1: Subspace 1 Electric Charge q_i (Blue = Attracted, Red = Repelled)
        q_img = charge_maps[i, 0].view(16, 16).cpu().numpy()
        im_q = axes[i, 1].imshow(q_img, cmap="coolwarm", vmin=-1.0, vmax=1.0)
        axes[i, 1].set_title(f"S1 Charge q_i\n[Blue:Attract | Red:Repel]", fontsize=9)
        axes[i, 1].axis("off")
        plt.colorbar(im_q, ax=axes[i, 1], fraction=0.046, pad=0.04)

        # Col 2: Subspace 1 Elasticity kappa
        k_img = kappa_maps[i, 0].view(16, 16).cpu().numpy()
        im_k = axes[i, 2].imshow(k_img, cmap="viridis", vmin=0.0, vmax=1.0)
        axes[i, 2].set_title(f"S1 Elasticity kappa\n[min:{k_img.min():.2f}, max:{k_img.max():.2f}]", fontsize=9)
        axes[i, 2].axis("off")
        plt.colorbar(im_k, ax=axes[i, 2], fraction=0.046, pad=0.04)

        # Col 3: Magnetic Lorentz Deflection Magnitude
        mag_img = l_maps[-1][i, 0].view(16, 16).cpu().numpy()
        im_m = axes[i, 3].imshow(mag_img, cmap="plasma")
        axes[i, 3].set_title("Lorentz Deflection |v x B|", fontsize=9)
        axes[i, 3].axis("off")
        plt.colorbar(im_m, ax=axes[i, 3], fraction=0.046, pad=0.04)

        # Col 4: Final Displacement Stretch from Anchor U
        stretch = (1.0 - (trajs[-1][i, 0] * u[i, 0]).sum(dim=-1)).view(16, 16).cpu().numpy()
        axes[i, 4].imshow(stretch, cmap="magma")
        axes[i, 4].set_title("Final Manifold Stretch", fontsize=9)
        axes[i, 4].axis("off")

    plt.suptitle("Electromagnetic Field Network (10 Epochs): Coulomb Dipoles & Lorentz Deflections", fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig("results/electromagnetic_cifar100_analysis.png", dpi=200, bbox_inches='tight')
    print("Saved results/electromagnetic_cifar100_analysis.png")
    plt.show()
