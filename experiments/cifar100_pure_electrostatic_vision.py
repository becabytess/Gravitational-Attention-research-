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
# 2. ELECTROSTATIC + HOOKE ROUND (SINGLE SPHERE S^191)
# ==========================================
class ElectrostaticHookeRound(nn.Module):
    """
    Electrostatic + Hooke Elastic Field on 1 Single Hypersphere S^(D-1):
      - 1 Single 192-dimensional space (No subspaces)
      - Zero magnetic field (B = 0, no velocity tracking)
      - Coulomb Force (Signed charges q_i in [-1, 1]):
          q_i * Q > 0: Attracts toward central attractor T*
          q_i * Q < 0: Repels away from central attractor T*
      - Hooke Spring Force:
          F_spring = kappa * (u - x) anchors tokens back to initial patch u
    """
    def __init__(self, dim=192, temp=10.0, eps=0.1):
        super().__init__()
        self.dim = dim
        self.temp = temp
        self.eps = eps

        # Query generator for finding central attractor
        self.query_gen = nn.Linear(dim, dim)

        # Central attractor charge Q (learnable scalar)
        self.attractor_charge = nn.Parameter(torch.tensor(1.0))

        # Electric coupling step size eta
        self.eta_e = nn.Parameter(torch.tensor(0.40))

        # Channel MLP mixer
        self.norm = nn.LayerNorm(dim)
        self.mlp  = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x, u, kappa, charge, eta_override=None, kappa_override=None):
        # x, u: (B, N, D) on S^(D-1)
        # kappa: (B, N, 1) in (0, 1)
        # charge: (B, N, 1) in (-1, 1)
        B, N, D = x.shape

        # 1. Global context probe
        space_avg = F.normalize(x.mean(dim=1, keepdim=True), dim=-1)
        q = F.normalize(self.query_gen(space_avg), dim=-1)

        # 2. Search for central attractor T*
        match_scores  = torch.matmul(q, x.transpose(-1, -2)) * self.temp
        match_weights = F.softmax(match_scores, dim=-1)
        t_star = F.normalize(torch.matmul(match_weights, x), dim=-1)
        best_idx = match_scores.squeeze(1).argmax(dim=-1)

        # 3. Coulomb Force (Signed attraction or repulsion)
        cos_sim = (x * t_star).sum(dim=-1, keepdim=True)
        dist_sq = (2.0 - 2.0 * cos_sim).clamp(min=0.0) + self.eps
        Q = self.attractor_charge
        coulomb_coupling = (charge * Q) / dist_sq
        force_coulomb = coulomb_coupling * (t_star - x)

        # 4. Hooke's Law Spring Force
        k_val = kappa if kappa_override is None else kappa_override
        force_spring = k_val * (u - x)

        # 5. Combined Physical Update (Coulomb + Hooke Springs)
        eta = self.eta_e if eta_override is None else eta_override
        f_net = eta * force_coulomb + force_spring
        x_moved = F.normalize(x + f_net, dim=-1)

        # 6. Residual Token Processing MLP
        x_out = F.normalize(x_moved + 0.25 * self.mlp(self.norm(x_moved)), dim=-1)

        return x_out, coulomb_coupling.squeeze(-1), best_idx


# ==========================================
# 3. ELECTROSTATIC VISION NETWORK (SINGLE SPHERE)
# ==========================================
class ElectrostaticVisionNet(nn.Module):
    def __init__(self, num_classes=100, dim=192, default_rounds=6):
        super().__init__()
        self.dim = dim
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
        self.kappa_proj = nn.Linear(dim, 1)
        self.kappa_scale = 8.0

        # Predicts per-token signed electric charge q_i in (-1, 1)
        self.charge_proj = nn.Linear(dim, 1)
        self.charge_scale = 4.0

        self.round = ElectrostaticHookeRound(dim=dim)
        self.head_norm = nn.LayerNorm(dim * 2)
        self.classifier = nn.Linear(dim * 2, num_classes)

    def forward(self, img, num_rounds=None, return_physics=False, eta_override=None, kappa_override=None):
        if num_rounds is None:
            num_rounds = self.default_rounds
        B = img.size(0)

        raw = self.patch_stem(img).flatten(2).transpose(1, 2) + self.pos_emb
        u = F.normalize(raw, dim=-1) # (B, 256, 192) on S^191
        x = u.clone()

        # Predict per-token stiffness kappa in (0, 1) and charge q_i in (-1, 1)
        if kappa_override is None:
            kappa = torch.sigmoid(self.kappa_scale * self.kappa_proj(raw))
        else:
            kappa = torch.full((B, 256, 1), kappa_override, device=img.device, dtype=u.dtype)

        charge = torch.tanh(self.charge_scale * self.charge_proj(raw))

        trajectories = [x]
        coulomb_maps = []
        chosen_indices = []

        for _ in range(num_rounds):
            x, c_pull, best_idx = self.round(x, u, kappa, charge, eta_override=eta_override, kappa_override=kappa_override)
            if return_physics:
                trajectories.append(x)
                coulomb_maps.append(c_pull)
                chosen_indices.append(best_idx)

        com = F.normalize(x.mean(dim=1), dim=-1)
        peak = x.max(dim=1).values
        logits = self.classifier(self.head_norm(torch.cat([com, peak], dim=-1)))

        if return_physics:
            return logits, u, kappa.squeeze(-1), charge.squeeze(-1), trajectories, coulomb_maps, chosen_indices
        return logits


# ==========================================
# 4. TRAIN FOR 10 EPOCHS (FAST TEST SCHEDULE)
# ==========================================
if __name__ == "__main__":
    model = ElectrostaticVisionNet(num_classes=100, dim=192, default_rounds=6).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters : {total_params:,} | Electrostatic + Hooke (Single Sphere S^191)")
    print(f"Physics Config   : Coulomb + Springs (Zero Magnetic, No Subspaces) | Epochs: 10\n")

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

    q_attract = model.round.attractor_charge.item()
    eta_val   = model.round.eta_e.item()

    print("\n=== EPOCH 10 SINGLE-SPHERE ELECTROSTATIC RESULTS ===")
    print(f"Attractor Charge Q : {q_attract:.4f}")
    print(f"Electric Step eta  : {eta_val:.4f}")
    print("-" * 60)
    print(f"1. Full Model (Coulomb + Springs) : {full_t1:.2f}% Top-1 ({full_t5:.2f}% Top-5)")
    print(f"2. Electric Turned Off (eta = 0)  : {noe_t1:.2f}% Top-1 (Drop: -{full_t1 - noe_t1:.2f}%)")
    print(f"3. Spring Turned Off (kappa = 0)  : {nospr_t1:.2f}% Top-1 (Drop: -{full_t1 - nospr_t1:.2f}%)")
    print(f"4. Constant Spring (kappa = 0.5)  : {flatspr_t1:.2f}% Top-1 (Drop: -{full_t1 - flatspr_t1:.2f}%)")

    # ==========================================
    # 6. DIAGNOSTIC VISUALIZATION
    # ==========================================
    model.eval()
    sample_u8 = val_x_u8[:4]
    sample_imgs = preprocess_gpu_batch(sample_u8, is_train=False)
    sample_labels = val_y[:4]

    with torch.no_grad():
        logits, u, kappa_maps, charge_maps, trajs, c_maps, best_idxs = model(sample_imgs, num_rounds=6, return_physics=True)
        probs = F.softmax(logits, dim=-1)

    fig, axes = plt.subplots(4, 4, figsize=(14, 12))
    for i in range(4):
        img_np = sample_u8[i].cpu().permute(1, 2, 0).numpy() / 255.0
        true_cls = classes[sample_labels[i].item()]
        top_p, top_i = probs[i].topk(1)
        pred_str = f"Pred: {classes[top_i[0].item()]} ({top_p[0].item()*100:.0f}%)\nTrue: {true_cls}"

        # Col 0: Image
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title(pred_str, fontsize=9, fontweight='bold')
        axes[i, 0].axis("off")

        # Col 1: Electric Charge q_i (Red = Attract, Blue = Repel)
        q_img = charge_maps[i].view(16, 16).cpu().numpy()
        im_q = axes[i, 1].imshow(q_img, cmap="coolwarm", vmin=-1.0, vmax=1.0)
        axes[i, 1].set_title("Charge q_i\n[Red:Attract | Blue:Repel]", fontsize=9)
        axes[i, 1].axis("off")
        plt.colorbar(im_q, ax=axes[i, 1], fraction=0.046, pad=0.04)

        # Col 2: Elasticity kappa (Stiffness)
        k_img = kappa_maps[i].view(16, 16).cpu().numpy()
        im_k = axes[i, 2].imshow(k_img, cmap="viridis", vmin=0.0, vmax=1.0)
        axes[i, 2].set_title(f"Elasticity kappa\n[min:{k_img.min():.2f}, max:{k_img.max():.2f}]", fontsize=9)
        axes[i, 2].axis("off")
        plt.colorbar(im_k, ax=axes[i, 2], fraction=0.046, pad=0.04)

        # Col 3: Total Displacement Stretch from Initial Patch u
        stretch = (1.0 - (trajs[-1][i] * u[i]).sum(dim=-1)).view(16, 16).cpu().numpy()
        im_s = axes[i, 3].imshow(stretch, cmap="magma")
        axes[i, 3].set_title("Total Displacement (1 - x_6 . u)", fontsize=9)
        axes[i, 3].axis("off")
        plt.colorbar(im_s, ax=axes[i, 3], fraction=0.046, pad=0.04)

    plt.suptitle("Single-Sphere Electrostatic Network (Coulomb + Springs, S^191)", fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig("results/single_sphere_electrostatic_cifar100.png", dpi=200, bbox_inches='tight')
    print("Saved results/single_sphere_electrostatic_cifar100.png")
    plt.show()
