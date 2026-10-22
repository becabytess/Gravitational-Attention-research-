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
# 2. PURE ELECTROSTATIC ROUND (NO SPRINGS, NO MAGNETIC, 1 SINGLE SPHERE)
# ==========================================
class PureElectrostaticRound(nn.Module):
    """
    Pure Electrostatic Dynamics on a Single Unit Hypersphere S^(D-1):
      - Single 192-dimensional space (No subspaces)
      - Zero magnetic field (B = 0, no velocity tracking)
      - Zero Hooke springs (kappa = 0, no anchoring to initial position)
      - Pure Coulomb Force:
          q_i * Q > 0: Attracts toward central attractor T*
          q_i * Q < 0: Repels away from central attractor T*
    """
    def __init__(self, dim=192, temp=10.0, eps=0.1):
        super().__init__()
        self.dim = dim
        self.temp = temp
        self.eps = eps

        # Query generator for finding central attractor
        self.query_gen = nn.Linear(dim, dim)

        # Predicts per-token signed electric charge q_i in (-1, 1)
        self.charge_proj = nn.Linear(dim, 1)
        self.charge_scale = 4.0

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

    def forward(self, x, eta_override=None):
        # x: (B, N, D) on S^(D-1)
        B, N, D = x.shape

        # 1. Global context probe
        space_avg = F.normalize(x.mean(dim=1, keepdim=True), dim=-1) # (B, 1, D)
        q = F.normalize(self.query_gen(space_avg), dim=-1)          # (B, 1, D)

        # 2. Search for central attractor T*
        match_scores  = torch.matmul(q, x.transpose(-1, -2)) * self.temp # (B, 1, N)
        match_weights = F.softmax(match_scores, dim=-1)
        t_star = F.normalize(torch.matmul(match_weights, x), dim=-1)      # (B, 1, D)
        best_idx = match_scores.squeeze(1).argmax(dim=-1)

        # 3. Predict per-token signed electric charge q_i in (-1, 1)
        charge = torch.tanh(self.charge_scale * self.charge_proj(x))       # (B, N, 1)

        # 4. Pure Coulomb Force
        # dist_sq = ||t_star - x||^2 = 2 - 2 * cos(theta)
        cos_sim = (x * t_star).sum(dim=-1, keepdim=True)                  # (B, N, 1)
        dist_sq = (2.0 - 2.0 * cos_sim).clamp(min=0.0) + self.eps
        Q = self.attractor_charge

        coulomb_coupling = (charge * Q) / dist_sq                          # (B, N, 1)
        force_coulomb = coulomb_coupling * (t_star - x)                    # (B, N, D)

        # 5. Pure Physical Step (No springs, No magnetic)
        eta = self.eta_e if eta_override is None else eta_override
        x_moved = F.normalize(x + eta * force_coulomb, dim=-1)

        # 6. Residual Token Processing MLP
        x_out = F.normalize(x_moved + 0.25 * self.mlp(self.norm(x_moved)), dim=-1)

        return x_out, charge.squeeze(-1), coulomb_coupling.squeeze(-1), best_idx


# ==========================================
# 3. PURE ELECTROSTATIC VISION NETWORK
# ==========================================
class PureElectrostaticVisionNet(nn.Module):
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

        self.round = PureElectrostaticRound(dim=dim)
        self.head_norm = nn.LayerNorm(dim * 2)
        self.classifier = nn.Linear(dim * 2, num_classes)

    def forward(self, img, num_rounds=None, return_physics=False, eta_override=None):
        if num_rounds is None:
            num_rounds = self.default_rounds
        B = img.size(0)

        raw = self.patch_stem(img).flatten(2).transpose(1, 2) + self.pos_emb
        u = F.normalize(raw, dim=-1) # (B, 256, 192) on S^191
        x = u.clone()

        trajectories = [x]
        charge_maps = []
        coulomb_maps = []
        chosen_indices = []

        for _ in range(num_rounds):
            x, charge, c_pull, best_idx = self.round(x, eta_override=eta_override)
            if return_physics:
                trajectories.append(x)
                charge_maps.append(charge)
                coulomb_maps.append(c_pull)
                chosen_indices.append(best_idx)

        com = F.normalize(x.mean(dim=1), dim=-1)
        peak = x.max(dim=1).values
        logits = self.classifier(self.head_norm(torch.cat([com, peak], dim=-1)))

        if return_physics:
            return logits, u, trajectories, charge_maps, coulomb_maps, chosen_indices
        return logits


# ==========================================
# 4. TRAIN FOR 10 EPOCHS (FAST TEST SCHEDULE)
# ==========================================
if __name__ == "__main__":
    model = PureElectrostaticVisionNet(num_classes=100, dim=192, default_rounds=6).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters : {total_params:,} | Pure Electrostatic (Single Sphere S^191)")
    print(f"Physics Config   : Coulomb Only (Zero Springs, Zero Magnetic, No Subspaces) | Epochs: 10\n")

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
    # 5. PURE ELECTROSTATIC ABLATION AT EPOCH 10
    # ==========================================
    @torch.no_grad()
    def eval_ablation_cifar(eta_override=None):
        model.eval()
        c1, c5 = 0, 0
        for i in range(0, val_x_u8.size(0), 500):
            imgs = preprocess_gpu_batch(val_x_u8[i:i+500], is_train=False)
            labels = val_y[i:i+500]
            with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                logits = model(imgs, num_rounds=6, eta_override=eta_override)
            _, pred = logits.topk(5, dim=-1, largest=True, sorted=True)
            correct = pred.eq(labels.view(-1, 1).expand_as(pred))
            c1 += correct[:, :1].sum().item()
            c5 += correct[:, :5].sum().item()
        return 100.0 * c1 / val_x_u8.size(0), 100.0 * c5 / val_x_u8.size(0)

    full_t1, full_t5 = eval_ablation_cifar()
    noe_t1, _        = eval_ablation_cifar(eta_override=0.0)

    q_attract = model.round.attractor_charge.item()
    eta_val   = model.round.eta_e.item()

    print("\n=== EPOCH 10 PURE ELECTROSTATIC RESULTS ===")
    print(f"Attractor Charge Q : {q_attract:.4f}")
    print(f"Electric Step eta  : {eta_val:.4f}")
    print("-" * 60)
    print(f"1. Full Pure Electrostatic Model  : {full_t1:.2f}% Top-1 ({full_t5:.2f}% Top-5)")
    print(f"2. Electric Turned Off (eta = 0)  : {noe_t1:.2f}% Top-1 (Drop: -{full_t1 - noe_t1:.2f}%)")

    # ==========================================
    # 6. DIAGNOSTIC VISUALIZATION (CHARGES & DISPLACEMENT)
    # ==========================================
    model.eval()
    sample_u8 = val_x_u8[:4]
    sample_imgs = preprocess_gpu_batch(sample_u8, is_train=False)
    sample_labels = val_y[:4]

    with torch.no_grad():
        logits, u, trajs, c_maps, f_maps, best_idxs = model(sample_imgs, num_rounds=6, return_physics=True)
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

        # Col 1: Final Round Electric Charge q_i (Blue = Repel, Red = Attract)
        q_img = c_maps[-1][i].view(16, 16).cpu().numpy()
        im_q = axes[i, 1].imshow(q_img, cmap="coolwarm", vmin=-1.0, vmax=1.0)
        axes[i, 1].set_title("Charge q_i\n[Red:Attract | Blue:Repel]", fontsize=9)
        axes[i, 1].axis("off")
        plt.colorbar(im_q, ax=axes[i, 1], fraction=0.046, pad=0.04)

        # Col 2: Net Coulomb Force Magnitude
        force_mag = f_maps[-1][i].abs().view(16, 16).cpu().numpy()
        im_f = axes[i, 2].imshow(force_mag, cmap="plasma")
        axes[i, 2].set_title("Coulomb Coupling |q*Q|/d^2", fontsize=9)
        axes[i, 2].axis("off")
        plt.colorbar(im_f, ax=axes[i, 2], fraction=0.046, pad=0.04)

        # Col 3: Total Displacement Stretch from Initial Patch u
        stretch = (1.0 - (trajs[-1][i] * u[i]).sum(dim=-1)).view(16, 16).cpu().numpy()
        im_s = axes[i, 3].imshow(stretch, cmap="magma")
        axes[i, 3].set_title("Total Displacement (1 - x_6 . u)", fontsize=9)
        axes[i, 3].axis("off")
        plt.colorbar(im_s, ax=axes[i, 3], fraction=0.046, pad=0.04)

    plt.suptitle("Pure Electrostatic Network (No Springs, No Magnetic, Single Sphere S^191)", fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig("results/pure_electrostatic_cifar100_analysis.png", dpi=200, bbox_inches='tight')
    print("Saved results/pure_electrostatic_cifar100_analysis.png")
    plt.show()
