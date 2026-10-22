import os
import gc
import json
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
# 2. DUAL-QUERY COULOMB ROUND (M=2 ATTRACTORS, SINGLE SPHERE S^191)
# ==========================================
class DualQueryCoulombRound(nn.Module):
    """
    Dual-Query Coulomb Field on a Single Hypersphere S^(D-1):
      - 1 Single 192-dimensional space (No subspaces)
      - M = 2 Global Queries: Finds TWO distinct central attractors T*_1 and T*_2
      - Multi-Pole Coulomb Superposition:
          F_coulomb = F_coulomb(T*_1, q_1, Q_1) + F_coulomb(T*_2, q_2, Q_2)
      - Hooke Springs: F_spring = kappa * (u - x)
      - O(M * N) linear complexity
    """
    def __init__(self, dim=192, num_queries=2, temp=10.0, eps=0.1):
        super().__init__()
        self.dim = dim
        self.M = num_queries
        self.temp = temp
        self.eps = eps

        # Projects global center of mass into M distinct queries
        self.query_gen = nn.Linear(dim, dim * num_queries)

        # Attractor central charges Q_1, Q_2 (learnable vector of shape (M,))
        self.attractor_charges = nn.Parameter(torch.ones(self.M))

        # Electric coupling step size per query
        self.eta_e = nn.Parameter(torch.full((self.M,), 0.40))

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
        # charge: (B, N, M) in (-1, 1) -- signed charge per query
        B, N, D = x.shape

        # 1. Global context probe -> M distinct queries
        space_avg = F.normalize(x.mean(dim=1, keepdim=True), dim=-1) # (B, 1, D)
        queries = self.query_gen(space_avg).view(B, self.M, D)       # (B, M, D)
        queries = F.normalize(queries, dim=-1)

        # 2. Retrieve M distinct central attractors T*_1, ..., T*_M
        # match_scores: (B, M, N)
        match_scores  = torch.matmul(queries, x.transpose(-1, -2)) * self.temp
        match_weights = F.softmax(match_scores, dim=-1)              # (B, M, N)
        t_stars = F.normalize(torch.matmul(match_weights, x), dim=-1) # (B, M, D)
        best_indices = match_scores.argmax(dim=-1)                   # (B, M)

        # 3. Multi-Center Coulomb Superposition
        # x: (B, N, D) -> (B, 1, N, D)
        # t_stars: (B, M, D) -> (B, M, 1, D)
        x_exp = x.unsqueeze(1)                                       # (B, 1, N, D)
        t_exp = t_stars.unsqueeze(2)                                 # (B, M, 1, D)
        cos_sim = (x_exp * t_exp).sum(dim=-1, keepdim=True)         # (B, M, N, 1)
        dist_sq = (2.0 - 2.0 * cos_sim).clamp(min=0.0) + self.eps   # (B, M, N, 1)

        # charge: (B, N, M) -> (B, M, N, 1)
        q_exp = charge.permute(0, 2, 1).unsqueeze(-1)               # (B, M, N, 1)
        Q_exp = self.attractor_charges.view(1, self.M, 1, 1)        # (1, M, 1, 1)

        coulomb_coupling = (q_exp * Q_exp) / dist_sq                # (B, M, N, 1)
        diff_vectors = (t_exp - x_exp)                              # (B, M, N, D)
        forces_m = coulomb_coupling * diff_vectors                  # (B, M, N, D)

        if eta_override is not None:
            if isinstance(eta_override, (int, float)):
                eta_m = torch.full((1, self.M, 1, 1), float(eta_override), device=x.device, dtype=x.dtype)
            else:
                eta_m = eta_override.view(1, self.M, 1, 1)
        else:
            eta_m = self.eta_e.view(1, self.M, 1, 1)

        # Net Coulomb force is sum over all M attractors
        force_coulomb_total = (eta_m * forces_m).sum(dim=1)         # (B, N, D)

        # 4. Hooke's Law Spring Force
        k_val = kappa if kappa_override is None else kappa_override
        force_spring = k_val * (u - x)                              # (B, N, D)

        # 5. Combined Physical Update
        f_net = force_coulomb_total + force_spring
        x_moved = F.normalize(x + f_net, dim=-1)

        # 6. Residual Token Processing MLP
        x_out = F.normalize(x_moved + 0.25 * self.mlp(self.norm(x_moved)), dim=-1)

        return x_out, q_exp.squeeze(-1), forces_m.norm(dim=-1), best_indices


# ==========================================
# 3. DUAL-QUERY VISION NETWORK
# ==========================================
class DualQueryVisionNet(nn.Module):
    def __init__(self, num_classes=100, dim=192, num_queries=2, default_rounds=6):
        super().__init__()
        self.dim = dim
        self.M = num_queries
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

        # Predicts per-token signed charges q_i in (-1, 1) for each of the M attractors
        self.charge_proj = nn.Linear(dim, num_queries)
        self.charge_scale = 4.0

        self.round = DualQueryCoulombRound(dim=dim, num_queries=num_queries)
        self.head_norm = nn.LayerNorm(dim * 2)
        self.classifier = nn.Linear(dim * 2, num_classes)

    def forward(self, img, num_rounds=None, return_physics=False, eta_override=None, kappa_override=None):
        if num_rounds is None:
            num_rounds = self.default_rounds
        B = img.size(0)

        raw = self.patch_stem(img).flatten(2).transpose(1, 2) + self.pos_emb
        u = F.normalize(raw, dim=-1) # (B, 256, 192) on S^191
        x = u.clone()

        if kappa_override is None:
            kappa = torch.sigmoid(self.kappa_scale * self.kappa_proj(raw))
        else:
            kappa = torch.full((B, 256, 1), kappa_override, device=img.device, dtype=u.dtype)

        charge = torch.tanh(self.charge_scale * self.charge_proj(raw)) # (B, 256, M)

        trajectories = [x]
        charge_maps = []
        force_maps = []
        chosen_indices = []

        for _ in range(num_rounds):
            x, q_maps, f_maps, best_idxs = self.round(x, u, kappa, charge, eta_override=eta_override, kappa_override=kappa_override)
            if return_physics:
                trajectories.append(x)
                charge_maps.append(q_maps)
                force_maps.append(f_maps)
                chosen_indices.append(best_idxs)

        com = F.normalize(x.mean(dim=1), dim=-1)
        peak = x.max(dim=1).values
        logits = self.classifier(self.head_norm(torch.cat([com, peak], dim=-1)))

        if return_physics:
            return logits, u, kappa.squeeze(-1), charge, trajectories, force_maps, chosen_indices
        return logits


# ==========================================
# 4. TRAIN FOR 10 EPOCHS (BENCHMARK AGAINST 1-QUERY)
# ==========================================
if __name__ == "__main__":
    model = DualQueryVisionNet(num_classes=100, dim=192, num_queries=2, default_rounds=6).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters : {total_params:,} | Dual-Query Coulomb (M=2 Attractors on S^191)")
    print(f"Benchmark Target : Beat Single-Query's 48.09% Top-1 at Epoch 10 | Epochs: 10\n")

    epochs = 10
    batch_size = 128
    val_batch_size = 500
    num_train = train_x_u8.size(0)
    num_val   = val_x_u8.size(0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3.5e-3, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))

    telemetry = {
        "experiment": "CIFAR-100 Dual-Query Coulomb Vision (M=2, S^191)",
        "parameters": total_params,
        "epochs": epochs,
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
        avg_v_loss = v_loss / (num_val / val_batch_size)
        avg_tr_loss = train_loss / num_batches
        epoch_sec = time.time() - t0

        telemetry["epoch_metrics"].append({
            "epoch": epoch,
            "train_loss": round(avg_tr_loss, 4),
            "val_loss": round(avg_v_loss, 4),
            "top1_acc": round(top1, 2),
            "top5_acc": round(top5, 2),
            "time_seconds": round(epoch_sec, 1)
        })

        q_str = ", ".join([f"Q{m+1}:{model.round.attractor_charges[m].item():.2f}" for m in range(2)])
        print(f"Epoch {epoch:02d}/10 ({epoch_sec:.1f}s) | Loss: {avg_tr_loss:.3f} | "
              f"Val Loss: {avg_v_loss:.3f} | Top-1: {top1:.2f}% | Top-5: {top5:.2f}% | [{q_str}]")

    # ==========================================
    # 5. DUAL-QUERY PHYSICS ABLATIONS
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
    # Turn off Attractor 2 only (keep Attractor 1)
    q1_only_t1, _    = eval_ablation_cifar(eta_override=torch.tensor([0.40, 0.0], device=device))
    # Turn off Attractor 1 only (keep Attractor 2)
    q2_only_t1, _    = eval_ablation_cifar(eta_override=torch.tensor([0.0, 0.40], device=device))
    # Both attractors off
    noe_t1, _        = eval_ablation_cifar(eta_override=0.0)
    nospr_t1, _      = eval_ablation_cifar(kappa_override=0.0)
    flatspr_t1, _    = eval_ablation_cifar(kappa_override=0.5)

    q1 = model.round.attractor_charges[0].item()
    q2 = model.round.attractor_charges[1].item()

    ablations = {
        "full_dual_query_top1": round(full_t1, 2),
        "single_query_baseline_top1": 48.09,
        "query_1_only_top1": round(q1_only_t1, 2),
        "query_2_only_top1": round(q2_only_t1, 2),
        "both_electric_off_top1": round(noe_t1, 2),
        "springs_off_top1": round(nospr_t1, 2),
        "constant_spring_top1": round(flatspr_t1, 2)
    }
    telemetry["ablations"] = ablations

    print("\n=== EPOCH 10 DUAL-QUERY (M=2) BENCHMARK VERDICT ===")
    print(f"Attractor Charges   : Q1 = {q1:.4f} | Q2 = {q2:.4f}")
    print("-" * 65)
    print(f"1. Full Dual-Query Model (M=2)          : {full_t1:.2f}% Top-1 ({full_t5:.2f}% Top-5)")
    print(f"   [1-Query Single-Space Baseline       : 48.09% Top-1]")
    print(f"2. Attractor 1 Only (Kill Q2)           : {q1_only_t1:.2f}% Top-1 (Drop: -{full_t1 - q1_only_t1:.2f}%)")
    print(f"3. Attractor 2 Only (Kill Q1)           : {q2_only_t1:.2f}% Top-1 (Drop: -{full_t1 - q2_only_t1:.2f}%)")
    print(f"4. Both Electric Forces Off (eta = 0)   : {noe_t1:.2f}% Top-1 (Drop: -{full_t1 - noe_t1:.2f}%)")
    print(f"5. Springs Off (kappa = 0)              : {nospr_t1:.2f}% Top-1 (Drop: -{full_t1 - nospr_t1:.2f}%)")
    print(f"6. Constant Spring (kappa = 0.5)        : {flatspr_t1:.2f}% Top-1 (Drop: -{full_t1 - flatspr_t1:.2f}%)")

    # SAVE NUMERICAL TELEMETRY TO JSON
    json_path = "results/cifar100_dual_query_10epochs.json"
    with open(json_path, "w") as f:
        json.dump(telemetry, f, indent=2)
    print(f"\nSaved raw numerical telemetry to {json_path}")

    # ==========================================
    # 6. DIAGNOSTIC VISUALIZATION (Q1 CHARGE, Q2 CHARGE, DISPLACEMENT)
    # ==========================================
    model.eval()
    sample_u8 = val_x_u8[:4]
    sample_imgs = preprocess_gpu_batch(sample_u8, is_train=False)
    sample_labels = val_y[:4]

    with torch.no_grad():
        logits, u, kappa_maps, charge_maps, trajs, f_maps, best_idxs = model(sample_imgs, num_rounds=6, return_physics=True)
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

        # Col 1: Attractor 1 Charge q_{i, 1}
        q1_img = charge_maps[i, :, 0].view(16, 16).cpu().numpy()
        im_q1 = axes[i, 1].imshow(q1_img, cmap="coolwarm", vmin=-1.0, vmax=1.0)
        axes[i, 1].set_title("Attractor 1 Charge q_1\n[Red:Attract | Blue:Repel]", fontsize=9)
        axes[i, 1].axis("off")
        plt.colorbar(im_q1, ax=axes[i, 1], fraction=0.046, pad=0.04)

        # Col 2: Attractor 2 Charge q_{i, 2}
        q2_img = charge_maps[i, :, 1].view(16, 16).cpu().numpy()
        im_q2 = axes[i, 2].imshow(q2_img, cmap="coolwarm", vmin=-1.0, vmax=1.0)
        axes[i, 2].set_title("Attractor 2 Charge q_2\n[Red:Attract | Blue:Repel]", fontsize=9)
        axes[i, 2].axis("off")
        plt.colorbar(im_q2, ax=axes[i, 2], fraction=0.046, pad=0.04)

        # Col 3: Elasticity kappa (Stiffness)
        k_img = kappa_maps[i].view(16, 16).cpu().numpy()
        im_k = axes[i, 3].imshow(k_img, cmap="viridis", vmin=0.0, vmax=1.0)
        axes[i, 3].set_title(f"Elasticity kappa\n[min:{k_img.min():.2f}, max:{k_img.max():.2f}]", fontsize=9)
        axes[i, 3].axis("off")
        plt.colorbar(im_k, ax=axes[i, 3], fraction=0.046, pad=0.04)

        # Col 4: Total Displacement Stretch from Initial Patch u
        stretch = (1.0 - (trajs[-1][i] * u[i]).sum(dim=-1)).view(16, 16).cpu().numpy()
        im_s = axes[i, 4].imshow(stretch, cmap="magma")
        axes[i, 4].set_title("Total Displacement (1 - x_6 . u)", fontsize=9)
        axes[i, 4].axis("off")
        plt.colorbar(im_s, ax=axes[i, 4], fraction=0.046, pad=0.04)

    plt.suptitle("Dual-Query Coulomb Network (M=2 Attractors on Single Sphere S^191, 10 Epochs)", fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig("results/dual_query_cifar100_analysis.png", dpi=200, bbox_inches='tight')
    print("Saved results/dual_query_cifar100_analysis.png")
    plt.show()
