import os
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# Ensure results directory exists
os.makedirs("results", exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on device: {device}")

# ==========================================
# 1. SYNTHETIC MQAR DATASET GENERATOR
# ==========================================
# Task Specification:
# - Sequence length T = 256
# - 16 Key-Value pairs inserted into the first half (t in [0, 128))
# - Queries presented in the second half (t in [128, 256))
# - Target at query position t_q is the corresponding value v_q
# ==========================================

class MQARDataset:
    def __init__(self, seq_len=256, num_kv_pairs=16, num_keys=64, num_vals=64):
        self.seq_len = seq_len
        self.num_kv = num_kv_pairs
        self.num_keys = num_keys
        self.num_vals = num_vals
        self.vocab_size = 1 + num_keys + num_vals
        self.key_offset = 1
        self.val_offset = 1 + num_keys

    def generate_batch(self, batch_size, device=device):
        T = self.seq_len
        inputs = torch.zeros(batch_size, T, dtype=torch.long, device=device)
        targets = torch.zeros(batch_size, T, dtype=torch.long, device=device)
        mask = torch.zeros(batch_size, T, dtype=torch.bool, device=device)

        half_T = T // 2

        for b in range(batch_size):
            # 1. Pick 16 distinct keys and 16 random values
            key_perm = torch.randperm(self.num_keys, device=device)[:self.num_kv] + self.key_offset
            vals = torch.randint(0, self.num_vals, (self.num_kv,), device=device) + self.val_offset

            # 2. Insert the 16 (k, v) pairs into first half
            available_slots = list(range(0, half_T - 2, 3))[:self.num_kv]
            for i, slot in enumerate(available_slots):
                inputs[b, slot]     = key_perm[i]
                inputs[b, slot + 1] = vals[i]

            # 3. Present the 16 queries in the second half
            query_order = torch.randperm(self.num_kv, device=device)
            query_slots = list(range(half_T + 2, T - 1, (half_T - 4) // self.num_kv))[:self.num_kv]

            for q_idx, q_slot in enumerate(query_slots):
                k_idx = query_order[q_idx]
                target_key = key_perm[k_idx]
                target_val = vals[k_idx]

                inputs[b, q_slot] = target_key
                targets[b, q_slot] = target_val
                mask[b, q_slot] = True

        return inputs, targets, mask


# ==========================================
# 2. ORTHOGONAL CAUSAL COULOMB ROUND (4 SPHERES)
# ==========================================
class OrthogonalCausalCoulombRound(nn.Module):
    """
    Causal Coulomb Attention on K Orthogonal Hyperspheres:
      - 4 Orthogonal 48-D Spheres S^47
      - Strict past light-cone mask (j < t)
      - Signed Coulomb Force:
          q_i * Q > 0: Matching Key-Value pairs ATTRACT
          q_i * Q < 0: Distractor tokens (0) actively REPEL
      - Hooke Springs: Anchors tokens back to initial coordinate
    """
    def __init__(self, dim=192, num_subspaces=4, max_seq_len=512, temp=10.0, eps=0.1):
        super().__init__()
        assert dim % num_subspaces == 0
        self.dim = dim
        self.K = num_subspaces
        self.d_k = dim // num_subspaces
        self.temp = temp
        self.eps = eps

        self.query_gen = nn.Linear(dim, dim)
        self.attractor_charge = nn.Parameter(torch.ones(self.K))
        self.eta_e = nn.Parameter(torch.full((self.K,), 0.40))

        # Strict past light-cone mask (j < t)
        past_mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=-1)
        past_mask[0, 0] = True
        self.register_buffer("past_mask", past_mask)

        self.norm = nn.LayerNorm(dim)
        self.mlp  = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x, u, kappa, charge):
        # x, u: (B, K, T, d_k) on S^(d_k-1)
        # kappa: (B, K, T, 1) in (0, 1)
        # charge: (B, K, T, 1) in (-1, 1)
        B, K, T, d_k = x.shape

        # 1. Causal Center of Mass up to step t per subspace
        counts = torch.arange(1, T + 1, device=x.device, dtype=x.dtype).view(1, 1, T, 1)
        causal_com = F.normalize(torch.cumsum(x, dim=2) / counts, dim=-1)

        # 2. Subspace queries
        com_flat = causal_com.permute(0, 2, 1, 3).reshape(B, T, self.dim)
        q = F.normalize(self.query_gen(com_flat).view(B, T, self.K, self.d_k).permute(0, 2, 1, 3), dim=-1)

        # 3. Pure Causal Attractor Retrieval (j < t)
        match_scores = torch.matmul(q, x.transpose(-1, -2)) * self.temp
        match_scores = match_scores.masked_fill(~self.past_mask[:T, :T].view(1, 1, T, T), float("-inf"))
        match_weights = F.softmax(match_scores, dim=-1)
        t_star = F.normalize(torch.matmul(match_weights, x), dim=-1)
        best_past_idx = match_scores.argmax(dim=-1)

        # 4. Coulomb Electrostatic Force (Pull & Push)
        cos_sim = (x * t_star).sum(dim=-1, keepdim=True)
        dist_sq = (2.0 - 2.0 * cos_sim).clamp(min=0.0) + self.eps
        Q_k = self.attractor_charge.view(1, K, 1, 1)
        force_coulomb = ((charge * Q_k) / dist_sq) * (t_star - x)

        # 5. Hooke Spring Force
        force_spring = kappa * (u - x)

        # 6. Combined Physical Step
        eta_k = self.eta_e.view(1, K, 1, 1)
        x_moved = F.normalize(x + eta_k * force_coulomb + force_spring, dim=-1)

        # 7. Cross-Subspace MLP Mixer
        x_flat = x_moved.permute(0, 2, 1, 3).reshape(B, T, self.dim)
        x_mlp  = x_flat + 0.25 * self.mlp(self.norm(x_flat))
        x_out  = F.normalize(x_mlp.view(B, T, self.K, self.d_k).permute(0, 2, 1, 3), dim=-1)

        return x_out, force_coulomb.norm(dim=-1), match_weights, best_past_idx


# ==========================================
# 3. CAUSAL COULOMB MQAR MODEL
# ==========================================
class CausalCoulombMQAR(nn.Module):
    def __init__(self, vocab_size=129, dim=192, num_subspaces=4, max_seq_len=256, default_rounds=6):
        super().__init__()
        self.dim = dim
        self.K = num_subspaces
        self.d_k = dim // num_subspaces
        self.default_rounds = default_rounds
        self.max_seq_len = max_seq_len

        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.kappa_proj = nn.Linear(dim, num_subspaces)
        self.charge_proj = nn.Linear(dim, num_subspaces)

        self.round = OrthogonalCausalCoulombRound(
            dim=dim, num_subspaces=num_subspaces, max_seq_len=max_seq_len
        )

        self.head_norm = nn.LayerNorm(dim * 2)
        self.lm_head   = nn.Linear(dim * 2, vocab_size)

    def forward(self, idx, num_rounds=None, return_attn=False):
        if num_rounds is None:
            num_rounds = self.default_rounds
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)

        raw = self.tok_emb(idx) + self.pos_emb(pos)
        u = F.normalize(raw.view(B, T, self.K, self.d_k).permute(0, 2, 1, 3), dim=-1)
        x = u.clone()

        u_flat = u.permute(0, 2, 1, 3).reshape(B, T, self.dim)
        kappa = torch.sigmoid(8.0 * self.kappa_proj(u_flat)).permute(0, 2, 1).unsqueeze(-1)
        charge = torch.tanh(4.0 * self.charge_proj(u_flat)).permute(0, 2, 1).unsqueeze(-1)

        last_attn = None
        for _ in range(num_rounds):
            x, _, attn_w, _ = self.round(x, u, kappa, charge)
            last_attn = attn_w

        x_final = x.permute(0, 2, 1, 3).reshape(B, T, self.dim)
        features = torch.cat([x_final, u_flat], dim=-1)
        logits = self.lm_head(self.head_norm(features))

        if return_attn:
            return logits, last_attn
        return logits


# ==========================================
# 4. RUN MQAR BENCHMARK (1500 STEPS)
# ==========================================
if __name__ == "__main__":
    dataset = MQARDataset(seq_len=256, num_kv_pairs=16, num_keys=64, num_vals=64)
    model = CausalCoulombMQAR(vocab_size=dataset.vocab_size, dim=192, num_subspaces=4, max_seq_len=256, default_rounds=6).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel Parameters : {total_params:,} | Causal Coulomb (4 Orthogonal Spheres S^47)")
    print(f"Task Config      : MQAR Sequence Length T = 256 | KV Pairs = 16 | Queries = 16")
    print(f"Vocab Size       : {dataset.vocab_size} (64 Keys, 64 Values, Distractor 0)")
    print(f"Benchmark Targets: Transformer: 7.66% Acc | Gravitational Attention: 12.63% Acc\n")

    steps = 1500
    batch_size = 64
    eval_interval = 100

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))

    telemetry = {
        "experiment": "Causal Coulomb MQAR Benchmark (T=256, 16 KV pairs, S^47)",
        "parameters": total_params,
        "total_steps": steps,
        "benchmark_targets": {
            "standard_transformer_loss": 3.7127,
            "standard_transformer_acc": 7.66,
            "gravitational_attention_loss": 2.6319,
            "gravitational_attention_acc": 12.63
        },
        "step_metrics": []
    }

    t0 = time.time()
    print("Step      Time     Loss       Recall Accuracy")
    print("-" * 48)

    for step in range(1, steps + 1):
        model.train()
        inputs, targets, mask = dataset.generate_batch(batch_size=batch_size, device=device)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=(device == "cuda")):
            logits = model(inputs, num_rounds=6)
            loss = F.cross_entropy(logits[mask], targets[mask])

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if step % eval_interval == 0:
            model.eval()
            with torch.no_grad():
                val_inputs, val_targets, val_mask = dataset.generate_batch(batch_size=128, device=device)
                with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                    val_logits = model(val_inputs, num_rounds=6)
                    val_loss = F.cross_entropy(val_logits[val_mask], val_targets[val_mask]).item()
                preds = val_logits[val_mask].argmax(dim=-1)
                acc = (preds == val_targets[val_mask]).float().mean().item() * 100.0

            elapsed = time.time() - t0
            telemetry["step_metrics"].append({
                "step": step, "loss": round(val_loss, 4), "accuracy": round(acc, 2), "elapsed_s": round(elapsed, 1)
            })

            print(f"Step {step:04d}    {elapsed:4.1f}s   {val_loss:.4f}       {acc:5.2f}%")
            t0 = time.time()

    # Save JSON telemetry
    json_path = "results/mqar_coulomb_telemetry.json"
    with open(json_path, "w") as f:
        json.dump(telemetry, f, indent=2)
    print(f"\nSaved raw telemetry to {json_path}")

    # ==========================================
    # 5. DEMO: ASSOCIATIVE RECALL ON A SAMPLE SEQUENCE
    # ==========================================
    model.eval()
    with torch.no_grad():
        demo_in, demo_tgt, demo_mask = dataset.generate_batch(batch_size=1, device=device)
        demo_logits, last_attn = model(demo_in, num_rounds=6, return_attn=True)

    query_steps = torch.where(demo_mask[0])[0].cpu().tolist()
    preds = demo_logits[0, query_steps].argmax(dim=-1).cpu().tolist()
    true_vals = demo_tgt[0, query_steps].cpu().tolist()
    queried_keys = demo_in[0, query_steps].cpu().tolist()

    print("\n" + "=" * 60)
    print("DEMO: COULOMB RECALL ON A SAMPLE SEQUENCE (T=256)")
    print("=" * 60)
    print(f"{'Query Step':<12} | {'Queried Key':<12} | {'True Value':<12} | {'Recalled':<12} | Match?")
    print("-" * 60)

    matches = 0
    for q_step, k, true_v, pred_v in zip(query_steps, queried_keys, true_vals, preds):
        is_match = (true_v == pred_v)
        matches += int(is_match)
        status = "✓ YES" if is_match else "✗ NO"
        print(f"{q_step:<12} | {k:<12} | {true_v:<12} | {pred_v:<12} | {status}")

    print("-" * 60)
    print(f"Sample Accuracy: {matches}/16 ({matches/16*100:.1f}%)")

    # ==========================================
    # 6. RETRIEVAL ATTENTION VISUALIZATION
    # ==========================================
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    attn_np = last_attn[0].cpu().numpy() # (4, 256, 256)

    for k in range(4):
        # Lookback attention from the 16 query positions
        q_attn = attn_np[k, query_steps, :128] # (16, 128)
        axes[k].imshow(q_attn, cmap='magma', aspect='auto')
        axes[k].set_title(f"Subspace {k+1} Query Attractor Match", fontsize=10)
        axes[k].set_xlabel("Past KV Pool [0, 128)")
        axes[k].set_ylabel("Query Index (1..16)")

    plt.suptitle("Causal Coulomb Attention: Which Past Key Attracts the Query at Step t?", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig("results/mqar_coulomb_proof.png", dpi=200, bbox_inches='tight')
    print("Saved results/mqar_coulomb_proof.png")
    plt.show()
