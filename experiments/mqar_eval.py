import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

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
        
        # Vocab allocation:
        # 0: Distractor / Padding token
        # 1 .. num_keys: Key tokens
        # (num_keys + 1) .. (num_keys + num_vals): Value tokens
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

        # Distractor tokens (randomly sampled from distractor range or filler)
        half_T = T // 2

        for b in range(batch_size):
            # 1. Pick 16 distinct keys and 16 random values
            key_perm = torch.randperm(self.num_keys, device=device)[:self.num_kv] + self.key_offset
            vals = torch.randint(0, self.num_vals, (self.num_kv,), device=device) + self.val_offset

            # 2. Insert the 16 (k, v) pairs into the first half [0, half_T - 2]
            # Each pair takes 2 consecutive slots: [t, t+1] = [k, v]
            # We choose 16 non-overlapping starting slots from 0 .. half_T - 2
            available_slots = list(range(0, half_T - 2, 3))[:self.num_kv]
            for i, slot in enumerate(available_slots):
                inputs[b, slot]     = key_perm[i]
                inputs[b, slot + 1] = vals[i]

            # 3. Present the 16 queries in the second half [half_T, T - 1]
            # Permute query order so model cannot memorize positional sequence
            query_order = torch.randperm(self.num_kv, device=device)
            query_slots = list(range(half_T + 2, T - 1, (half_T - 4) // self.num_kv))[:self.num_kv]

            for q_idx, q_slot in enumerate(query_slots):
                k_idx = query_order[q_idx]
                target_key = key_perm[k_idx]
                target_val = vals[k_idx]

                # At q_slot, input is the query key
                inputs[b, q_slot] = target_key
                # The next token target is the corresponding value
                targets[b, q_slot] = target_val
                # Loss and accuracy are evaluated strictly at the query positions
                mask[b, q_slot] = True

        return inputs, targets, mask


# ==========================================
# 2. PURE ORTHOGONAL GRAVITATIONAL ROUND (ZERO FORCED DECAY)
# ==========================================
class PureOrthogonalGravitationalRound(nn.Module):
    def __init__(self, dim=192, num_subspaces=4, max_seq_len=512, temp=10.0, eps=0.1):
        super().__init__()
        assert dim % num_subspaces == 0
        self.dim = dim
        self.K = num_subspaces
        self.d_k = dim // num_subspaces
        self.temp = temp
        self.eps = eps

        # Projects combined particle + field state into K subspace queries
        self.query_gen = nn.Linear(dim, dim)

        # Uniform initialization across all K subspaces
        self.log_mass = nn.Parameter(torch.zeros(self.K))
        self.eta      = nn.Parameter(torch.full((self.K,), 0.40))

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

    def forward(self, x, u, kappa):
        # x, u: (B, K, T, d_k), kappa: (B, K, T, 1)
        B, K, T, d_k = x.shape

        # 1. Causal Center of Mass up to step t
        counts = torch.arange(1, T + 1, device=x.device, dtype=x.dtype).view(1, 1, T, 1)
        causal_com = F.normalize(torch.cumsum(x, dim=2) / counts, dim=-1)

        # 2. Probe query combines field center of mass + current token state
        # (Allows query to represent: "I am query key k_q, search the past for k_q!")
        probe_in   = F.normalize(causal_com + x, dim=-1)
        probe_flat = probe_in.permute(0, 2, 1, 3).reshape(B, T, self.dim)
        q = self.query_gen(probe_flat).view(B, T, K, d_k).permute(0, 2, 1, 3)
        q = F.normalize(q, dim=-1)

        # 3. Pure Causal Attractor Retrieval in each subspace (j < t) -- ZERO FORCED DECAY
        match_scores = torch.matmul(q, x.transpose(-1, -2)) * self.temp
        match_scores = match_scores.masked_fill(~self.past_mask[:T, :T].view(1, 1, T, T), float("-inf"))

        match_weights = F.softmax(match_scores, dim=-1)
        t_star = F.normalize(torch.matmul(match_weights, x), dim=-1)
        best_past_idx = match_scores.argmax(dim=-1)

        # 4. Inverse-Square Gravitational Pull toward T*_k inside each orthogonal sphere
        cos_to_tstar = (x * t_star).sum(dim=-1, keepdim=True)
        dist_sq = (2.0 - 2.0 * cos_to_tstar).clamp(min=0.0) + self.eps
        mass = F.softplus(self.log_mass).view(1, K, 1, 1)
        gravity_pull = mass / dist_sq
        force_grav = gravity_pull * (t_star - x)

        # 5. Hooke's Law Spring Force back to Anchor U
        force_spring = kappa * (u - x)

        # 6. Geodesic Step + Cross-Subspace Residual MLP Mixing
        eta_k = self.eta.view(1, K, 1, 1)
        x_moved = F.normalize(x + eta_k * force_grav + force_spring, dim=-1)

        x_flat = x_moved.permute(0, 2, 1, 3).reshape(B, T, self.dim)
        x_mlp  = x_flat + 0.25 * self.mlp(self.norm(x_flat))
        x_out  = F.normalize(x_mlp.view(B, T, K, d_k).permute(0, 2, 1, 3), dim=-1)

        return x_out, gravity_pull.squeeze(-1), best_past_idx


# ==========================================
# 3. PURE ORTHOGONAL GRAVITATIONAL NETWORK FOR MQAR
# ==========================================
class GravitationalMQARNet(nn.Module):
    def __init__(self, vocab_size, dim=192, num_subspaces=4, max_seq_len=256, num_rounds=6):
        super().__init__()
        self.dim = dim
        self.K = num_subspaces
        self.d_k = dim // num_subspaces
        self.num_rounds = num_rounds
        self.max_seq_len = max_seq_len

        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.kappa_proj = nn.Linear(dim, num_subspaces)
        self.kappa_scale = 8.0

        # One shared weight-tied round module unrolled R times
        self.round = PureOrthogonalGravitationalRound(
            dim=dim, num_subspaces=num_subspaces, max_seq_len=max_seq_len
        )

        self.head_norm = nn.LayerNorm(dim * 2)
        self.readout_head = nn.Linear(dim * 2, vocab_size)

    def forward(self, idx, return_physics=False, eta_override=None, kappa_override=None):
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)

        raw = self.tok_emb(idx) + self.pos_emb(pos)
        u = F.normalize(raw.view(B, T, self.K, self.d_k).permute(0, 2, 1, 3), dim=-1)
        x = u.clone()

        u_flat = u.permute(0, 2, 1, 3).reshape(B, T, self.dim)
        if kappa_override is None:
            kappa = torch.sigmoid(self.kappa_scale * self.kappa_proj(u_flat)).permute(0, 2, 1).unsqueeze(-1)
        else:
            kappa = torch.full((B, self.K, T, 1), kappa_override, device=idx.device, dtype=u.dtype)

        saved_eta = self.round.eta.data.clone()
        if eta_override is not None:
            self.round.eta.data.fill_(eta_override)

        trajectories = [x]
        gravity_maps_k = []
        chosen_indices_k = []

        for _ in range(self.num_rounds):
            x, g_pull_k, best_idx_k = self.round(x, u, kappa)
            if return_physics:
                trajectories.append(x)
                gravity_maps_k.append(g_pull_k)
                chosen_indices_k.append(best_idx_k)

        if eta_override is not None:
            self.round.eta.data.copy_(saved_eta)

        x_final = x.permute(0, 2, 1, 3).reshape(B, T, self.dim)
        counts = torch.arange(1, T + 1, device=x.device, dtype=x.dtype).view(1, T, 1)
        causal_com = F.normalize(torch.cumsum(x_final, dim=1) / counts, dim=-1)
        readout = self.head_norm(torch.cat([x_final, causal_com], dim=-1))
        logits = self.readout_head(readout)

        if return_physics:
            return logits, u, kappa.squeeze(-1), trajectories, gravity_maps_k, chosen_indices_k
        return logits


# ==========================================
# 4. TRAIN ON MQAR BENCHMARK (T=256, 16 KV PAIRS)
# ==========================================
if __name__ == "__main__":
    dataset = MQARDataset(seq_len=256, num_kv_pairs=16, num_keys=64, num_vals=64)
    model = GravitationalMQARNet(
        vocab_size=dataset.vocab_size,
        dim=192,
        num_subspaces=4,
        max_seq_len=256,
        num_rounds=6
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters : {total_params:,} | 4 Orthogonal 48-D Spheres | Rounds: 6")
    print(f"Task Config      : MQAR Sequence Length T = 256 | KV Pairs = 16 | Queries = 16")
    print(f"Vocab Size       : {dataset.vocab_size} (64 Keys, 64 Values, Distractor 0)\n")

    steps = 1500
    batch_size = 64
    eval_interval = 100

    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))

    print("Step      Time     Loss       Recall Accuracy")
    print("-" * 48)

    t0 = time.time()
    for step in range(1, steps + 1):
        model.train()
        inputs, targets, mask = dataset.generate_batch(batch_size=batch_size)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=(device == "cuda")):
            logits = model(inputs)
            # Loss strictly evaluated at the 16 query positions
            active_logits = logits[mask]
            active_targets = targets[mask]
            loss = F.cross_entropy(active_logits, active_targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if step % eval_interval == 0 or step == steps:
            model.eval()
            total_corr = 0
            total_count = 0
            val_loss = 0.0

            with torch.no_grad():
                for _ in range(20):
                    v_in, v_tgt, v_mask = dataset.generate_batch(batch_size=batch_size)
                    with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                        v_logits = model(v_in)
                        v_act_logits = v_logits[v_mask]
                        v_act_targets = v_tgt[v_mask]
                        val_loss += F.cross_entropy(v_act_logits, v_act_targets).item()

                    preds = v_act_logits.argmax(dim=-1)
                    total_corr += (preds == v_act_targets).sum().item()
                    total_count += v_act_targets.numel()

            recall_acc = 100.0 * total_corr / total_count
            avg_vloss = val_loss / 20
            print(f"Step {step:04d}   {time.time()-t0:5.1f}s   {avg_vloss:.4f}     {recall_acc:6.2f}%")
            t0 = time.time()


    # ==========================================
    # 5. DIAGNOSTIC EVALUATION & RETRIEVAL PROOF
    # ==========================================
    model.eval()
    sample_inputs, sample_targets, sample_mask = dataset.generate_batch(batch_size=1)

    with torch.no_grad():
        logits, u, kappa_map, trajs, g_maps_k, best_idxs_k = model(sample_inputs, return_physics=True)
        preds = logits.argmax(dim=-1)

    query_indices = sample_mask[0].nonzero().squeeze(-1).cpu().numpy()
    print("\n" + "=" * 60)
    print("DEMO: ASSOCIATIVE RECALL ON A SAMPLE SEQUENCE (T=256)")
    print("=" * 60)
    print(f"{'Query Step':<12} | {'Queried Key':<12} | {'True Value':<12} | {'Recalled':<12} | {'Match?'}")
    print("-" * 60)

    correct_recalls = 0
    for q_idx in query_indices:
        q_key = sample_inputs[0, q_idx].item()
        true_val = sample_targets[0, q_idx].item()
        pred_val = preds[0, q_idx].item()
        match = "✓ YES" if pred_val == true_val else "✗ NO"
        if pred_val == true_val:
            correct_recalls += 1
        print(f"{q_idx:<12} | {q_key:<12} | {true_val:<12} | {pred_val:<12} | {match}")

    print("-" * 60)
    print(f"Sample Accuracy: {correct_recalls}/{len(query_indices)} ({100.0*correct_recalls/len(query_indices):.1f}%)\n")


    # ==========================================
    # 6. VISUALIZE ATTRACTOR RETRIEVAL HEATMAP
    # ==========================================
    # Plot which past token T* was retrieved across all 4 orthogonal subspaces
    r6_idxs = best_idxs_k[-1][0].cpu().numpy()  # (K, T)
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 8), sharex=True)
    for k in range(4):
        ax = axes[k]
        im = ax.plot(range(256), r6_idxs[k], color='cyan' if k%2==0 else 'magenta', lw=1.5, label=f"Subspace {k+1} T*")
        # Highlight query positions with vertical dashed lines
        for q_step in query_indices:
            ax.axvline(q_step, color='yellow', alpha=0.3, linestyle='--', lw=0.8)
        ax.set_ylabel(f"S{k+1} T* Index", fontsize=9, fontweight='bold')
        ax.set_ylim(0, 256)
        ax.grid(True, alpha=0.2)
        ax.legend(loc="upper left", fontsize=8)

    axes[-1].set_xlabel("Timestep t (0..128: KV Insertions | 128..256: Probed Queries)", fontsize=11, fontweight='bold')
    plt.suptitle("MQAR (T=256, 16 Keys): Which Past Token T* Attracts the Query at Step t?", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("results/mqar_retrieval_proof.png", dpi=200, bbox_inches='tight')
    print("Saved results/mqar_retrieval_proof.png")
    plt.show()
