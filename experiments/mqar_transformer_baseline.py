import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# Ensure results directory exists (avoids Colab/Jupyter FileNotFoundError)
os.makedirs("results", exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on device: {device}")

# ==========================================
# 1. IDENTICAL SYNTHETIC MQAR DATASET
# ==========================================
class MQARDataset:
    def __init__(self, seq_len=256, num_kv_pairs=16, num_keys=64, num_vals=64):
        self.seq_len = seq_len
        self.num_kv = num_kv_pairs
        
        # Vocab: 0 = distractor, 1..64 = keys, 65..128 = values
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
            # 1. Sample 16 distinct keys and 16 random values
            key_perm = torch.randperm(self.num_keys, device=device)[:self.num_kv] + self.key_offset
            vals = torch.randint(0, self.num_vals, (self.num_kv,), device=device) + self.val_offset

            # 2. Insert the 16 (k, v) pairs into the first half [0, half_T - 2]
            available_slots = list(range(0, half_T - 2, 3))[:self.num_kv]
            for i, slot in enumerate(available_slots):
                inputs[b, slot]     = key_perm[i]
                inputs[b, slot + 1] = vals[i]

            # 3. Present 16 queries in the second half [half_T, T - 1] in scrambled order
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
# 2. STANDARD CAUSAL TRANSFORMER (FULL SELF-ATTENTION)
# ==========================================
class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x, return_attn=False):
        B, T, C = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        if return_attn:
            # Explicit attention matrix for diagnostics
            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
            scores = scores.masked_fill(~causal_mask.view(1, 1, T, T), float("-inf"))
            weights = F.softmax(scores, dim=-1)
            out = torch.matmul(weights, v)
            out = out.transpose(1, 2).contiguous().view(B, T, C)
            return self.out_proj(out), weights
        else:
            # Ultra-fast native FlashAttention / SDPA
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            out = out.transpose(1, 2).contiguous().view(B, T, C)
            return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=2.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, num_heads)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x, return_attn=False):
        if return_attn:
            attn_out, weights = self.attn(self.ln1(x), return_attn=True)
            x = x + attn_out
            x = x + self.mlp(self.ln2(x))
            return x, weights
        else:
            x = x + self.attn(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
            return x


class StandardCausalTransformerLM(nn.Module):
    def __init__(self, vocab_size, dim=128, num_layers=2, num_heads=4, max_seq_len=256, mlp_ratio=2.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, idx, return_attn=False):
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)

        attn_maps = []
        for block in self.blocks:
            if return_attn:
                x, weights = block(x, return_attn=True)
                attn_maps.append(weights)
            else:
                x = block(x)

        logits = self.head(self.ln_f(x))
        if return_attn:
            return logits, attn_maps
        return logits


# ==========================================
# 3. TRAIN STANDARD TRANSFORMER ON MQAR
# ==========================================
if __name__ == "__main__":
    dataset = MQARDataset(seq_len=256, num_kv_pairs=16, num_keys=64, num_vals=64)

    # 2-layer Causal Transformer with dim=128 matches the ~300k parameter scale
    model = StandardCausalTransformerLM(
        vocab_size=dataset.vocab_size,
        dim=128,
        num_layers=2,
        num_heads=4,
        max_seq_len=256,
        mlp_ratio=2.0
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters : {total_params:,} | Standard Causal Transformer (2 Layers, 4 Heads)")
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
    # 4. DIAGNOSTIC EVALUATION (SAMPLE SEQUENCE)
    # ==========================================
    model.eval()
    sample_inputs, sample_targets, sample_mask = dataset.generate_batch(batch_size=1)

    with torch.no_grad():
        logits, attn_maps = model(sample_inputs, return_attn=True)
        preds = logits.argmax(dim=-1)

    query_indices = sample_mask[0].nonzero().squeeze(-1).cpu().numpy()
    print("\n" + "=" * 60)
    print("DEMO: TRANSFORMER RECALL ON A SAMPLE SEQUENCE (T=256)")
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
    # 5. VISUALIZE ATTENTION MAP
    # ==========================================
    # Layer 2, Head 0 attention matrix (shape: T x T)
    l2_attn = attn_maps[-1][0, 0].cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(l2_attn, cmap="viridis", aspect="auto")
    for q_step in query_indices:
        ax.axhline(q_step, color='red', alpha=0.3, linestyle='--', lw=0.8)
    ax.set_title("Standard Causal Transformer (Layer 2, Head 0) Attention Weights", fontsize=12, fontweight='bold')
    ax.set_xlabel("Key Position (0..128: Stored KV Pairs)", fontsize=10, fontweight='bold')
    ax.set_ylabel("Query Position (128..256: Probed Queries)", fontsize=10, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig("results/mqar_transformer_proof.png", dpi=200, bbox_inches='tight')
    print("Saved results/mqar_transformer_proof.png")
    plt.show()
