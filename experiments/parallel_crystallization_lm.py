import time
import urllib.request
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Ensure results directory exists
os.makedirs("results", exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on device: {device}")

# ==========================================
# 1. LOAD TINY SHAKESPEARE INTO VRAM
# ==========================================
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
text = urllib.request.urlopen(url).read().decode("utf-8")
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])
data = torch.tensor(encode(text), dtype=torch.long, device=device)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
print(f"Tiny Shakespeare loaded | Length: {len(text):,} chars | Vocab: {vocab_size}")

seq_len = 64
batch_size = 128

def get_masked_batch(split, b_size=batch_size, mask_ratio=0.5):
    """
    Creates parallel infilling batches:
    A prefix prompt is given, and the rest is MASKED to be generated in parallel!
    """
    src = train_data if split == "train" else val_data
    ix = torch.randint(0, len(src) - seq_len, (b_size,), device=device)
    offsets = torch.arange(seq_len, device=device).unsqueeze(0)
    targets = src[ix.unsqueeze(1) + offsets] # Ground truth (B, seq_len)

    # Prompt length: randomly between 12 and 24 characters
    prompt_lens = torch.randint(12, 25, (b_size, 1), device=device)
    pos_idx = torch.arange(seq_len, device=device).unsqueeze(0)
    mask = pos_idx >= prompt_lens # True for tokens to be generated in parallel!

    return targets, mask


# ==========================================
# 2. BIDIRECTIONAL CRYSTALLIZATION ROUND (ZERO CAUSAL MASK)
# ==========================================
class CrystallizationRound(nn.Module):
    """
    Bidirectional Coulomb Field Relaxation on K Orthogonal Hyperspheres:
      - NO causal temporal mask: All particles interact simultaneously!
      - Fixed Prompt Anchors (kappa ≈ 1.0) exert strong Coulomb fields
      - Free Mask Particles (low initial kappa) migrate & settle into word coordinates
      - Over R rounds, particles crystallize into grammatically coherent equilibrium
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
        self.attractor_charge = nn.Parameter(torch.ones(self.K))
        self.eta_e = nn.Parameter(torch.full((self.K,), 0.40))

        # Cross-subspace manifold mixer
        self.norm = nn.LayerNorm(dim)
        self.mlp  = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x, u, kappa, charge):
        # x, u: (B, K, T, d_k) on S^(d_k - 1)
        # kappa: (B, K, T, 1) in (0, 1)
        # charge: (B, K, T, 1) in (-1, 1)
        B, K, T, d_k = x.shape

        # 1. Global Field Average (Global Semantic State)
        space_avg = F.normalize(x.mean(dim=2, keepdim=True), dim=-1)
        probe_flat = F.normalize(space_avg + x, dim=-1).permute(0, 2, 1, 3).reshape(B, T, self.dim)

        # 2. Subspace queries
        q = F.normalize(self.query_gen(probe_flat.mean(dim=1)).view(B, K, 1, d_k), dim=-1)

        # 3. Retrieve central attractor T*_k across ALL tokens (Bidirectional!)
        match_scores = torch.matmul(q, x.transpose(-1, -2)) * self.temp # (B, K, 1, T)
        t_star = F.normalize(torch.matmul(F.softmax(match_scores, dim=-1), x), dim=-1) # (B, K, 1, d_k)

        # 4. Multi-Center Coulomb Electrostatic Force (Signed Pull & Push)
        cos_sim = (x * t_star).sum(dim=-1, keepdim=True)
        dist_sq = (2.0 - 2.0 * cos_sim).clamp(min=0.0) + self.eps
        Q_k = self.attractor_charge.view(1, K, 1, 1)
        force_coulomb = ((charge * Q_k) / dist_sq) * (t_star - x)

        # 5. Hooke Spring Force tethered to anchor U
        force_spring = kappa * (u - x)

        # 6. Physical Relaxation Step
        eta_k = self.eta_e.view(1, K, 1, 1)
        x_moved = F.normalize(x + eta_k * force_coulomb + force_spring, dim=-1)

        # 7. Cross-Subspace Token Interaction MLP
        x_flat = x_moved.permute(0, 2, 1, 3).reshape(B, T, self.dim)
        x_mlp  = x_flat + 0.25 * self.mlp(self.norm(x_flat))
        x_out  = F.normalize(x_mlp.view(B, T, K, d_k).permute(0, 2, 1, 3), dim=-1)

        return x_out


# ==========================================
# 3. PARALLEL PARTICLE CRYSTALLIZATION GENERATOR
# ==========================================
class ParticleCrystallizationGenerator(nn.Module):
    def __init__(self, vocab_size, dim=192, num_subspaces=4, max_seq_len=64, default_rounds=6):
        super().__init__()
        self.dim = dim
        self.K = num_subspaces
        self.d_k = dim // num_subspaces
        self.default_rounds = default_rounds
        self.max_seq_len = max_seq_len

        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        # Special learned Blank/Mask Particle Coordinate on S^(D-1)
        self.blank_particle = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        self.kappa_proj = nn.Linear(dim, num_subspaces)
        self.charge_proj = nn.Linear(dim, num_subspaces)

        self.round = CrystallizationRound(dim=dim, num_subspaces=num_subspaces)

        self.head_norm = nn.LayerNorm(dim * 2)
        self.lm_head   = nn.Linear(dim * 2, vocab_size)

    def forward(self, input_ids, mask, num_rounds=None):
        if num_rounds is None:
            num_rounds = self.default_rounds
        B, T = input_ids.shape
        pos = torch.arange(0, T, device=input_ids.device).unsqueeze(0)

        # Embed known tokens
        tok_embeds = self.tok_emb(input_ids)
        blank_exp = self.blank_particle.expand(B, T, self.dim)

        # Where mask=True, replace embedding with the blank particle coordinate!
        raw_tokens = torch.where(mask.unsqueeze(-1), blank_exp, tok_embeds)
        raw = raw_tokens + self.pos_emb(pos)

        # Initial anchor coordinates on S^47
        u = F.normalize(raw.view(B, T, self.K, self.d_k).permute(0, 2, 1, 3), dim=-1)
        x = u.clone()

        u_flat = u.permute(0, 2, 1, 3).reshape(B, T, self.dim)
        # Prompt tokens have higher stiffness; mask tokens are more fluid
        base_kappa = torch.sigmoid(8.0 * self.kappa_proj(u_flat)).permute(0, 2, 1).unsqueeze(-1)
        # Lock prompt stiffness near 0.95 so prompt does not deform
        prompt_stiffness = torch.where(mask.unsqueeze(1).unsqueeze(-1), base_kappa, torch.full_like(base_kappa, 0.95))

        charge = torch.tanh(4.0 * self.charge_proj(u_flat)).permute(0, 2, 1).unsqueeze(-1)

        # PHYSICAL RELAXATION OVER R ROUNDS (All particles settle in parallel!)
        for _ in range(num_rounds):
            x = self.round(x, u, prompt_stiffness, charge)

        x_final = x.permute(0, 2, 1, 3).reshape(B, T, self.dim)
        features = torch.cat([x_final, u_flat], dim=-1)
        logits = self.lm_head(self.head_norm(features)) # Emits ALL T positions at once!
        return logits

    @torch.no_grad()
    def crystallize_in_parallel(self, prompt_text="ROMEO:\n", target_total_len=64, num_rounds=6):
        """
        ONE FORWARD PASS GENERATION:
        Emits the ENTIRE continuation in parallel in a single forward pass!
        """
        self.eval()
        prompt_tokens = encode(prompt_text)
        p_len = len(prompt_tokens)
        assert p_len < target_total_len

        # Pad with dummy tokens
        full_tokens = prompt_tokens + [0] * (target_total_len - p_len)
        input_ids = torch.tensor(full_tokens, dtype=torch.long, device=device).unsqueeze(0)

        # Mask is False for prompt, True for all parallel target positions
        mask = torch.zeros((1, target_total_len), dtype=torch.bool, device=device)
        mask[:, p_len:] = True

        t_start = time.time()
        # ONE SINGLE FORWARD PASS!
        logits = self(input_ids, mask, num_rounds=num_rounds)
        generation_time_ms = (time.time() - t_start) * 1000

        # Parallel argmax across all masked positions simultaneously!
        predicted_ids = logits.argmax(dim=-1)[0].tolist()

        # Combine fixed prompt with the crystallized tokens
        final_chars = [itos[tok] for tok in prompt_tokens] + [itos[predicted_ids[t]] for t in range(p_len, target_total_len)]
        generated_text = "".join(final_chars)
        return generated_text, generation_time_ms


# ==========================================
# 4. TRAIN THE PARALLEL GENERATOR (10 EPOCHS)
# ==========================================
if __name__ == "__main__":
    model = ParticleCrystallizationGenerator(vocab_size, dim=192, num_subspaces=4, max_seq_len=seq_len, default_rounds=6).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel Parameters : {total_params:,} (Parallel Particle Crystallization Generator)")
    print(f"Generation Mode  : Non-Autoregressive | All Tokens Settle in ONE Pass | Epochs: 10\n")

    epochs = 10
    steps_per_epoch = 250
    val_steps = 40

    optimizer = torch.optim.AdamW(model.parameters(), lr=3.5e-3, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * steps_per_epoch, eta_min=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))

    telemetry = {
        "experiment": "Parallel Particle Crystallization Generator (Tiny Shakespeare)",
        "parameters": total_params,
        "epochs": epochs,
        "epoch_metrics": []
    }

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0

        for _ in range(steps_per_epoch):
            targets, mask = get_masked_batch("train")
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                logits = model(targets, mask, num_rounds=6)
                # Compute loss ONLY on the parallel-generated masked positions!
                loss = F.cross_entropy(logits[mask], targets[mask])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss, val_acc = 0.0, 0.0
        with torch.no_grad():
            for _ in range(val_steps):
                targets, mask = get_masked_batch("val", b_size=256)
                with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                    logits = model(targets, mask, num_rounds=6)
                    loss = F.cross_entropy(logits[mask], targets[mask])
                val_loss += loss.item()
                val_acc += logits[mask].argmax(dim=-1).eq(targets[mask]).float().mean().item()

        v_loss = val_loss / val_steps
        v_acc = 100.0 * (val_acc / val_steps)
        epoch_sec = time.time() - t0

        telemetry["epoch_metrics"].append({
            "epoch": epoch,
            "train_loss": round(train_loss / steps_per_epoch, 4),
            "val_loss": round(v_loss, 4),
            "masked_token_acc": round(v_acc, 2),
            "epoch_time_sec": round(epoch_sec, 1)
        })

        print(f"Epoch {epoch:02d}/10 ({epoch_sec:.1f}s) | Train Loss: {train_loss/steps_per_epoch:.3f} | "
              f"Val Loss: {v_loss:.3f} | Masked Parallel Acc: {v_acc:.2f}%")

    # SAVE NUMERICAL TELEMETRY
    json_path = "results/parallel_crystallization_telemetry.json"
    with open(json_path, "w") as f:
        json.dump(telemetry, f, indent=2)
    print(f"\nSaved raw telemetry to {json_path}")

    # ==========================================
    # 5. LIVE DEMO: 1-PASS PARALLEL CRYSTALLIZATION
    # ==========================================
    print("\n" + "=" * 60)
    print("DEMO: 1-PASS PARALLEL GENERATION (ENTIRE CONTINUATION IN 1 STEP)")
    print("=" * 60)

    test_prompts = [
        "ROMEO:\n",
        "JULIET:\n",
        "HAMLET:\nTo be, or not ",
        "KING:\nMy crown is "
    ]

    for p in test_prompts:
        output, ms = model.crystallize_in_parallel(prompt_text=p, target_total_len=64)
        print(f"\n[Prompt]: {repr(p)}")
        print(f"[Crystallized in 1 Pass ({ms:.1f} ms)]:\n{output}")
        print("-" * 50)
