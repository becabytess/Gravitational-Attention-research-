import time
import urllib.request
import os
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

block_size = 128
batch_size = 128

def get_gpu_batch(split, b_size=batch_size):
    src = train_data if split == "train" else val_data
    ix = torch.randint(0, len(src) - block_size - 1, (b_size,), device=device)
    offsets = torch.arange(block_size, device=device).unsqueeze(0)
    return src[ix.unsqueeze(1) + offsets], src[ix.unsqueeze(1) + offsets + 1]


# ==========================================
# 2. ORTHOGONAL CAUSAL COULOMB ROUND (ZERO DECAY, STRICT CAUSALITY)
# ==========================================
class OrthogonalCausalCoulombRound(nn.Module):
    """
    Causal Coulomb Electrostatic Dynamics on K Orthogonal Hyperspheres:
      - K = 4 orthogonal spheres (dim = 192, d_k = 48 on S^47)
      - Causal Past Light-Cone (j < t) with ZERO temporal decay
      - Coulomb Force (Signed charges q_i,k in [-1, 1]):
          q_i,k * Q_k > 0: Attracts toward past attractor T*_k
          q_i,k * Q_k < 0: Repels away from past attractor T*_k
      - Hooke Springs: F_spring = kappa_k * (u_k - x_k) anchors tokens back to initial token u_k
      - Zero vector interference across orthogonal spheres
    """
    def __init__(self, dim=192, num_subspaces=4, max_seq_len=512, temp=10.0, eps=0.1):
        super().__init__()
        assert dim % num_subspaces == 0
        self.dim = dim
        self.K = num_subspaces
        self.d_k = dim // num_subspaces
        self.temp = temp
        self.eps = eps

        # Projects causal center of mass into K subspace queries
        self.query_gen = nn.Linear(dim, dim)

        # Attractor central charge Q_k (learnable scalar per subspace)
        self.attractor_charge = nn.Parameter(torch.ones(self.K))
        self.eta_e            = nn.Parameter(torch.full((self.K,), 0.40))

        # Strict past light-cone mask (j < t); (0,0) is enabled so t=0 softmax is valid (zero force)
        past_mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=-1)
        past_mask[0, 0] = True
        self.register_buffer("past_mask", past_mask)

        # Cross-subspace manifold mixer
        self.norm = nn.LayerNorm(dim)
        self.mlp  = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x, u, kappa, charge, eta_override=None, kappa_override=None):
        # x, u: (B, K, T, d_k) on S^(d_k - 1)
        # kappa: (B, K, T, 1) in (0, 1)
        # charge: (B, K, T, 1) in (-1, 1)
        B, K, T, d_k = x.shape

        # 1. Causal Center of Mass up to step t per subspace
        counts = torch.arange(1, T + 1, device=x.device, dtype=x.dtype).view(1, 1, T, 1)
        causal_com = F.normalize(torch.cumsum(x, dim=2) / counts, dim=-1)

        # 2. Subspace queries strictly from Causal Center of Mass
        com_flat = causal_com.permute(0, 2, 1, 3).reshape(B, T, self.dim)
        q = self.query_gen(com_flat).view(B, T, K, d_k).permute(0, 2, 1, 3)
        q = F.normalize(q, dim=-1)

        # 3. PURE Causal Attractor Retrieval in each subspace (j < t)
        match_scores = torch.matmul(q, x.transpose(-1, -2)) * self.temp
        match_scores = match_scores.masked_fill(~self.past_mask[:T, :T].view(1, 1, T, T), float("-inf"))
        match_weights = F.softmax(match_scores, dim=-1)
        t_star = F.normalize(torch.matmul(match_weights, x), dim=-1)
        best_past_idx = match_scores.argmax(dim=-1)

        # 4. Coulomb Electrostatic Force (Signed attraction or repulsion)
        cos_to_tstar = (x * t_star).sum(dim=-1, keepdim=True)
        dist_sq = (2.0 - 2.0 * cos_to_tstar).clamp(min=0.0) + self.eps
        Q_k = self.attractor_charge.view(1, K, 1, 1)
        force_coulomb = ((charge * Q_k) / dist_sq) * (t_star - x)

        # 5. Hooke's Law Spring Force
        k_val = kappa if kappa_override is None else kappa_override
        force_spring = k_val * (u - x)

        # 6. Combined Physical Step
        eta_k = self.eta_e.view(1, K, 1, 1) if eta_override is None else eta_override
        f_net = eta_k * force_coulomb + force_spring
        x_moved = F.normalize(x + f_net, dim=-1)

        # 7. Cross-Subspace Residual MLP
        x_flat = x_moved.permute(0, 2, 1, 3).reshape(B, T, self.dim)
        x_mlp  = x_flat + 0.25 * self.mlp(self.norm(x_flat))
        x_out  = F.normalize(x_mlp.view(B, T, K, d_k).permute(0, 2, 1, 3), dim=-1)

        return x_out, force_coulomb.norm(dim=-1), best_past_idx


# ==========================================
# 3. CAUSAL COULOMB LANGUAGE MODEL
# ==========================================
class CausalCoulombLM(nn.Module):
    def __init__(self, vocab_size, dim=192, num_subspaces=4, max_seq_len=256, default_rounds=6):
        super().__init__()
        self.dim = dim
        self.K = num_subspaces
        self.d_k = dim // num_subspaces
        self.default_rounds = default_rounds
        self.max_seq_len = max_seq_len

        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.kappa_proj = nn.Linear(dim, num_subspaces)
        self.kappa_scale = 8.0

        self.charge_proj = nn.Linear(dim, num_subspaces)
        self.charge_scale = 4.0

        self.round = OrthogonalCausalCoulombRound(
            dim=dim, num_subspaces=num_subspaces, max_seq_len=max_seq_len
        )

        self.head_norm = nn.LayerNorm(dim * 2)
        self.lm_head   = nn.Linear(dim * 2, vocab_size)

    def forward(self, idx, num_rounds=None, return_physics=False, eta_override=None, kappa_override=None):
        if num_rounds is None:
            num_rounds = self.default_rounds
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

        charge = torch.tanh(self.charge_scale * self.charge_proj(u_flat)).permute(0, 2, 1).unsqueeze(-1)

        trajectories = [x]
        coulomb_maps_k = []
        chosen_indices_k = []

        for _ in range(num_rounds):
            x, c_pull_k, best_idx_k = self.round(x, u, kappa, charge, eta_override=eta_override, kappa_override=kappa_override)
            if return_physics:
                trajectories.append(x)
                coulomb_maps_k.append(c_pull_k)
                chosen_indices_k.append(best_idx_k)

        x_final = x.permute(0, 2, 1, 3).reshape(B, T, self.dim)
        features = torch.cat([x_final, u_flat], dim=-1)
        logits = self.lm_head(self.head_norm(features))

        if return_physics:
            return logits, u, kappa.squeeze(-1), charge.squeeze(-1), trajectories, coulomb_maps_k, chosen_indices_k
        return logits

    @torch.no_grad()
    def generate(self, prompt="ROMEO:\n", max_new_tokens=200, temperature=0.8):
        self.eval()
        idx = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_seq_len:]
            logits = self(idx_cond, num_rounds=self.default_rounds)[:, -1, :]
            probs = F.softmax(logits / temperature, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return decode(idx[0].tolist())


# ==========================================
# 4. VERIFY CAUSALITY & TRAIN
# ==========================================
if __name__ == "__main__":
    model = CausalCoulombLM(vocab_size, dim=192, num_subspaces=4, max_seq_len=block_size, default_rounds=6).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # STRICT CAUSALITY VERIFICATION
    model.eval()
    with torch.no_grad():
        s1 = torch.randint(0, vocab_size, (1, block_size), device=device)
        s2 = s1.clone()
        s2[:, 64:] = torch.randint(0, vocab_size, (1, block_size - 64), device=device)
        leak = (model(s1)[:, :64, :] - model(s2)[:, :64, :]).abs().max().item()

    print(f"\nModel Parameters : {total_params:,} (Orthogonal Coulomb LM | 4 Spheres S^47)")
    print(f"Causality Check  : Max leak from future tokens (t >= 64) into past (t < 64) = {leak:.2e}")
    print(f"Benchmark Target : Beat Gravity's Val Loss: 1.605 | Acc: 52.57% at Epoch 10\n")

    epochs = 10
    steps_per_epoch = 250
    val_steps = 40

    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-3, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * steps_per_epoch, eta_min=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0

        for _ in range(steps_per_epoch):
            xb, yb = get_gpu_batch("train")
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                logits = model(xb, num_rounds=6)
                loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_loss += loss.item()

        model.eval()
        val_loss, val_acc = 0.0, 0.0
        with torch.no_grad():
            for _ in range(val_steps):
                xb, yb = get_gpu_batch("val", b_size=256)
                with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                    logits = model(xb, num_rounds=6)
                    loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
                val_loss += loss.item()
                val_acc += logits.argmax(dim=-1).eq(yb).float().mean().item()

        v_loss = val_loss / val_steps
        v_acc  = 100.0 * (val_acc / val_steps)
        q_vals = [f"{q:.2f}" for q in model.round.attractor_charge.detach().cpu().numpy()]
        print(f"Epoch {epoch:02d}/10 ({time.time()-t0:.1f}s) | Train: {train_loss/steps_per_epoch:.3f} | "
              f"Val Loss: {v_loss:.3f} (Acc: {v_acc:.2f}%) | Attractor Q: [{','.join(q_vals)}]")

    # ==========================================
    # 5. PHYSICS ABLATIONS AT EPOCH 10
    # ==========================================
    @torch.no_grad()
    def eval_ablation_lm(eta_override=None, kappa_override=None):
        model.eval()
        v_loss, v_acc = 0.0, 0.0
        for _ in range(val_steps):
            xb, yb = get_gpu_batch("val", b_size=256)
            with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                logits = model(xb, num_rounds=6, eta_override=eta_override, kappa_override=kappa_override)
                loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
            v_loss += loss.item()
            v_acc += logits.argmax(dim=-1).eq(yb).float().mean().item()
        return v_loss / val_steps, 100.0 * (val_acc / val_steps)

    full_loss, full_acc   = eval_ablation_lm()
    noe_loss, noe_acc     = eval_ablation_lm(eta_override=0.0)
    nospr_loss, nospr_acc = eval_ablation_lm(kappa_override=0.0)

    print("\n=== EPOCH 10 COULOMB VS. GRAVITY LM VERDICT ===")
    print(f"1. Full Coulomb LM (4 Spheres)       : Val Loss {full_loss:.3f} | Acc: {full_acc:.2f}%")
    print(f"   Gravity Baseline at Epoch 10       : Val Loss 1.605 | Acc: 52.57%")
    print(f"2. Electric Force Off (eta = 0)       : Val Loss {noe_loss:.3f} | Acc: {noe_acc:.2f}% (Drop: -{full_acc - noe_acc:.2f}%)")
    print(f"3. Springs Off (kappa = 0)            : Val Loss {nospr_loss:.3f} | Acc: {nospr_acc:.2f}% (Drop: -{full_acc - nospr_acc:.2f}%)")

    # ==========================================
    # 6. GENERATE SHAKESPEARE SAMPLE
    # ==========================================
    print("\n=== GENERATED COULOMB SHAKESPEARE SAMPLE ===")
    sample = model.generate(prompt="ROMEO:\n", max_new_tokens=250, temperature=0.8)
    print(sample)
    print("=" * 44)
