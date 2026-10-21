import os
import time
import urllib.request
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
# 2. PURE ORTHOGONAL CAUSAL ROUND (ZERO DECAY)
# ==========================================
class PureOrthogonalCausalRound(nn.Module):
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

        # Uniform initialization across all K subspaces
        self.log_mass = nn.Parameter(torch.zeros(self.K))
        self.eta      = nn.Parameter(torch.full((self.K,), 0.40))

        # Strict past light-cone mask (j < t)
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

    def forward(self, x, u, kappa):
        B, K, T, d_k = x.shape

        # 1. Causal Center of Mass per subspace
        counts = torch.arange(1, T + 1, device=x.device, dtype=x.dtype).view(1, 1, T, 1)
        causal_com = F.normalize(torch.cumsum(x, dim=2) / counts, dim=-1)

        # 2. Subspace queries strictly from the Causal Center of Mass
        com_flat = causal_com.permute(0, 2, 1, 3).reshape(B, T, self.dim)
        q = self.query_gen(com_flat).view(B, T, K, d_k).permute(0, 2, 1, 3)
        q = F.normalize(q, dim=-1)

        # 3. Pure Causal Attractor Retrieval in each subspace (j < t) -- ZERO DECAY
        match_scores = torch.matmul(q, x.transpose(-1, -2)) * self.temp
        match_scores = match_scores.masked_fill(~self.past_mask[:T, :T].view(1, 1, T, T), float("-inf"))

        match_weights = F.softmax(match_scores, dim=-1)
        t_star = F.normalize(torch.matmul(match_weights, x), dim=-1)
        best_past_idx = match_scores.argmax(dim=-1)

        # 4. Inverse-Square Gravitational Pull toward T*_k
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
# 3. GRAVITATIONAL BLOCK (UNSHARED PHYSICAL REGIME)
# ==========================================
class GravitationalBlock(nn.Module):
    """
    A single independent Gravitational Block containing:
      - Its own dynamic spring constant projection: kappa_proj
      - Its own physical round with independent (W_q, Masses, Etas, MLP)
      - Relaxes tokens for `rounds_per_block` iterations
    """
    def __init__(self, dim=192, num_subspaces=4, max_seq_len=256, rounds_per_block=3):
        super().__init__()
        self.dim = dim
        self.K = num_subspaces
        self.d_k = dim // num_subspaces
        self.rounds_per_block = rounds_per_block

        self.kappa_proj = nn.Linear(dim, num_subspaces)
        self.kappa_scale = 8.0

        # One shared round inside this block, unrolled rounds_per_block times
        self.round = PureOrthogonalCausalRound(
            dim=dim, num_subspaces=num_subspaces, max_seq_len=max_seq_len
        )

    def forward(self, u_in, return_physics=False, eta_override=None, kappa_override=None):
        B, K, T, d_k = u_in.shape
        # The input equilibrium manifold from the previous block becomes the anchor for this block!
        u = u_in
        x = u.clone()

        u_flat = u.permute(0, 2, 1, 3).reshape(B, T, self.dim)
        if kappa_override is None:
            kappa = torch.sigmoid(self.kappa_scale * self.kappa_proj(u_flat)).permute(0, 2, 1).unsqueeze(-1)
        else:
            kappa = torch.full((B, self.K, T, 1), kappa_override, device=u_in.device, dtype=u.dtype)

        saved_eta = self.round.eta.data.clone()
        if eta_override is not None:
            self.round.eta.data.fill_(eta_override)

        block_trajs = [x]
        block_pulls = []
        block_idxs  = []

        # Relax for rounds_per_block iterations within this block's physical field
        for _ in range(self.rounds_per_block):
            x, g_pull, best_idx = self.round(x, u, kappa)
            if return_physics:
                block_trajs.append(x)
                block_pulls.append(g_pull)
                block_idxs.append(best_idx)

        if eta_override is not None:
            self.round.eta.data.copy_(saved_eta)

        # Geodesic residual handoff to the next block
        out = F.normalize(u_in + x, dim=-1)

        if return_physics:
            return out, kappa.squeeze(-1), block_trajs, block_pulls, block_idxs
        return out


# ==========================================
# 4. MULTI-BLOCK HIERARCHICAL GRAVITATIONAL LM
# ==========================================
class MultiBlockGravitationalLM(nn.Module):
    """
    Stacks L independent Gravitational Blocks in series.
    Each block has its own unshared physical regime and relaxes for R rounds.
    Total physical steps across depth = L x R.
    """
    def __init__(self, vocab_size, dim=192, num_subspaces=4, num_blocks=2, rounds_per_block=3, max_seq_len=256):
        super().__init__()
        self.dim = dim
        self.K = num_subspaces
        self.d_k = dim // num_subspaces
        self.num_blocks = num_blocks
        self.rounds_per_block = rounds_per_block
        self.max_seq_len = max_seq_len

        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        # L independent Gravitational Blocks
        self.blocks = nn.ModuleList([
            GravitationalBlock(
                dim=dim,
                num_subspaces=num_subspaces,
                max_seq_len=max_seq_len,
                rounds_per_block=rounds_per_block
            )
            for _ in range(num_blocks)
        ])

        self.head_norm = nn.LayerNorm(dim * 2)
        self.lm_head   = nn.Linear(dim * 2, vocab_size)

    def forward(self, idx, return_physics=False, eta_override=None, kappa_override=None):
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)

        raw = self.tok_emb(idx) + self.pos_emb(pos)
        # Initial projection onto K orthogonal unit spheres
        u = F.normalize(raw.view(B, T, self.K, self.d_k).permute(0, 2, 1, 3), dim=-1)

        all_block_kappas = []
        all_block_idxs   = []

        # Pass through the L hierarchical blocks in series
        for block in self.blocks:
            if return_physics:
                u, b_kappa, _, _, b_idxs = block(
                    u, return_physics=True, eta_override=eta_override, kappa_override=kappa_override
                )
                all_block_kappas.append(b_kappa)
                all_block_idxs.append(b_idxs)
            else:
                u = block(
                    u, return_physics=False, eta_override=eta_override, kappa_override=kappa_override
                )

        x_final = u.permute(0, 2, 1, 3).reshape(B, T, self.dim)
        counts = torch.arange(1, T + 1, device=idx.device, dtype=x_final.dtype).view(1, T, 1)
        causal_com = F.normalize(torch.cumsum(x_final, dim=1) / counts, dim=-1)
        readout = self.head_norm(torch.cat([x_final, causal_com], dim=-1))
        logits = self.lm_head(readout)

        if return_physics:
            return logits, all_block_kappas, all_block_idxs
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.75):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_seq_len:]
            logits = self(idx_cond)[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, 1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ==========================================
# 5. VERIFY CAUSALITY & TRAIN (2 BLOCKS x 3 ROUNDS = 6 TOTAL STEPS)
# ==========================================
if __name__ == "__main__":
    num_blocks = 2
    rounds_per_block = 3

    model = MultiBlockGravitationalLM(
        vocab_size=vocab_size,
        dim=192,
        num_subspaces=4,
        num_blocks=num_blocks,
        rounds_per_block=rounds_per_block,
        max_seq_len=block_size
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    model.eval()
    with torch.no_grad():
        s1 = torch.randint(0, vocab_size, (1, block_size), device=device)
        s2 = s1.clone()
        s2[:, 64:] = torch.randint(0, vocab_size, (1, block_size - 64), device=device)
        leak = (model(s1)[:, :64, :] - model(s2)[:, :64, :]).abs().max().item()

    print(f"\nModel Parameters : {total_params:,} | {num_blocks} Blocks x {rounds_per_block} Rounds ({num_blocks * rounds_per_block} Total Steps)")
    print(f"Causality Check  : Max leak from future tokens (t >= 64) into past (t < 64) = {leak:.2e}\n")

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
                logits = model(xb)
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
                    logits = model(xb)
                    val_loss += F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1)).item()
                    val_acc  += (logits.argmax(dim=-1) == yb).float().mean().item() * 100.0

        val_loss /= val_steps
        val_acc  /= val_steps

        # Print masses across all blocks
        m_reports = []
        for b_i, b in enumerate(model.blocks):
            b_masses = F.softplus(b.round.log_mass).detach().cpu().numpy()
            m_str = ",".join([f"{m:.2f}" for m in b_masses])
            m_reports.append(f"B{b_i+1}:[{m_str}]")
        print(f"Epoch {epoch:02d}/10 ({time.time()-t0:.1f}s) | Train: {train_loss/steps_per_epoch:.3f} | "
              f"Val Loss: {val_loss:.3f} (Acc: {val_acc:.2f}%) | {' | '.join(m_reports)}")


    # ==========================================
    # 6. ABLATION & SAMPLE GENERATION
    # ==========================================
    @torch.no_grad()
    def eval_ablation(eta_override=None, kappa_override=None):
        model.eval()
        total_loss, total_acc = 0.0, 0.0
        for _ in range(val_steps):
            xb, yb = get_gpu_batch("val", b_size=256)
            with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                logits = model(xb, eta_override=eta_override, kappa_override=kappa_override)
                total_loss += F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1)).item()
                total_acc  += (logits.argmax(dim=-1) == yb).float().mean().item() * 100.0
        return total_loss / val_steps, total_acc / val_steps

    full_loss, full_acc       = eval_ablation()
    nograv_loss, nograv_acc   = eval_ablation(eta_override=0.0)
    nospr_loss, nospr_acc     = eval_ablation(kappa_override=0.0)
    flatspr_loss, flatspr_acc = eval_ablation(kappa_override=0.5)

    print(f"\n=== MULTI-BLOCK ({num_blocks} BLOCKS x {rounds_per_block} ROUNDS) ABLATION RESULTS ===")
    for b_idx, b in enumerate(model.blocks):
        masses = F.softplus(b.round.log_mass).detach().cpu().numpy()
        etas   = b.round.eta.detach().cpu().numpy()
        print(f"Block {b_idx+1}:")
        for k in range(4):
            print(f"  Subspace {k+1} (48-D) | Mass M_{k+1}: {masses[k]:.4f} | Pull eta_{k+1}: {etas[k]:.4f}")
    print("-" * 72)
    print(f"1. Full Multi-Block Model                : Val Loss {full_loss:.3f} | Acc: {full_acc:.2f}%")
    print(f"2. Gravity Turned Off (eta = 0)          : Val Loss {nograv_loss:.3f} | Acc: {nograv_acc:.2f}% (Drop: -{full_acc - nograv_acc:.2f}%)")
    print(f"3. Spring Turned Off (kappa = 0)         : Val Loss {nospr_loss:.3f} | Acc: {nospr_acc:.2f}% (Drop: -{full_acc - nospr_acc:.2f}%)")
    print(f"4. Constant Spring (kappa = 0.5)         : Val Loss {flatspr_loss:.3f} | Acc: {flatspr_acc:.2f}% (Drop: -{full_acc - flatspr_acc:.2f}%)")

    print("\n=== GENERATED SHAKESPEARE SAMPLE ===")
    prompt = "ROMEO:\n"
    context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=220, temperature=0.75)[0].tolist()))
    print("=" * 40)


    # ==========================================
    # 7. VISUALIZE MULTI-BLOCK LOOKBACK HIERARCHY
    # ==========================================
    quote = "To be, or not to be, that is the question:"
    q_tensor = torch.tensor([encode(quote)], dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        logits, all_kappas, all_idxs = model(q_tensor, return_physics=True)

    L = len(quote)
    char_labels = [c if c != " " else "␣" for c in quote]

    fig, axes = plt.subplots(num_blocks, 1, figsize=(14, 3.5 * num_blocks), sharex=True)
    if num_blocks == 1:
        axes = [axes]

    for b_idx in range(num_blocks):
        ax = axes[b_idx]
        # Final round of this block
        b_r_last = all_idxs[b_idx][-1][0].cpu().numpy() # (K, L)
        b_lookback = np.zeros((4, L))
        for k in range(4):
            for t in range(1, L):
                b_lookback[k, t] = t - b_r_last[k, t]

        im = ax.imshow(b_lookback, cmap="magma", aspect="auto")
        ax.set_yticks(range(4))
        ax.set_yticklabels([f"S{k+1}" for k in range(4)], fontweight="bold")
        ax.set_title(f"Block {b_idx + 1} Spacetime Lookback Distance (t - j)", fontsize=11, fontweight="bold")

        for k in range(4):
            for t in range(1, L):
                dist = int(b_lookback[k, t])
                txt_col = "black" if dist > b_lookback.max() * 0.5 else "cyan"
                ax.text(t, k, str(dist), ha="center", va="center", color=txt_col, fontsize=7.5, fontfamily="monospace", fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="Chars Back")

    axes[-1].set_xticks(range(L))
    axes[-1].set_xticklabels(char_labels, fontsize=11, fontfamily="monospace", fontweight="bold")
    plt.suptitle("Hierarchical Multi-Block Gravitational Network: How Depth Divides Spacetime Lookback", fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig("results/multiblock_shakespeare_analysis.png", dpi=200, bbox_inches='tight')
    print("Saved results/multiblock_shakespeare_analysis.png")
    plt.show()
