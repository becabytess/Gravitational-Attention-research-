import time
import urllib.request
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

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

seq_len = 128
batch_size = 128

def get_mlm_batch(split, b_size=batch_size, mask_prob=0.15):
    """
    Standard BERT-style 15% Masked Language Modeling:
    Randomly masks 15% of tokens uniformly across sequence.
    Known 85% tokens act as physical anchor particles.
    """
    src = train_data if split == "train" else val_data
    ix = torch.randint(0, len(src) - seq_len, (b_size,), device=device)
    offsets = torch.arange(seq_len, device=device).unsqueeze(0)
    targets = src[ix.unsqueeze(1) + offsets] # (B, seq_len)

    # 15% independent random mask across sequence
    mask = torch.rand((b_size, seq_len), device=device) < mask_prob
    # Keep first and last characters unmasked as stable boundary condition
    mask[:, 0] = False
    mask[:, -1] = False
    return targets, mask


# ==========================================
# 2. MULTI-CENTROID PHYSICAL TRAJECTORY BLOCK
# ==========================================
class CentroidTrajectoryLMBlock(nn.Module):
    """
    Physical Multi-Centroid Dynamics for Bidirectional Language Modeling:
      - 4 Orthogonal Subspaces (G=4), each on S^47
      - K=8 semantic centroid attractors per subspace (vowels, consonants, syntax, dialogue)
      - Learned Token Mass: Suppresses uninformative/masked tokens from forming centroids
      - Hooke Springs: Stiff on known tokens (anchors), zero on masked tokens (free to settle)
      - 2D Trajectory CNN across (Rounds, Tokens) to integrate momentum & relaxation path
    """
    def __init__(self, dim=192, num_groups=4, num_centroids=8, num_rounds=6):
        super().__init__()
        self.dim = dim
        self.G = num_groups
        self.d_g = dim // num_groups # 48
        self.K = num_centroids # 8
        self.R = num_rounds

        # Base queries for the K attractors in each group: (1, G, K, d_g)
        self.query_base = nn.Parameter(torch.randn(1, self.G, self.K, self.d_g) * 0.2)
        self.query_mod = nn.Linear(self.d_g, self.K * self.d_g)

        self.scale = nn.Parameter(torch.ones(1, self.G, 1, 1) * 8.0)
        self.eta_e = nn.Parameter(torch.full((self.G,), 0.40))

        # Learned token mass (saliency prior): full token state -> group mass
        self.token_mass = nn.Linear(dim, self.G)
        nn.init.zeros_(self.token_mass.weight)
        nn.init.constant_(self.token_mass.bias, 1.5)

        # Trajectory CNN across (Rounds, Sequence Length)
        self.traj_cnn = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=(3, 3), padding=(1, 1), groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=(num_rounds, 3), padding=(0, 1), groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        self.traj_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x, u, kappa, charge, is_masked=None):
        B, G, T, d_g = x.shape
        K, R = self.K, self.R
        transitions = []

        for r in range(R):
            # 1. Modulate centroid queries by subspace mean
            group_mean = x.mean(dim=2) # (B, G, d_g)
            q_mod = self.query_mod(group_mean).view(B, G, K, d_g)
            q = F.normalize(self.query_base + q_mod, dim=-1) # (B, G, K, d_g)

            # 2. Learned Token Mass (Masked tokens receive heavy negative prior)
            x_flat_r = x.permute(0, 2, 1, 3).reshape(B, T, self.dim)
            mass_logits = self.token_mass(x_flat_r) # (B, T, G)
            if is_masked is not None:
                # Force masked positions to have zero mass so they don't corrupt centroids
                mass_logits = mass_logits - 100.0 * is_masked.unsqueeze(-1).float()
            mass = F.logsigmoid(mass_logits).permute(0, 2, 1).unsqueeze(2) # (B, G, 1, T)

            # 3. Gather T tokens into K centroids on S^(d_g-1)
            scale = torch.clamp(self.scale, min=1.0, max=25.0)
            gather_scores = scale * torch.matmul(q, x.transpose(-1, -2)) + mass # (B, G, K, T)
            centroid_weights = F.softmax(gather_scores, dim=-1)
            centroids = F.normalize(torch.matmul(centroid_weights, x), dim=-1) # (B, G, K, d_g)

            # 4. Pull tokens toward matching centroids
            pull_scores = scale * torch.matmul(x, q.transpose(-1, -2)) # (B, G, T, K)
            pull_weights = F.softmax(pull_scores, dim=-1)
            assigned_centroid = torch.matmul(pull_weights, centroids) # (B, G, T, d_g)

            # 5. Coulomb Attraction/Repulsion + Hooke Springs
            force_coulomb = charge * (assigned_centroid - x)
            force_spring = kappa * (u - x)

            eta = self.eta_e.view(1, G, 1, 1)
            delta_x = eta * force_coulomb + force_spring
            x_moved = F.normalize(x + delta_x, dim=-1)

            # Record velocity transition vector
            delta_flat = (x_moved - x).permute(0, 2, 1, 3).reshape(B, T, self.dim)
            transitions.append(delta_flat)
            x = x_moved

        # 6. Trajectory CNN across velocity path: (B, D, R, T)
        traj = torch.stack(transitions, dim=2).permute(0, 3, 2, 1).contiguous()
        traj_features = self.traj_cnn(traj).squeeze(2).transpose(1, 2) # (B, T, D)

        # 7. Cross-subspace residual MLP mixing
        x_flat = x.permute(0, 2, 1, 3).reshape(B, T, self.dim)
        x_out_flat = x_flat + self.traj_proj(traj_features)
        x_mlp = x_out_flat + 0.25 * self.mlp(self.norm(x_out_flat))
        x_out = F.normalize(x_mlp.view(B, T, G, d_g).permute(0, 2, 1, 3), dim=-1)
        return x_out


# ==========================================
# 3. FULL CENTROID-TRAJECTORY BERT MLM NETWORK
# ==========================================
class CentroidTrajectoryBertMLM(nn.Module):
    def __init__(self, vocab_size, dim=192, num_groups=4, num_centroids=8, num_rounds=6, max_seq_len=128):
        super().__init__()
        self.dim = dim
        self.G = num_groups
        self.d_g = dim // num_groups
        self.max_seq_len = max_seq_len

        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.mask_particle = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        self.charge_proj = nn.Linear(dim, num_groups)

        self.block = CentroidTrajectoryLMBlock(dim, num_groups, num_centroids, num_rounds)
        self.head_norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size)

    def forward(self, input_ids, mask):
        B, T = input_ids.shape
        pos = torch.arange(0, T, device=input_ids.device).unsqueeze(0)

        # 1. Embed tokens with [MASK] token substitution
        tok_embeds = self.tok_emb(input_ids)
        mask_exp = self.mask_particle.expand(B, T, self.dim)
        raw_tokens = torch.where(mask.unsqueeze(-1), mask_exp, tok_embeds)
        raw = raw_tokens + self.pos_emb(pos)

        # 2. Partition into 4 unit hyperspheres S^47
        u = F.normalize(raw.view(B, T, self.G, self.d_g).permute(0, 2, 1, 3), dim=-1)
        x = u.clone()

        # 3. Physical Boundary Conditions:
        # Known tokens: stiff spring anchor (kappa = 0.85) to lock context
        # Masked tokens: zero spring anchor (kappa = 0.00) free to settle into energy minimum!
        kappa_val = torch.where(mask.unsqueeze(1).unsqueeze(-1), 0.0, 0.85) # (B, 1, T, 1)
        charge = torch.tanh(3.0 * self.charge_proj(raw)).permute(0, 2, 1).unsqueeze(-1) # (B, G, T, 1)

        # 4. Multi-round physical relaxation
        x_settled = self.block(x, u, kappa_val, charge, is_masked=mask)

        # 5. Project to vocabulary logits
        x_flat = x_settled.permute(0, 2, 1, 3).reshape(B, T, self.dim)
        logits = self.lm_head(self.head_norm(x_flat))
        return logits

    @torch.no_grad()
    def infill_text(self, sentence_with_masks):
        """
        Interactive Infilling Demo:
        Replaces each '[?]' with the physically settled character prediction.
        """
        self.eval()
        tokens = []
        mask_flags = []
        parts = sentence_with_masks.split("[?]")
        for p_idx, part in enumerate(parts):
            for ch in part:
                tokens.append(stoi.get(ch, 0))
                mask_flags.append(False)
            if p_idx < len(parts) - 1:
                tokens.append(0) # dummy token
                mask_flags.append(True)

        T = min(len(tokens), self.max_seq_len)
        input_ids = torch.tensor(tokens[:T], dtype=torch.long, device=device).unsqueeze(0)
        mask_t = torch.tensor(mask_flags[:T], dtype=torch.bool, device=device).unsqueeze(0)

        logits = self(input_ids, mask_t)
        preds = logits.argmax(dim=-1)[0].tolist()

        result = []
        for t in range(T):
            if mask_flags[t]:
                result.append(itos[preds[t]])
            else:
                result.append(sentence_with_masks_chars(sentence_with_masks, t, mask_flags))
        return "".join(result)

def sentence_with_masks_chars(s, idx, mask_flags):
    # Helper to reconstruct original characters at non-masked positions
    chars_list = []
    parts = s.split("[?]")
    for p_idx, part in enumerate(parts):
        for ch in part:
            chars_list.append(ch)
        if p_idx < len(parts) - 1:
            chars_list.append("[?]")
    return chars_list[idx] if idx < len(chars_list) else ""


# ==========================================
# 4. TRAIN CENTROID-TRAJECTORY BERT MLM (10 EPOCHS)
# ==========================================
if __name__ == "__main__":
    model = CentroidTrajectoryBertMLM(vocab_size, dim=192, num_groups=4, num_centroids=8, num_rounds=6, max_seq_len=seq_len).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel Parameters : {total_params:,} | Centroid-Trajectory BERT MLM (4 Spheres S^47, K=8 Centroids)")
    print(f"Task Config      : 15% Random Masking | Bidirectional Physical Relaxation | Epochs: 10\n")

    epochs = 10
    steps_per_epoch = 250
    val_steps = 40

    optimizer = torch.optim.AdamW(model.parameters(), lr=3.5e-3, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * steps_per_epoch, eta_min=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))

    telemetry = {
        "experiment": "Centroid-Trajectory BERT MLM (Tiny Shakespeare)",
        "parameters": total_params,
        "epochs": epochs,
        "epoch_metrics": []
    }

    t_start = time.time()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0

        for _ in range(steps_per_epoch):
            targets, mask = get_mlm_batch("train")
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                logits = model(targets, mask)
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
                targets, mask = get_mlm_batch("val", b_size=256)
                with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                    logits = model(targets, mask)
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
            "masked_accuracy": round(v_acc, 2),
            "time_sec": round(epoch_sec, 1)
        })

        print(f"Epoch {epoch:02d}/10 ({epoch_sec:.1f}s) | Train Loss: {train_loss/steps_per_epoch:.3f} | "
              f"Val Loss: {v_loss:.3f} | Masked Acc: {v_acc:.2f}%")

    total_time = time.time() - t_start
    print(f"\nTraining completed in {total_time:.1f}s!")

    # SAVE NUMERICAL TELEMETRY
    json_path = "results/centroid_trajectory_shakespeare_mlm.json"
    with open(json_path, "w") as f:
        json.dump(telemetry, f, indent=2)
    print(f"Saved raw telemetry to {json_path}")

    # ==========================================
    # 5. INTERACTIVE BIDIRECTIONAL INFILLING DEMO
    # ==========================================
    print("\n" + "=" * 60)
    print("DEMO: BIDIRECTIONAL MASKED INFILLING (PHYSICAL SETTLEMENT)")
    print("=" * 60)

    test_sentences = [
        "ROMEO:\nWh[?]t lady is th[?]t which doth enrich the hand",
        "JULIET:\nO Rom[?]o, Romeo! wheref[?]re art thou Romeo?",
        "To b[?], or not to be, th[?]t is the question:",
        "My cr[?]wn is in my he[?]rt, not on my head;"
    ]

    for s in test_sentences:
        try:
            filled = model.infill_text(s)
            print(f"\n[Masked Input] : {s}")
            print(f"[Field Settled]: {filled}")
            print("-" * 50)
        except Exception as e:
            print(f"Demo error: {e}")
