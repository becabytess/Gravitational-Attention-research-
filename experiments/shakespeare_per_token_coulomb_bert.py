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

seq_len = 128
batch_size = 128

def get_mlm_batch(split, b_size=batch_size, mask_prob=0.15):
    """
    Standard BERT-style 15% Masked Language Modeling:
    Randomly masks 15% of tokens uniformly across sequence.
    Known 85% tokens provide bidirectional physical context.
    """
    src = train_data if split == "train" else val_data
    ix = torch.randint(0, len(src) - seq_len, (b_size,), device=device)
    offsets = torch.arange(seq_len, device=device).unsqueeze(0)
    targets = src[ix.unsqueeze(1) + offsets]

    mask = torch.rand((b_size, seq_len), device=device) < mask_prob
    mask[:, 0] = False
    mask[:, -1] = False
    return targets, mask


# ==========================================
# 2. PER-TOKEN QUERY COULOMB ROUND (QUADRATIC T x T, SINGLE SPHERE S^191)
# ==========================================
class PerTokenCoulombRound(nn.Module):
    """
    Per-Token Query Coulomb Field on Single Sphere S^(D-1):
      - Every particle i projects its OWN query q_i
      - Queries full sequence (T x T) to find its PERSONAL attractor T*_i
      - Token 15 (between 'h' and 't') directly attends to 'h' and 't' as its attractor!
      - Coulomb Force (Pull/Push): F = (q_i * Q / d^2) * (t*_i - x_i)
      - Hooke Spring: F_spring = kappa * (u - x)
    """
    def __init__(self, dim=192, temp=10.0, eps=0.1):
        super().__init__()
        self.dim = dim
        self.temp = temp
        self.eps = eps

        self.query_gen = nn.Linear(dim, dim)
        self.attractor_charge = nn.Parameter(torch.tensor(1.0))
        self.eta_e = nn.Parameter(torch.tensor(0.40))

        self.norm = nn.LayerNorm(dim)
        self.mlp  = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x, u, kappa, charge):
        # x, u: (B, T, D) on S^(D-1)
        # kappa, charge: (B, T, 1)
        B, T, D = x.shape

        # 1. Per-Token Queries: Every particle projects its own probe!
        q = F.normalize(self.query_gen(x), dim=-1) # (B, T, D)

        # 2. Full T x T Attention: Token i finds its personal attractor T*_i
        match_scores = torch.matmul(q, x.transpose(-1, -2)) * self.temp # (B, T, T)
        match_weights = F.softmax(match_scores, dim=-1)                  # (B, T, T)
        t_star = F.normalize(torch.matmul(match_weights, x), dim=-1)     # (B, T, D) - Personal Attractor!

        # 3. Personal Coulomb Force
        cos_sim = (x * t_star).sum(dim=-1, keepdim=True)                # (B, T, 1)
        dist_sq = (2.0 - 2.0 * cos_sim).clamp(min=0.0) + self.eps
        coulomb_coupling = (charge * self.attractor_charge) / dist_sq   # (B, T, 1)
        force_coulomb = coulomb_coupling * (t_star - x)                 # (B, T, D)

        # 4. Hooke Spring
        force_spring = kappa * (u - x)

        # 5. Physical Step
        f_net = self.eta_e * force_coulomb + force_spring
        x_moved = F.normalize(x + f_net, dim=-1)

        # 6. Residual Token MLP
        x_out = F.normalize(x_moved + 0.25 * self.mlp(self.norm(x_moved)), dim=-1)
        return x_out


# ==========================================
# 3. PER-TOKEN QUERY COULOMB BERT MODEL
# ==========================================
class PerTokenCoulombBert(nn.Module):
    def __init__(self, vocab_size, dim=192, max_seq_len=128, default_rounds=6):
        super().__init__()
        self.dim = dim
        self.default_rounds = default_rounds
        self.max_seq_len = max_seq_len

        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.mask_particle = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.kappa_proj = nn.Linear(dim, 1)
        self.charge_proj = nn.Linear(dim, 1)

        self.round = PerTokenCoulombRound(dim=dim)

        self.head_norm = nn.LayerNorm(dim * 2)
        self.lm_head   = nn.Linear(dim * 2, vocab_size)

    def forward(self, input_ids, mask, num_rounds=None):
        if num_rounds is None:
            num_rounds = self.default_rounds
        B, T = input_ids.shape
        pos = torch.arange(0, T, device=input_ids.device).unsqueeze(0)

        tok_embeds = self.tok_emb(input_ids)
        mask_exp = self.mask_particle.expand(B, T, self.dim)
        raw_tokens = torch.where(mask.unsqueeze(-1), mask_exp, tok_embeds)
        raw = raw_tokens + self.pos_emb(pos)

        u = F.normalize(raw, dim=-1)
        x = u.clone()

        base_kappa = torch.sigmoid(8.0 * self.kappa_proj(raw))
        stiffness = torch.where(mask.unsqueeze(-1), base_kappa, torch.full_like(base_kappa, 0.95))
        charge = torch.tanh(4.0 * self.charge_proj(raw))

        for _ in range(num_rounds):
            x = self.round(x, u, stiffness, charge)

        features = torch.cat([x, u], dim=-1)
        return self.lm_head(self.head_norm(features))

    @torch.no_grad()
    def infill_text(self, sentence_with_masks, num_rounds=6):
        """
        Interactive Infilling:
        Replaces each '[?]' with the model's physically settled character prediction.
        """
        self.eval()
        tokens = []
        mask_positions = []
        parts = sentence_with_masks.split("[?]")
        
        char_idx = 0
        for p_idx, part in enumerate(parts):
            for ch in part:
                tokens.append(stoi.get(ch, 0))
                char_idx += 1
            if p_idx < len(parts) - 1:
                mask_positions.append(char_idx)
                tokens.append(0) # dummy token for mask
                char_idx += 1

        T = min(len(tokens), self.max_seq_len)
        input_ids = torch.tensor(tokens[:T], dtype=torch.long, device=device).unsqueeze(0)
        mask_t = torch.zeros((1, T), dtype=torch.bool, device=device)
        for pos in mask_positions:
            if pos < T:
                mask_t[0, pos] = True

        logits = self(input_ids, mask_t, num_rounds=num_rounds)
        preds = logits.argmax(dim=-1)[0].tolist()

        # Reconstruct string
        out = ""
        for p_idx, part in enumerate(parts):
            out += part
            if p_idx < len(parts) - 1:
                pos = mask_positions[p_idx]
                if pos < T:
                    out += itos[preds[pos]]
                else:
                    out += "?"
        return out


# ==========================================
# 4. TRAIN PER-TOKEN COULOMB BERT (10 EPOCHS)
# ==========================================
if __name__ == "__main__":
    model = PerTokenCoulombBert(vocab_size, dim=192, max_seq_len=seq_len).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel Parameters : {total_params:,} (Per-Token Query Coulomb BERT | S^191)")
    print(f"Attention Mode   : Full T x T (128x128) | Personal Attractor T*_i per Token | Epochs: 10\n")

    epochs = 10
    steps_per_epoch = 250
    val_steps = 40

    optimizer = torch.optim.AdamW(model.parameters(), lr=3.5e-3, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * steps_per_epoch, eta_min=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))

    telemetry = {
        "experiment": "Per-Token Query Coulomb BERT MLM (Single Sphere S^191)",
        "parameters": total_params,
        "epochs": epochs,
        "epoch_metrics": []
    }

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0

        for _ in range(steps_per_epoch):
            targets, mask = get_mlm_batch("train")
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                logits = model(targets, mask, num_rounds=6)
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
                    logits = model(targets, mask, num_rounds=6)
                    loss = F.cross_entropy(logits[mask], targets[mask])
                val_loss += loss.item()
                val_acc += logits[mask].argmax(dim=-1).eq(targets[mask]).float().mean().item()

        v_loss = val_loss / val_steps
        v_acc = 100.0 * (val_acc / val_steps)
        epoch_sec = time.time() - t0

        telemetry["epoch_metrics"].append({
            "epoch": epoch, "train_loss": round(train_loss / steps_per_epoch, 4),
            "val_loss": round(v_loss, 4), "masked_token_acc": round(v_acc, 2),
            "time_sec": round(epoch_sec, 1)
        })

        q_val = model.round.attractor_charge.item()
        print(f"Epoch {epoch:02d}/10 ({epoch_sec:.1f}s) | Train Loss: {train_loss/steps_per_epoch:.3f} | "
              f"Val Loss: {v_loss:.3f} | Masked Token Acc: {v_acc:.2f}% | [Q: {q_val:.2f}]")

    # SAVE TELEMETRY
    json_path = "results/per_token_coulomb_bert_telemetry.json"
    with open(json_path, "w") as f:
        json.dump(telemetry, f, indent=2)
    print(f"\nSaved raw telemetry to {json_path}")

    # ==========================================
    # 5. INTERACTIVE INFILLING DEMO
    # ==========================================
    print("\n" + "=" * 60)
    print("DEMO: PER-TOKEN COULOMB INFILLING (PHYSICAL SETTLEMENT)")
    print("=" * 60)

    test_sentences = [
        "ROMEO:\nWh[?]t lady is th[?]t which doth enrich the hand",
        "JULIET:\nO Rom[?]o, Romeo! wheref[?]re art thou Romeo?",
        "To b[?], or not to be, th[?]t is the question:",
        "My cr[?]wn is in my he[?]rt, not on my head;"
    ]

    for s in test_sentences:
        filled = model.infill_text(s)
        print(f"\n[Masked Input] : {s}")
        print(f"[Field Settled]: {filled}")
        print("-" * 50)
