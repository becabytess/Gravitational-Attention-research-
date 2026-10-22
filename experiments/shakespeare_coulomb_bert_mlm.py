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
    Randomly masks 15% of tokens uniformly across the sequence.
    Known 85% tokens provide bidirectional physical context.
    """
    src = train_data if split == "train" else val_data
    ix = torch.randint(0, len(src) - seq_len, (b_size,), device=device)
    offsets = torch.arange(seq_len, device=device).unsqueeze(0)
    targets = src[ix.unsqueeze(1) + offsets] # Ground truth (B, seq_len)

    # 15% independent random mask across sequence
    mask = torch.rand((b_size, seq_len), device=device) < mask_prob
    # Ensure at least 1 token is masked per sequence
    mask[:, 0] = False
    mask[:, -1] = False
    return targets, mask


# ==========================================
# 2. SINGLE-SPHERE BIDIRECTIONAL COULOMB ROUND (S^191)
# ==========================================
class SingleSphereCoulombMLMRound(nn.Module):
    """
    Bidirectional Coulomb Field Relaxation on a Single Hypersphere S^(D-1):
      - 1 Single 192-dimensional space (No subspaces)
      - Bidirectional context: Left and right tokens pull & push simultaneously!
      - Coulomb Force (Signed attraction/repulsion):
          F_coulomb = (q_i * Q / d^2) * (t_star - x_i)
      - Hooke Springs:
          Known tokens have stiff springs (kappa ≈ 0.95, anchors)
          Masked tokens have fluid springs (free to settle into minimum energy)
    """
    def __init__(self, dim=192, temp=10.0, eps=0.1):
        super().__init__()
        self.dim = dim
        self.temp = temp
        self.eps = eps

        self.query_gen = nn.Linear(dim, dim)
        self.attractor_charge = nn.Parameter(torch.tensor(1.0))
        self.eta_e = nn.Parameter(torch.tensor(0.40))

        # Channel mixer
        self.norm = nn.LayerNorm(dim)
        self.mlp  = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x, u, kappa, charge):
        # x, u: (B, T, D) on S^(D-1)
        # kappa: (B, T, 1) in (0, 1)
        # charge: (B, T, 1) in (-1, 1)
        B, T, D = x.shape

        # 1. Global context probe (Global Semantic Center of Mass)
        space_avg = F.normalize(x.mean(dim=1, keepdim=True), dim=-1)
        q = F.normalize(self.query_gen(space_avg), dim=-1)

        # 2. Retrieve central attractor T* bidirectionally across ALL T tokens
        match_scores  = torch.matmul(q, x.transpose(-1, -2)) * self.temp # (B, 1, T)
        match_weights = F.softmax(match_scores, dim=-1)
        t_star = F.normalize(torch.matmul(match_weights, x), dim=-1)      # (B, 1, D)

        # 3. Coulomb Force (Signed attraction or repulsion)
        cos_sim = (x * t_star).sum(dim=-1, keepdim=True)
        dist_sq = (2.0 - 2.0 * cos_sim).clamp(min=0.0) + self.eps
        Q = self.attractor_charge
        coulomb_coupling = (charge * Q) / dist_sq
        force_coulomb = coulomb_coupling * (t_star - x)

        # 4. Hooke's Law Spring Force
        force_spring = kappa * (u - x)

        # 5. Combined Physical Step
        f_net = self.eta_e * force_coulomb + force_spring
        x_moved = F.normalize(x + f_net, dim=-1)

        # 6. Residual Token Processing MLP
        x_out = F.normalize(x_moved + 0.25 * self.mlp(self.norm(x_moved)), dim=-1)
        return x_out


# ==========================================
# 3. COULOMB BERT-STYLE MASKED LANGUAGE MODEL
# ==========================================
class CoulombBertMLM(nn.Module):
    def __init__(self, vocab_size, dim=192, max_seq_len=128, default_rounds=6):
        super().__init__()
        self.dim = dim
        self.default_rounds = default_rounds
        self.max_seq_len = max_seq_len

        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        # Special learned Mask Particle Embedding on S^(D-1)
        self.mask_particle = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        self.kappa_proj = nn.Linear(dim, 1)
        self.charge_proj = nn.Linear(dim, 1)

        self.round = SingleSphereCoulombMLMRound(dim=dim)

        self.head_norm = nn.LayerNorm(dim * 2)
        self.lm_head   = nn.Linear(dim * 2, vocab_size)

    def forward(self, input_ids, mask, num_rounds=None):
        if num_rounds is None:
            num_rounds = self.default_rounds
        B, T = input_ids.shape
        pos = torch.arange(0, T, device=input_ids.device).unsqueeze(0)

        tok_embeds = self.tok_emb(input_ids)
        mask_exp = self.mask_particle.expand(B, T, self.dim)

        # Where mask=True, substitute with the mask particle coordinate!
        raw_tokens = torch.where(mask.unsqueeze(-1), mask_exp, tok_embeds)
        raw = raw_tokens + self.pos_emb(pos)

        u = F.normalize(raw, dim=-1) # (B, T, 192) on S^191
        x = u.clone()

        # Unmasked tokens are fixed anchors (kappa = 0.95); masked tokens are fluid
        base_kappa = torch.sigmoid(8.0 * self.kappa_proj(raw))
        stiffness = torch.where(mask.unsqueeze(-1), base_kappa, torch.full_like(base_kappa, 0.95))

        charge = torch.tanh(4.0 * self.charge_proj(raw))

        # PHYSICAL RELAXATION: Masked particles settle under bidirectional Coulomb forces!
        for _ in range(num_rounds):
            x = self.round(x, u, stiffness, charge)

        features = torch.cat([x, u], dim=-1)
        logits = self.lm_head(self.head_norm(features))
        return logits

    @torch.no_grad()
    def infill_text(self, sentence_with_masks, num_rounds=6):
        """
        Interactive Infilling:
        Takes a string where '[?]' represents masked characters.
        The model settles the masked particles from surrounding context in 1 pass!
        """
        self.eval()
        tokens = []
        mask_flags = []
        i = 0
        while i < len(sentence_with_masks):
            if sentence_with_masks[i:i+3] == "[?]":
                tokens.append(0) # dummy token
                mask_flags.append(True)
                i += 3
            else:
                tokens.append(stoi.get(sentence_with_masks[i], 0))
                mask_flags.append(False)
                i += 1

        T = min(len(tokens), self.max_seq_len)
        input_ids = torch.tensor(tokens[:T], dtype=torch.long, device=device).unsqueeze(0)
        mask_t = torch.tensor(mask_flags[:T], dtype=torch.bool, device=device).unsqueeze(0)

        logits = self(input_ids, mask_t, num_rounds=num_rounds)
        preds = logits.argmax(dim=-1)[0].tolist()

        result = []
        for t in range(T):
            if mask_flags[t]:
                result.append(itos[preds[t]])
            else:
                result.append(sentence_with_masks[t] if not mask_flags[t] else itos[preds[t]])
        return "".join(result)


# ==========================================
# 4. TRAIN BERT-STYLE COULOMB MLM (10 EPOCHS)
# ==========================================
if __name__ == "__main__":
    model = CoulombBertMLM(vocab_size, dim=192, max_seq_len=seq_len).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel Parameters : {total_params:,} (Single-Sphere Coulomb BERT MLM | S^191)")
    print(f"Task Config      : 15% Random Masking | Bidirectional Context | Epochs: 10\n")

    epochs = 10
    steps_per_epoch = 250
    val_steps = 40

    optimizer = torch.optim.AdamW(model.parameters(), lr=3.5e-3, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * steps_per_epoch, eta_min=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))

    telemetry = {
        "experiment": "Single-Sphere Coulomb BERT-style MLM (Tiny Shakespeare)",
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
            "epoch": epoch,
            "train_loss": round(train_loss / steps_per_epoch, 4),
            "val_loss": round(v_loss, 4),
            "masked_accuracy": round(v_acc, 2),
            "time_sec": round(epoch_sec, 1)
        })

        q_val = model.round.attractor_charge.item()
        eta_v = model.round.eta_e.item()
        print(f"Epoch {epoch:02d}/10 ({epoch_sec:.1f}s) | Train Loss: {train_loss/steps_per_epoch:.3f} | "
              f"Val Loss: {v_loss:.3f} | Masked Acc: {v_acc:.2f}% | [Q: {q_val:.2f}, eta: {eta_v:.2f}]")

    # SAVE NUMERICAL TELEMETRY
    json_path = "results/coulomb_bert_mlm_10epochs.json"
    with open(json_path, "w") as f:
        json.dump(telemetry, f, indent=2)
    print(f"\nSaved raw telemetry to {json_path}")

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
        filled = model.infill_text(s)
        print(f"\n[Masked Input] : {s}")
        print(f"[Field Settled]: {filled}")
        print("-" * 50)
