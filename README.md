# Gravitational Attention Networks

> **Replacing Pairwise Self-Attention with Gravito-Elastic Field Dynamics on Orthogonal Hyperspheres**

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 1. The Core Idea: Field Dynamics on Hyperspheres

Traditional neural networks (Transformers, MLPs, ResNets) rely on unconstrained affine transformations at every depth:

$$
\mathbf{x}^{(l+1)} = \sigma\left(\mathbf{W}^{(l)} \mathbf{x}^{(l)} + \mathbf{b}^{(l)}\right)
$$

By layer $L$, the geometric coordinate manifold has been continuously stretched, warped, and destroyed.

**Gravitational Attention Networks** propose a fundamental paradigm shift:
Tokens are treated as **discrete physical particles residing on a permanent compact manifold**—specifically, a Cartesian product of $K$ orthogonal unit hyperspheres:

$$
\mathcal{M} = \mathbb{S}^{d_1-1} \times \mathbb{S}^{d_2-1} \times \dots \times \mathbb{S}^{d_K-1}, \quad \text{where } d_k = \frac{D}{K}
$$

Tokens never leave these spherical surfaces. Instead of layer-wise transformation, tokens **migrate** across the surface of the hyperspheres governed by classical equations of motion until they reach dynamic equilibrium:

$$
\Delta \mathbf{x}_i \propto \mathbf{F}_{\text{net}, i} = \eta \mathbf{F}_{\text{gravity}, i} + \mathbf{F}_{\text{Hooke}, i}
$$

```
                +-------------------------------------------+
                |           Unit Hypersphere S^(d-1)        |
                |                                           |
     Anchor     |      Hooke Restoring Force F_spring       |
       u_i  o===|==================================>        |
                |                                   \       |
                |                                    x_i    |  Token particle
                |                                   /       |
                |      Inverse-Square Gravity      v        |
                |   <==============================o        |
                |                                  T*       |  Winning Attractor
                +-------------------------------------------+
```

---

## 2. The Three Physical Pillars

### 2.1 Gravitational Pull Toward Retrieved Real Tokens ($T^\ast$)

Instead of computing an $O(N^2)$ all-pairs attention matrix or pulling tokens toward synthetic centroids, each subspace generates a probe query $\mathbf{q}_k$ from the collective field average, identifies the best-matching real token $T_k^\ast$, and endows it with learnable mass $M_k$. Every token $\mathbf{x}_i$ is pulled toward $T_k^\ast$ via an inverse-square force law in $O(N)$ linear time:

$$
\mathbf{F}_{\text{grav}, i, k} = \frac{M_k}{d_{i, k}^2} \left(\mathbf{t}_k^\ast - \mathbf{x}_{i, k}\right), \quad d_{i, k}^2 = 2 - 2\langle \mathbf{x}_{i, k}, \mathbf{t}_k^\ast \rangle + \epsilon
$$

### 2.2 Hooke's Law Elastic Anchoring ($\mathbf{u}_i$)

Pure gravitational attraction causes total collapse into a single point/singularity (over-smoothing). To preserve identity and topology, every token is tethered to its initial coordinate $\mathbf{u}_i$ (patch embedding + position) via a virtual spring exerting a restoring force:

$$
\mathbf{F}_{\text{spring}, i, k} = \kappa_{i, k} (\mathbf{u}_{i, k} - \mathbf{x}_{i, k})
$$

Each token predicts its own dynamic spring constant $\kappa_{i, k} \in (0, 1)$, allowing structural/foreground regions to stay rigid ($\kappa \to 1$) while contextual/background regions remain fluid ($\kappa \to 0$).

### 2.3 Orthogonal Subspace Multi-Body Dynamics

When $K$ attractors pull inside the *same* vector space, opposing force vectors collide and cancel each other out (**destructive interference**). By decomposing the $D$-dimensional space into $K$ mutually orthogonal hyperspheres $\mathbb{S}^{d_k-1}$, $K$ independent gravitational bodies pull tokens along perpendicular degrees of freedom with **zero vector cancellation**.

---

## 3. Current Architecture

The architecture uses **one single weight-tied recurrent module** (`OrthogonalVisionRound` / `LinearCausalGravitationalRound`) unrolled iteratively for $R$ rounds ($R=6$):

```mermaid
flowchart TD
    In["Input Sequence / Image Tokens"] --> Stem["Embedding / Patch Stem (D=192)"]
    Stem --> Pos["+ Positional Encoding"]
    Pos --> Init["Project & Normalize onto K=4 Spheres: S^47"]
    Init --> U["Anchor u_i and Initial State x^(0) = u_i"]
    U --> Kappa["Per-Token, Per-Subspace Elasticity kappa_{i,k}"]
    
    subgraph RecurrentLoop ["Weight-Tied Gravitational Round (Looped R = 6 Times)"]
        CoM["1. Center of Mass (Global Mean or Causal Prefix Scan)"]
        Query["2. Subspace Probe Queries q_k"]
        Attract["3. Retrieve Real Attractor T*_k per Subspace"]
        Gravity["4. Inverse-Square Gravitational Pull: F_grav = (M / d^2) * (T* - x)"]
        Spring["5. Hooke's Law Restoring Spring: F_spring = kappa * (u - x)"]
        Step["6. Geodesic Step + Residual Cross-Subspace MLP"]
        
        CoM --> Query --> Attract --> Gravity
        Spring --> Step
        Gravity --> Step
    end
    
    U --> RecurrentLoop
    Kappa --> RecurrentLoop
    RecurrentLoop --> Readout["Dual Readout: Cat[Tokens, Center-of-Mass]"]
    Readout --> Head["LayerNorm + Task Head (Classifier / LM Head)"]
```

---

## 4. Vision Benchmarks & Physics Ablations (CIFAR-100)

Trained on CIFAR-100 ($N=256$ patches, $D=192$, $K=4$ orthogonal $48$-D spheres, 6 shared rounds) for 40 epochs:

* **Parameters:** **351,568**
* **Top-1 Accuracy:** **59.22%**
* **Top-5 Accuracy:** **85.32%**

### Physics Ablations (CIFAR-100 Epoch 40, Zero Retraining)
Evaluating the model while isolating each physical force demonstrates that the network organizes its core computation around the physical field:

| Configuration | Top-1 Accuracy | Top-5 Accuracy | Delta vs. Full | Physical Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Full Orthogonal Model** | **59.22%** | **85.32%** | — | Balanced gravito-elastic field dynamics |
| **Gravity Disabled ($\eta = 0$)** | **10.04%** | — | **-49.18%** | Field dynamics do ~85% of classification work |
| **Uniform Springs ($\kappa = 0.5$)** | **30.08%** | — | **-29.14%** | Elasticity differentiation is critical for perception |
| **Springs Disabled ($\kappa = 0$)** | **1.22%** | — | **-58.00%** | Complete collapse into a gravitational singularity |

---

## 5. Emergent Visual Discoveries

### Emergent Object Shadows (Unsupervised Segmentation)
Without any segmentation supervision, structural foreground tokens automatically assign themselves near-maximum stiffness ($\kappa \to 1.0$), remaining fixed near their anchors, while fluid background tokens are dragged across the hypersphere toward the attractor:

![Emergent Object Shadows](results/emergent_object_shadows.png)

### Subspace Division of Labor & Dual Push-Pull Masks
Inspecting the 4 orthogonal spheres reveals functional division of labor: Subspace 4's attractor ($T_4^\ast$, yellow dot) consistently locks onto the primary object center, while Subspaces 1–3 anchor to peripheral boundaries. In addition, Subspaces 1 and 4 exhibit complementary push-pull elasticity masks (center-surround antagonism):

![Subspace Elasticity Partitioning](results/subspace_elasticity_partitioning.png)

---

## 6. Strictly $O(T)$ Linear Causal Language Modeling (Tiny Shakespeare)

Standard causal self-attention scales quadratically as $O(T^2)$ in time and memory due to the lower-triangular causal attention matrix. 

Gravitational Attention achieves **strictly $O(T)$ linear-time autoregressive modeling with ZERO $T \times T$ matrices** via exact exponential cancellation.

### 6.1 Exact Exponential Cancellation Derivation
In the past light-cone ($j < t$), query-token match scores decay with learned temporal horizon $\gamma_k$:

$$
\text{Score}(t, j) = s_j - \gamma_k (t - j) = s_j - \gamma_k t + \gamma_k j
$$

When evaluating normalized softmax weights over past tokens:

$$
w_{t, j} = \frac{\exp\left(s_j - \gamma_k(t - j)\right)}{\sum_{j < t} \exp\left(s_j - \gamma_k(t - j)\right)} = \frac{\exp(-\gamma_k t) \cdot \exp(s_j + \gamma_k j)}{\exp(-\gamma_k t) \cdot \sum_{j < t} \exp(s_j + \gamma_k j)} = \frac{\exp(s_j + \gamma_k j)}{\sum_{j < t} \exp(s_j + \gamma_k j)}
$$

Because the term $\exp(-\gamma_k t)$ is strictly independent of index $j$, **it factors out of both the numerator and denominator and cancels completely**. 

Consequently, the retrieved past attractor $\mathbf{t}_{t, k}^\ast$ reduces to a cumulative prefix sum (`cumsum`):

$$
\mathbf{t}_{t, k}^\ast = \text{Normalize}\left(\frac{\sum_{j < t} \exp(s_j + \gamma_k j) \mathbf{x}_{j, k}}{\sum_{j < t} \exp(s_j + \gamma_k j)}\right)
$$

This operates in **strictly $O(T)$ time and memory**. No $T \times T$ attention matrix is ever materialized.

### 6.2 Tiny Shakespeare Benchmark Results
Trained on Tiny Shakespeare (1.11M characters, $T=128$, $D=192$, $K=4$ orthogonal $48$-D spheres, 6 shared rounds, 10 epochs):

* **Parameters:** **249,105**
* **Causality Verification:** Maximum future-to-past information leak ($t < 64$ from future tokens $t \ge 64$) = **$0.00\text{e}+00$**
* **Validation Loss:** **1.583**
* **Next-Character Accuracy:** **53.60%**

#### Physics Ablation (Tiny Shakespeare Epoch 10)

| Configuration | Val Loss | Next-Char Accuracy | Delta vs. Full |
| :--- | :---: | :---: | :--- |
| **Full $O(T)$ Orthogonal Model** | **1.583** | **53.60%** | — |
| **Constant Spring ($\kappa = 0.5$)** | **2.335** | **33.38%** | **-20.22%** |
| **Gravity Disabled ($\eta = 0$)** | **3.979** | **16.47%** | **-37.13%** |
| **Springs Disabled ($\kappa = 0$)** | **7.534** | **4.32%** | **-49.29%** (Catastrophic collapse) |

### 6.3 Emergent Linguistic Dynamics: Phonetic Backbones & Word Horizons
Analyzing per-character elasticity $\kappa_{t, k}$ and spacetime lookback distance $(t - j)$ across Hamlet's *"To be, or not to be, that is the question:"* reveals functional hierarchy across the 4 orthogonal spheres:

![Causal LM Shakespeare Analysis](results/causal_lm_shakespeare_analysis.png)

1. **Subspaces 3 & 4 ($\gamma = 4.23, 4.29$ — Markovian Phonetic Backbone):** Under steep decay, lookback distance is strictly locked at **1** (lookback of $t-1$) across every character, providing an unbroken Markovian chain for spelling.
2. **Subspace 2 ($\gamma = 2.86$ — Digram Buffer):** Alternates between lookback 1 and 2 characters back, capturing immediate phonemes and vowel-consonant transitions.
3. **Subspace 1 ($\gamma = 2.08$ — Multi-Character Morpheme Horizon):** Lowest temporal decay rate. Lookback expands to distances 2 and 3, tracking word progressions (`To␣` $\to$ 1, 2, 3; `or␣` $\to$ 1, 2, 3).
4. **Punctuation & Boundary Pillars:** In Subspace 1, syntactic punctuation (`,` and `:`) and clause starts receive peak spring stiffness ($\kappa \to 1.0$), serving as fixed grammatical anchors.

### 6.4 Sample Generation
```text
ROMEO:
Wht, they no con?

CAPlacior and?

HER:
Giard one. Is welld they should be hears
Which I should not content of the man hers?
And, be hither encounted to the touch'd mine.

First ServinLeCUS:
A beast what you can penter n
```

---

## 7. Repository Structure

```
├── experiments/
│   ├── cifar100_orthogonal_vision.py        # 2D Vision: CIFAR-100 training, ablation & visualizer
│   └── shakespeare_causal_lm.py             # 1D Causal LM: Strictly O(T) Shakespeare training & visualizer
├── results/
│   ├── emergent_object_shadows.png           # 2D Vision: Unsupervised figure-ground segmentation heatmaps
│   ├── subspace_elasticity_partitioning.png   # 2D Vision: 4-subspace attractor & elasticity analysis
│   └── causal_lm_shakespeare_analysis.png    # 1D Causal LM: Lookback distance & elasticity analysis
├── README.md
└── .gitignore
```

---

## 8. Quick Start

### Installation
```bash
pip install torch torchvision datasets matplotlib numpy
```

### Reproduce Vision Benchmark (CIFAR-100, 40 Epochs)
```bash
python experiments/cifar100_orthogonal_vision.py
```

### Reproduce Causal LM Benchmark (Tiny Shakespeare, 10 Epochs)
```bash
python experiments/shakespeare_causal_lm.py
```
Both scripts load datasets directly into GPU VRAM, verify mathematical physics invariants (causality checks, geodesic normalization), execute full physics ablation suites, and save high-resolution figures to `results/`.
