# DaoGAN: A Lightweight Framework for Stable GAN Training via Dynamic Adversarial Balance

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.3.1](https://img.shields.io/badge/PyTorch-2.3.1-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the complete implementation of **DaoGAN**, a lightweight GAN training framework that stabilizes adversarial training through **zero-parameter optimization-level interventions**:

- **DaoSheng (Temperature Scaling)**: Controls generator output diversity via a temperature factor applied to logits.
- **Yin-Yang Balance Mechanism**: Dynamically modulates generator/discriminator learning rates based on real-time discriminator output statistics, achieving adversarial equilibrium without architectural changes.

## Key Features

- **Zero additional parameters** — no auxiliary models, no architectural modifications
- **Low overhead** (3–6% training time increase, single extra discriminator forward pass)
- **Statistically rigorous evaluation** — 10 independent random seeds, Welch's t-test, Benjamini-Hochberg FDR correction, Hedges' g effect sizes, post-hoc power analysis
- **Fully reproducible** — all hyperparameters, seeds, and evaluation protocols documented

---

## Installation

```bash
git clone https://github.com/jerhua1024/dao-gan.git
cd dao-gan
```

Install the locally modified torch-fidelity (required for evaluation):
```bash
cd torch-fidelity
pip install -e . --no-deps
cd ..
```

Install dependencies (CUDA 12.1):
```bash
pip install -r requirements.txt \
  --index-url https://download.pytorch.org/whl/cu121 \
  --extra-index-url https://pypi.org/simple
```

### Environment

| Component | Version |
|-----------|---------|
| Python | 3.11 |
| PyTorch | 2.3.1 + cu121 |
| torch-fidelity | v0.4.0 (locally modified) |
| statsmodels | 0.14.6 (statistical analysis) |
| GPU | NVIDIA RTX 4060 (8 GB VRAM) |

---

## Usage

### Run All Experiments (7 modes × 10 seeds)

```bash
python run_all_experiments.py
```

This executes all configurations in seed-first order across 10 random seeds.

### Run a Single Configuration

```bash
python train.py --mode yinyang_gradscale --seed 42
```

### Supported Modes

| Mode | Description |
|------|-------------|
| `baseline` | Standard SN-GANs (DCGAN) with spectral normalization |
| `daosheng_temp07` | Temperature scaling only (τ=0.7) |
| `daosheng_temp08` | Temperature scaling only (τ=0.8) |
| `daosheng_temp085` | Temperature scaling only (τ=0.85) |
| `daosheng_temp09` | Temperature scaling only (τ=0.9) |
| `yinyang_gradscale` | Yin-Yang dynamic LR modulation only |
| `yinyang_daosheng` | Combined Yin-Yang + Temperature (τ=0.85) |

### Post-Training Analysis Pipeline

```bash
# Step 1: Extract TensorBoard scalars
python scripts/1_extract_tb_scalars.py

# Step 2: Compress to lightweight format
python scripts/2_compress_scalars.py

# Step 3: Compute aggregated statistics (mean ± std across seeds)
python scripts/3_compute_plot_data.py

# Step 4: Generate FID training curve figure
python scripts/4_plot_fid_facet_paper.py

# Step 5: Generate result tables with full statistical analysis
python scripts/5_generate_paper_table.py

# Step 6: Generate qualitative comparison figure
python scripts/6_generate_fig3_qualitative.py
```

---

## Experimental Results (CIFAR-10, n=10 seeds)

Final metrics at step=100,000 (mean ± std across 10 independent random seeds):

| Method | FID ↓ | IS ↑ | KID ↓ (×10⁻²) |
|--------|-------|------|---------------|
| SN-GANs (Baseline) | 28.82 ± 0.68 | 6.90 ± 0.08 | 2.29 ± 0.10 |
| DaoSheng (τ=0.7) | 29.16 ± 0.78 | 6.87 ± 0.07 | 2.30 ± 0.06 |
| DaoSheng (τ=0.8) | 28.52 ± 1.12 | 6.93 ± 0.08 | 2.22 ± 0.13 |
| DaoSheng (τ=0.85) | 28.99 ± 1.10 | 6.91 ± 0.10 | 2.26 ± 0.15 |
| DaoSheng (τ=0.9) | 29.32 ± 0.92 | 6.86 ± 0.04 | 2.33 ± 0.05 |
| **Yin-Yang GradScale** | **27.65 ± 0.97** †‡ | **6.97 ± 0.13** | **2.14 ± 0.10** †‡ |
| Yin-Yang + DaoSheng (τ=0.85) | 28.13 ± 1.03 | 6.94 ± 0.09 | 2.19 ± 0.16 |

† p_raw < 0.05; ‡ p_FDR < 0.05 after Benjamini-Hochberg correction (3 core comparisons)

**Key statistical findings:**
- Yin-Yang vs Baseline (FID): p=0.006 (FDR-corrected p=0.009), Hedges' g=-1.35 (large effect), post-hoc power=81%
- DaoSheng (τ=0.8) shows exploratory improvement (p=0.469, not significant) — reported as honest null result

---

## Relationship to Modern GAN Baselines

DaoGAN is complementary to, not competitive with, modern GAN paradigms:

| Method | Paradigm | Architecture Change | Extra Params | Theoretical Guarantee |
|--------|----------|---------------------|--------------|----------------------|
| **DaoGAN** | Optimization-level LR modulation | None | 0 | Empirical (no formal proof) |
| R3GAN (NeurIPS 2024) | Loss reform (RpGAN+R1+R2) + modern architecture | Yes (ResNeXt) | Yes | Local convergence proof |
| CSA-GAN (Neurocomputing 2025) | Contrastive self-adversarial + feature entropy | Encoder needed | Yes | Empirical |
| Li-CFG (ML 2024) | Lipschitz-constrained functional gradient | No | Gradient penalty | Convergence guarantee |

DaoGAN's unique value: a **paradigm-agnostic optimization plugin** for existing GAN pipelines that cannot afford architectural redesign.

---

## Project Structure

```
dao-gan/
├── train.py                    # Main training script (7 modes)
├── run_all_experiments.py      # Batch experiment runner
├── requirements.txt            # Python dependencies
├── scripts/                     # Analysis pipeline (6 scripts)
│   ├── 1_extract_tb_scalars.py # Extract TensorBoard → JSON
│   ├── 2_compress_scalars.py   # Compress to lightweight format
│   ├── 3_compute_plot_data.py  # Aggregate across seeds
│   ├── 4_plot_fid_facet_paper.py # FID training curves
│   ├── 5_generate_paper_table.py # Statistical tables (Welch/FDR/Hedges'g/power)
│   └── 6_generate_fig3_qualitative.py # Qualitative comparison
├── torch-fidelity/             # Locally modified evaluation library
├── .gitignore
└── README.md
```

Note: Training logs, datasets, extracted metrics, figures, tables, and paper source files are excluded from version control (see `.gitignore`).

---

## Citation

If you use this work, please cite:

```bibtex
@article{daogan2026,
  title={DaoGAN: A Lightweight Framework for Stable GAN Training via Dynamic Adversarial Balance},
  author={Wang, Jianhua and Mo, Taiping and others},
  journal={arXiv preprint (to appear)},
  year={2026}
}
```

## License

MIT License — see [LICENSE](LICENSE).

## Contact

Maintained by Jer Hua.

For technical discussions, feedback or code contributions, please open an Issue or Pull Request on GitHub.
