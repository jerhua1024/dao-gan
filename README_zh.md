---
title: "DaoGAN: A Daoism-Inspired GAN Training Framework"
description: "Dynamic gradient scaling and temperature-controlled diversity for stable GAN training"
authors: [Jianhua Wang, Taiping Mo, Shoushuai Jiang, Boxue Chang, Xingxiao Tian]
date: 2026-04-06
version: 1.0.0
---

# DaoGAN：受道家思想启发的 GAN 训练框架
> **DaoGAN**: A Daoism-Inspired Framework for Stable GAN Training via Dynamic Gradient Scaling and Temperature-Controlled Diversity

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3.1%2Bcu121-ee4c2c)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2506.00001-b31b1b)](https://arxiv.org/abs/2506.00001)

---

## 📋 快速导航
- [✨ 核心特性](#-核心特性)
- [🧭 核心理念](#-核心理念)
- [📦 安装指南](#-安装指南)
- [🚀 快速开始](#-快速开始)
- [⚙️ 配置与实验](#️-配置与实验)
- [📊 数据提取与可视化](#-数据提取与可视化)
- [🔬 实验结果](#-实验结果)
- [📁 项目结构](#-项目结构)
- [🔧 VSCode 开发配置](#-vscode-开发配置)
- [🤝 贡献指南](#-贡献指南)
- [📜 引用](#-引用)
- [📄 许可证](#-许可证)

---

## ✨ 核心特性
| 特性 | 描述 | 优势 |
| :--- | :--- | :--- |
| **🌱 道生机制 (DaoSheng)** | 通过温度因子 `τ` 调控生成样本多样性分布 | 模拟自然演化，兼顾生成质量与多样性 |
| **☯️ 阴阳平衡机制 (YinYang)** | 依据判别器输出动态缩放生成器/判别器学习率，制衡训练差异 | 自适应调参，杜绝人工调参成本，训练稳定性大幅提升 |
| **🔧 轻量化易集成** | 无需修改网络结构、无额外新增训练参数，仅梯度层面实现优化 | 即插即用，兼容主流原生 GAN 系列模型 |
| **📈 高可复现性** | 全局随机种子锁定、学习率轨迹全记录、自动化指标评估 | 支持多种子统计实验，实验结果客观可信 |
| **🌏 双语友好** | 代码中英双语注释，配套中文使用文档 | 降低入门门槛，便于学术交流与工程落地 |

---

## 🧭 核心理念
DaoGAN 创新性融合中国传统道家哲学思想，将古典哲学思想映射为可落地、可计算的 GAN 训练优化策略。

| 道家概念 | 哲学内涵 | DaoGAN 技术实现 | 实际技术价值 |
| :--- | :--- | :--- | :--- |
| **道生一，一生二，二生三，三生万物** | 万物由本源规律自然演化生成 | 温度因子 `τ` 分布映射：`output = tanh(logits / τ)` | 小 `τ` 强化细节质感，大 `τ` 拓宽样本多样性 |
| **阴阳平衡，动态调和** | 对立要素相互制衡、动态均衡 | 基于 `D_real`/`D_fake` 均值自适应缩放 G/D 学习率 | 有效缓解梯度消失、梯度爆炸，加速平稳收敛 |
| **无为而治，顺势而为** | 减少人为强制干预，遵循事物自身规律 | 不新增网络分支与训练模块，仅梯度策略优化 | 模型体量无增加，几乎无额外计算开销 |

> 💡 设计哲学：**最好的训练优化，是让模型自主寻得训练平衡**

---

## 📦 安装指南
### 🔍 环境依赖要求
| 依赖组件 | 推荐运行版本 | 备注说明 |
| :--- | :--- | :--- |
| Python | 3.11.x（推荐 3.11.4） | 兼容区间 `>=3.11, <3.12` |
| PyTorch | 2.3.1+cu121 | 强制匹配 CUDA12.1 编译版本 |
| CUDA | 12.1 | 显卡驱动版本 ≥ 530.30.02 |
| 操作系统 | Win10/11、Linux、macOS | Windows 优先使用 Git Bash 终端 |

### 🚀 环境部署两种方案
#### 方案A：本地Wheel离线安装（国内网络首选）

    # 1. 克隆项目仓库
    git clone https://github.com/jerhua1024/dao-gan.git
    cd dao-gan

    # 2. 创建并激活独立虚拟环境
    python -m venv .venv
    source .venv/Scripts/activate      # Git Bash
    # .venv\Scripts\Activate.ps1       # PowerShell

    # 3. 提前手动下载对应离线包
    # torch: https://download.pytorch.org/whl/cu121/torch-2.3.1+cu121-cp311-cp311-win_amd64.whl
    # torchvision: https://download.pytorch.org/whl/cu121/torchvision-0.18.1+cu121-cp311-cp311-win_amd64.whl

    # 4. 离线安装CUDA版本PyTorch
    pip install "你的本地存放路径/torch-2.3.1+cu121-cp311-cp311-win_amd64.whl"
    pip install "你的本地存放路径/torchvision-0.18.1+cu121-cp311-cp311-win_amd64.whl"

    # 5. 安装其余项目依赖
    pip install -r requirements.txt

    # 6. 安装定制版 torch-fidelity（禁止依赖覆盖）
    pip install -e torch-fidelity --no-deps

#### 方案B：在线一键安装（外网稳定环境使用）

    # 1. 克隆进入项目
    git clone https://github.com/jerhua1024/dao-gan.git
    cd dao-gan

    # 2. 初始化虚拟环境
    python -m venv .venv
    .venv/Scripts/activate

    # 3. 升级pip至高版本
    python -m pip install --upgrade pip

    # 4. 批量安装全部依赖
    pip install -r requirements.txt \
      --index-url https://download.pytorch.org/whl/cu121 \
      --extra-index-url https://pypi.org/simple

    # 5. 安装定制评估库
    pip install -e torch-fidelity --no-deps

### ✅ 环境安装校验

    # 环境版本校验
    python -c "
    import torch
    print(f'PyTorch Version: {torch.__version__}')
    print(f'CUDA Available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA Version: {torch.version.cuda}')
        print(f'GPU Device: {torch.cuda.get_device_name(0)}')
    "

    # 快速10步训练可用性测试
    python train.py --mode baseline --seed 42 --num_total_steps 10

    # 查看日志输出文件
    ls logs/seed42/

---

## 🚀 快速开始运行
### 单组实验调试运行

    # 基线GAN快速调试（100步）
    python train.py --mode baseline --seed 42 --num_total_steps 100

    # 核心阴阳平衡机制完整训练
    python train.py --mode yinyang_gradscale --seed 123 --num_total_steps 100000 --batch_size 64

### 多模式批量自动化实验

    # 前台批量运行
    python run_all_experiments.py

    # Linux 服务器后台静默运行
    nohup python run_all_experiments.py > logs/batch_run.log 2>&1 &

> 批量实验模式、随机种子可直接修改 `run_all_experiments.py` 头部配置变量。

### 核心训练参数释义
| 参数名 | 参数类型 | 默认值 | 参数说明 |
| :--- | :--- | :--- | :--- |
| `--mode` | str | baseline | 实验运行模式 |
| `--seed` | int | 42 | 全局随机种子，保证实验可复现 |
| `--temperature` | float | 1.0 | 道生机制温度超参 |
| `--batch_size` | int | 64 | 训练批次大小 |
| `--num_total_steps` | int | 100000 | 整体训练迭代步数 |
| `--lr` | float | 2e-4 | 基础初始学习率 |
| `--disable_sn` | flag | False | 关闭判别器谱归一化 |

---

## ⚙️ 实验模式配置
| 运行模式 | 启动命令 | 功能说明 | 适用场景 |
| :--- | :--- | :--- | :--- |
| baseline | `--mode baseline` | 原始Hinge Loss SNGAN基线 | 实验对标、代码验证 |
| daosheng_temp07 | `--mode daosheng_temp07` | 道生机制 τ=0.7 | 小温度侧重细节生成 |
| daosheng_temp08 | `--mode daosheng_temp08` | 道生机制 τ=0.8 | 通用均衡默认参数 |
| daosheng_temp085 | `--mode daosheng_temp085` | 道生机制 τ=0.85 | 细节与多样性均衡 |
| daosheng_temp09 | `--mode daosheng_temp09` | 道生机制 τ=0.9 | 大温度侧重样本多样性 |
| yinyang_gradscale | `--mode yinyang_gradscale` | 阴阳梯度动态缩放 | 论文核心主推方案 |
| yinyang_daosheng | `--mode yinyang_daosheng` | 阴阳+道生双机制融合 | 多机制协同消融实验 |

---

## 📊 数据提取与论文绘图全流程
训练完成
→ 1_extract_tb_scalars.py 提取TensorBoard原始指标
→ 2_compress_scalars.py 精简压缩实验数据
→ 3_compute_plot_data.py 计算多种子均值±标准差
→ 4_plot_fid_facet_paper.py 绘制论文FID收敛曲线图
→ 5_generate_paper_table.py 生成论文定量结果表格
→ 6_generate_fig3_qualitative.py 生成定性可视化对比图

### 一键执行脚本命令

    # 1.提取日志指标
    python scripts/1_extract_tb_scalars.py
    # 2.统计均值方差()
    python scripts/2_compress_scalars.py #运行前需要修改INPUT_FILE = Path("extracted_scalars/scalars_*.json")
    # 3.数据轻量化压缩
    python scripts/3_compute_plot_data.py #运行前需要修改INPUT_FILE = Path("extracted_scalars/scalars_*_lite.json")
    # 4.绘制FID曲线
    python scripts/4_plot_fid_facet_paper.py
    # 5.生成实验数据表
    python scripts/5_generate_paper_table.py
    # 6.生成样本可视化图
    python scripts/6_generate_fig3_qualitative.py

---

## 🔬 实验定量结果（CIFAR-10，10种子统计）
| Method | FID ↓ | IS ↑ | KID ↓ (×10⁻²) |
|---|---|---|---|
| SNGAN (Baseline) | $28.82 \pm 0.68$ | $6.90 \pm 0.08$ | $2.29 \pm 0.10$ |
| τ=0.7 | $29.16 \pm 0.78$ | $6.87 \pm 0.07$ | $2.30 \pm 0.06$ |
| τ=0.8 | $28.52 \pm 1.12$ | $6.93 \pm 0.08$ | $2.22 \pm 0.13$ |
| τ=0.85 | $28.99 \pm 1.10$ | $6.91 \pm 0.10$ | $2.26 \pm 0.15$ |
| τ=0.9 | $29.32 \pm 0.92$ | $6.86 \pm 0.04$ | $2.33 \pm 0.05$ |
| Yin-Yang GradScale | $\mathbf{27.65 \pm 0.97}^{\dagger\ddagger}$ | $\mathbf{6.97 \pm 0.13}$ | $\mathbf{2.14 \pm 0.10}^{\dagger\ddagger}$ |
| Yin-Yang + DaoSheng ($\tau=0.85$) | $28.13 \pm 1.03$ | $6.94 \pm 0.09$ | $2.19 \pm 0.16$ |

---

## 📁 项目标准目录结构
dao-gan/
├── README.md                     # 项目主文档
├── run_all_experiments.py        # 批量实验调度脚本
├── train.py                      # 核心训练主程序
├── requirements.txt              # 项目依赖清单
├── scripts/                      # 数据分析绘图脚本集
│   ├── 1_extract_tb_scalars.py
│   ├── 2_compress_scalars.py
│   ├── 3_compute_plot_data.py
│   ├── 4_plot_fid_facet_paper.py
│   ├── 5_generate_paper_table.py
│   └── 6_generate_fig3_qualitative.py
├── logs/                         # 训练日志与模型权重
├── extracted_scalars/            # 提取原始实验指标
├── plot_data/                    # 统计后绘图数据
├── figures/                      # 论文最终效果图
├── tables/                       # 论文实验数据表
├── docs/                         # 相关论文文稿
└── torch-fidelity/               # 定制化评估指标库

---

## 📜 学术引用
若本项目对你的研究工作有帮助，请引用如下文献：

    @article{wang2025daogan,
      title={DaoGAN: A Lightweight, Philosophy-Inspired Framework for Stable and Diverse GAN Training},
      author={Jianhua Wang, Taiping Mo, Shoushuai Jiang, Boxue Chang, Xingxiao Tian},
      journal={arXiv preprint arXiv:2506.00001},
      year={2025}
    }

---

## 📄 开源许可证
本项目基于 **MIT License** 开源，可自由用于学术研究、二次开发与工程落地，使用过程需保留原始版权与开源协议声明。

---

## 🙏 项目致谢
感谢中华传统道家哲学思想为深度学习优化方向提供全新设计思路；感谢所有项目贡献者、科研同行与审稿专家提供宝贵修改意见。

> 道生一，一生二，二生三，三生万物，万物负阴而抱阳，冲气以为和 ——《道德经》第四十二章

**项目维护者**：[Jer Hua](https://github.com/jerhua1024)
**最后文档更新时间**：2026-04-06
**项目状态**：🟢 持续迭代开发中