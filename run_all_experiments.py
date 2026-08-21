# run_all_experiments.py
# -*- coding: utf-8 -*-
"""
Batch Experiment Runner for DaoGAN
DaoGAN 批量实验运行脚本

This script runs all specified training configurations across multiple random seeds.
本脚本在多个随机种子下运行所有指定的训练配置。

Execution order: seed-first (i.e., run all modes for seed=42, then seed=85, etc.)
执行顺序：种子优先（即先对 seed=42 运行所有模式，再对 seed=77 运行所有模式）

Author: [jianhua]
Date: 2025
"""

import os
import subprocess
import datetime


# ======================
# Experimental Configuration / 实验配置
# ======================

# Training modes to evaluate / 待评估的训练模式
MODES = [
    "baseline",               # Baseline SNGAN / 基线 SNGAN
    "daosheng_temp07",        # DaoSheng with τ=0.7
    "daosheng_temp08",        # DaoSheng with τ=0.8
    "daosheng_temp085",       # DaoSheng with τ=0.85 (optimal in prior runs)
    "daosheng_temp09",        # DaoSheng with τ=0.9
    "yinyang_gradscale",      # True-YinYang gradient scaling (no DaoSheng)
    "yinyang_daosheng",       # True-YinYang + DaoSheng (τ=0.85)
    # "greenenergy",          # Optional: low-batch variant (uncomment if needed)
]

# Random seeds for reproducibility and robustness evaluation
# 用于可复现性与鲁棒性评估的随机种子
SEEDS = [42, 77, 85, 99, 123, 156, 200, 314, 404, 521]


def run_command(command: str) -> None:
    """
    Execute a shell command and stream its output in real time.
    执行 shell 命令并实时打印输出。

    Args:
        command (str): The command to execute / 要执行的命令
    """
    print(f"\n{'='*70}")
    print(f" Running command:\n{command}")
    print(f"{'='*70}")
    
    # Use shell=True to support command string with spaces
    process = subprocess.Popen(command, shell=True)
    process.wait()
    
    if process.returncode != 0:
        print(f"\n [ERROR] Command failed with return code {process.returncode}")
        print(f"Command: {command}")
    else:
        print(f"\n [SUCCESS] Command completed successfully")


def main():
    # Get the directory of this script and set as working directory
    # 获取脚本所在目录并设为工作目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    print(f"\n Starting batch experiments at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Project root: {os.getcwd()}")
    print(f"   Modes: {MODES}")
    print(f"   Seeds: {SEEDS}")
    print(f"   Execution order: seed-first (all modes per seed)")

    # Run experiments in SEED-first order
    # 按“种子优先”顺序运行实验
    for seed in SEEDS:
        print(f"\n{'#'*20} Running all modes for Seed = {seed} {'#'*20}")
        for mode in MODES:
            print(f"\n  Mode: {mode}")
            cmd = f"python train.py --mode {mode} --seed {seed}"
            run_command(cmd)

    # Final summary
    print(f"\n All experiments completed successfully at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Total runs: {len(MODES) * len(SEEDS)}")
    print(f"   Logs are saved under ./logs/seed{SEEDS[0]}/, etc.")


if __name__ == "__main__":
    main()