# scripts/1_extract_tb_scalars.py
# 生产级数据提取：支持新旧代码混合 / 部分训练 / Seed复用 / 自动去重 / 防覆盖 / 版本感知
import os
from pathlib import Path
import re
import json
import datetime
from tensorboard.backend.event_processing import event_accumulator

# === 配置 ===
LOGS_DIR = Path("logs")
OUTPUT_DIR = Path("extracted_scalars")
OUTPUT_DIR.mkdir(exist_ok=True)

# 【对齐优化版 train.py】完整标签集
# 旧版日志缺失的标签会自动返回空列表，安全无害
SCALAR_TAGS = {
    # 基础指标（新旧版共有）
    "FID": "metrics/frechet_inception_distance",
    "IS": "metrics/inception_score_mean",
    "KID": "metrics/kernel_inception_distance_mean",
    "Loss_G": "loss/loss_G",
    "Loss_D": "loss/loss_D",
    # 🆕 新版专属标签（旧版日志中会自动返回空）
    "Balance_LrG_Mult": "balance/lr_G_mult",
    "Balance_LrD_Mult": "balance/lr_D_mult",
    "LR_G_Scheduled": "LR/G_scheduled",
    "LR_D_Scheduled": "LR/D_scheduled",
}

METHOD_NAME_MAP = {
    "Baseline": "baseline",
    "Daosheng-Temp07": "daosheng_temp07",
    "Daosheng-Temp08": "daosheng_temp08",
    "Daosheng-Temp085": "daosheng_temp085",
    "Daosheng-Temp09": "daosheng_temp09",
    "True-YinYang-Gradscale": "yinyang_gradscale",
    "True-YinYang-Daosheng": "yinyang_daosheng",
}


def extract_seed_from_path(path: Path) -> int:
    match = re.search(r"seed(\d+)", str(path))
    return int(match.group(1)) if match else None


def extract_method_from_dirname(dirname: str) -> str:
    sorted_names = sorted(METHOD_NAME_MAP.keys(), key=lambda x: -len(x))
    for name in sorted_names:
        if name in dirname:
            return METHOD_NAME_MAP[name]
    return "unknown"


def extract_scalars_from_tb(log_dir: Path, tags: dict):
    """提取 TensorBoard Scalars，缺失标签自动返回空列表"""
    try:
        ea = event_accumulator.EventAccumulator(str(log_dir))
        ea.Reload()
        data = {}
        for key, tag in tags.items():
            if tag in ea.scalars.Keys():
                steps = [s.step for s in ea.scalars.Items(tag)]
                values = [s.value for s in ea.scalars.Items(tag)]
                data[key] = {"steps": steps, "values": values}
            else:
                # 关键：缺失标签返回空结构，确保新旧日志兼容
                data[key] = {"steps": [], "values": []}
        return data
    except Exception as e:
        print(f"TB read error: {log_dir} -> {e}")
        return None


def extract_lr_history_json(log_dir: Path):
    """提取 lr_history.json，不存在时返回 None"""
    lr_json_path = log_dir / "lr_history.json"
    if lr_json_path.exists():
        try:
            with open(lr_json_path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"LR history read error: {lr_json_path} -> {e}")
    return None


def detect_code_version(log_dir: Path) -> str:
    """
    【新增】自动检测日志来源的代码版本
    - modern: 存在 lr_history.json（新版代码生成）
    - legacy: 不存在 lr_history.json（旧版代码生成）
    """
    if (log_dir / "lr_history.json").exists():
        return "modern"
    return "legacy"


def main():
    # 结构: {method: {seed: {"tb": {...}, "lr_history": {...}, "code_version": str, "source_dir": str}}}
    all_data = {}
    extracted_runs = []  # 用于打印明细报告

    # 1. 收集所有候选运行目录
    candidates = []
    for seed_dir in sorted(LOGS_DIR.iterdir()):
        if not seed_dir.is_dir() or not seed_dir.name.startswith("seed"):
            continue
        seed = extract_seed_from_path(seed_dir)
        if seed is None:
            continue
        for exp_dir in sorted(seed_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            method = extract_method_from_dirname(exp_dir.name)
            if method == "unknown":
                continue
            candidates.append({"seed": seed, "method": method, "dir": exp_dir})

    # 2. 按 (method, seed) 分组，保留目录时间戳最新的一次（防重复运行覆盖）
    grouped = {}
    for run in candidates:
        key = (run["method"], run["seed"])
        if key not in grouped or run["dir"].name > grouped[key]["dir"].name:
            grouped[key] = run

    # 3. 提取数据 + 版本检测
    for (method, seed), run in sorted(grouped.items()):
        exp_dir = run["dir"]
        code_version = detect_code_version(exp_dir)
        
        print(f"Extracting: method={method}, seed={seed}, ver={code_version} | {exp_dir.name}")
        
        tb_data = extract_scalars_from_tb(exp_dir, SCALAR_TAGS)
        if tb_data is None:
            continue
            
        lr_history = extract_lr_history_json(exp_dir)
        
        all_data.setdefault(method, {})[seed] = {
            "tb": tb_data,
            "lr_history": lr_history,
            "code_version": code_version,  # 关键字段：标记数据来源
            "source_dir": str(exp_dir)
        }
        extracted_runs.append({
            "method": method,
            "seed": seed,
            "version": code_version,
            "dirname": exp_dir.name
        })

    # 4. 保存（带时间戳防覆盖）
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"scalars_{timestamp}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)

    # 5. 打印明细报告（区分新旧版本）
    print("\n" + "="*80)
    print("提取完成报告")
    print("="*80)
    print(f"输出文件: {output_file}")
    print(f"提取方法数: {len(all_data)}")
    print(f"总运行次数: {len(extracted_runs)}")
    
    # 按版本分组统计
    legacy_count = sum(1 for r in extracted_runs if r["version"] == "legacy")
    modern_count = sum(1 for r in extracted_runs if r["version"] == "modern")
    print(f"版本分布: 旧版代码 (legacy)={legacy_count} 次 | 新版代码 (modern)={modern_count} 次")
    print("-"*80)
    
    # 打印明细（按方法+Seed 排序）
    for r in sorted(extracted_runs, key=lambda x: (x["method"], x["seed"])):
        ver_tag = "legacy" if r["version"] == "legacy" else "modern"
        print(f"  - {r['method']:30} | seed={r['seed']:3} | {ver_tag:12} | {r['dirname']}")
    
    print("="*80)
    print("下游任务:")
    print(" 运行脚本:2_compress_scalars.py，压缩数据")


if __name__ == "__main__":
    main()