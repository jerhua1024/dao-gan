# compute_plot_data.py
# 全 7 机制正文版：直接使用指定 lite 文件，计算全部 7 机制的 combined 分组
import json
from pathlib import Path
import numpy as np
import datetime

# === 🔧 硬编码配置 ===
INPUT_FILE = Path("extracted_scalars/scalars_20260520_221206_lite.json")
OUTPUT_DIR = Path("plot_data")
OUTPUT_DIR.mkdir(exist_ok=True)

# 🔹 全部 7 个机制模式，均聚合全部seeds（论文正文必需）
GROUPS = {
    "baseline/combined": {"42", "77", "85", "99", "123", "156", "200", "314", "404", "521"},
    "daosheng_temp07/combined": {"42", "77", "85", "99", "123", "156", "200", "314", "404", "521"},
    "daosheng_temp08/combined": {"42", "77", "85", "99", "123", "156", "200", "314", "404", "521"},
    "daosheng_temp085/combined": {"42", "77", "85", "99", "123", "156", "200", "314", "404", "521"},
    "daosheng_temp09/combined": {"42", "77", "85", "99", "123", "156", "200", "314", "404", "521"},
    "yinyang_gradscale/combined": {"42", "77", "85", "99", "123", "156", "200", "314", "404", "521"},
    "yinyang_daosheng/combined": {"42", "77", "85", "99", "123", "156", "200", "314", "404", "521"},
}

METRICS = ["FID", "IS", "KID"]  # 与 lite 文件对齐


def compute_stats(seeds_data):
    """对齐多 Seed 并计算 mean ± std（极简实现）"""
    result = {}
    for metric in METRICS:
        steps_list, vals_list = [], []
        for seed_data in seeds_data.values():
            tb = seed_data.get("tb", seed_data)
            if metric in tb:
                s, v = tb[metric].get("steps"), tb[metric].get("values")
                if s and v and len(s) == len(v):
                    steps_list.append(np.array(s))
                    vals_list.append(np.array(v))
        
        if not steps_list:
            result[metric] = {"steps": [], "mean": [], "std": [], "n_seeds": 0}
            continue
        
        # 步数交集对齐
        common = set(steps_list[0])
        for s in steps_list[1:]: common &= set(s)
        common = sorted(int(x) for x in common)
        
        if not common:
            result[metric] = {"steps": [], "mean": [], "std": [], "n_seeds": 0}
            continue
        
        # 收集对齐后的数值
        aligned = []
        for step in common:
            vals = [vals_list[i][np.where(steps_list[i] == step)[0][0]] 
                   for i in range(len(steps_list)) 
                   if np.any(steps_list[i] == step)]
            if vals: aligned.append(vals)
        
        if not aligned:
            result[metric] = {"steps": common, "mean": [], "std": [], "n_seeds": 0}
            continue
        
        arr = np.array(aligned)  # (T, N)
        n = arr.shape[1]
        result[metric] = {
            "steps": common,
            "mean": np.mean(arr, axis=1).tolist(),
            "std": np.std(arr, axis=1, ddof=1 if n > 1 else 0).tolist(),
            "n_seeds": n
        }
    return result


def main():
    # 1. 加载数据
    if not INPUT_FILE.exists():
        print(f"文件不存在: {INPUT_FILE}")
        return
    print(f"加载: {INPUT_FILE.name} ({INPUT_FILE.stat().st_size/1024:.1f} KB)")
    
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        all_data = json.load(f)  
    
    # 2. 计算统计量
    plot_data = {}
    for group_key, seed_set in GROUPS.items():
        method = group_key.split("/")[0]
        if method not in all_data: 
            print(f"跳过: {method} 未找到")
            continue

        # 检查缺失的 Seed
        present_seeds = set(all_data[method].keys())
        missing = seed_set - present_seeds
        if missing:
            print(f"{method} 缺失的 Seed: {sorted(missing, key=int)}")

        group_data = {s: all_data[method][s] for s in seed_set if s in all_data.get(method, {})}
        if not group_data:  
            print(f"跳过: {group_key} 无有效数据")
            continue
            
        print(f"计算 {group_key}: N={len(group_data)} seeds")
        plot_data[group_key] = compute_stats(group_data)
    
    # 3. 保存结果（带时间戳防覆盖）
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = OUTPUT_DIR / f"plot_data_{ts}.json"
    with open(output, "w", encoding="utf-8") as f:
        json.dump(plot_data, f, indent=2, ensure_ascii=False)
    
    # 4. 打印报告
    print(f"\n 完成: {output}")
    for k, v in plot_data.items():
        fid = v.get("FID", {})
        print(f"   {k:35} | FID(N={fid.get('n_seeds', 0):2}) | 步数={len(fid.get('steps', [])):2}")
    
    print("💡 下游任务:")
    print("   运行脚本:python scripts/4_plot_fid_facet_paper.py，绘图")


if __name__ == "__main__":
    main()