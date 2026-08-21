# scripts/compress_scalars.py
"""
数据压缩脚本：从完整版 scalars JSON 生成轻量版
默认仅保留核心评估指标 (FID/IS/KID)，可按需保留 Loss、机制标签与 lr_history。
适用于大规模 GAN 实验数据的职责分离与高效协作。

使用示例
默认压缩（仅保留 FID/IS/KID，移除 Loss/机制/lr_history）
python scripts/compress_scalars.py

保留机制标签（用于学习率轨迹验证）
python scripts/compress_scalars.py --keep-mechanism

保留 Loss（用于附录单 Seed 曲线）
python scripts/compress_scalars.py --keep-losses

保留 lr_history（仅调试用，体积 ~850MB）
python scripts/compress_scalars.py --keep-lr-history

定义输入输出路径
python scripts/compress_scalars.py extracted_scalars/scalars_2026****_******.json -o plot_data/core_only.json

"""
import json
import sys
from pathlib import Path
import argparse


def compress_scalars(input_file: Path, output_file: Path, 
                     keep_losses: bool = False, 
                     keep_mechanism: bool = False, 
                     keep_lr_history: bool = False):
    """
    压缩 TB 提取数据，过滤冗余标签以减小体积。
    
    Args:
        input_file: 完整版 scalars_*.json 路径
        output_file: 输出精简版路径
        keep_losses: 是否保留 Loss_G/Loss_D
        keep_mechanism: 是否保留阴阳机制相关标签 (Balance_*, LR_*_Scheduled)
        keep_lr_history: 是否保留 lr_history 详细轨迹（默认移除以节省 ~400MB）
    """
    print(f"Loading: {input_file.name} ({input_file.stat().st_size / 1024**2:.1f} MB)")
    
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. 定义保留的 TensorBoard 标签集合
    keep_tags = {"FID", "IS", "KID"}  # 核心评估指标（正文必需）
    if keep_losses:
        keep_tags.update({"Loss_G", "Loss_D"})
    if keep_mechanism:
        keep_tags.update({"Balance_LrG_Mult", "Balance_LrD_Mult", "LR_G_Scheduled", "LR_D_Scheduled"})

    compressed = {}
    removed_tb_count = 0
    total_tb_count = 0

    # 2. 遍历并过滤数据
    for method, seeds in data.items():
        compressed[method] = {}
        for seed, run_data in seeds.items():
            tb_data = run_data.get("tb", {})
            filtered_tb = {}

            for tag, values in tb_data.items():
                total_tb_count += 1
                if tag in keep_tags:
                    filtered_tb[tag] = values
                else:
                    removed_tb_count += 1

            # 3. 构建压缩记录（始终保留溯源元数据）
            comp_run = {
                "tb": filtered_tb,
                "code_version": run_data.get("code_version"),
                "source_dir": run_data.get("source_dir")
            }

            # 4. 按需保留 lr_history（默认移除）
            if keep_lr_history:
                lr_hist = run_data.get("lr_history")
                if lr_hist is not None:
                    comp_run["lr_history"] = lr_hist

            compressed[method][seed] = comp_run

    # 5. 保存：紧凑分隔符 + 缩进排版，兼顾体积与人工可读性
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(compressed, f, indent=2, ensure_ascii=False, separators=(",", ": "))

    # 6. 打印统计报告
    orig_size = input_file.stat().st_size / 1024**2
    new_size = output_file.stat().st_size / 1024**2
    total_runs = sum(len(seeds) for seeds in compressed.values())

    print(f"\nCompression complete:")
    print(f"   Output: {output_file.name} ({new_size:.1f} MB, {100*new_size/orig_size:.1f}% of original)")
    print(f"   Removed TB tags: {removed_tb_count}/{total_tb_count} ({100*removed_tb_count/total_tb_count:.1f}%)")
    print(f"   Retained TB tags: {sorted(list(keep_tags))}")
    print(f"   lr_history: {'Retained' if keep_lr_history else '🗑️ Removed (saves ~400MB)'}")
    print(f"   Methods: {sorted(list(compressed.keys()))}")
    print(f"   Total runs: {total_runs}")


def main():
    parser = argparse.ArgumentParser(
        description="将完整版 TB 数据压缩为轻量版，默认仅保留核心评估指标 (FID/IS/KID)。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "input", type=Path, nargs="?",
        default=Path("extracted_scalars/scalars_20260520_221206.json"),
        help="输入的完整版 JSON 文件路径\n(默认自动匹配最新提取结果)"
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="输出的精简版 JSON 路径\n(默认: 输入文件名 + _lite.json)"
    )
    parser.add_argument("--keep-losses", action="store_true", help="保留 Loss_G 与 Loss_D（附录曲线用）")
    parser.add_argument("--keep-mechanism", action="store_true", help="保留 Balance_* 与 LR_*_Scheduled 机制标签")
    parser.add_argument("--keep-lr-history", action="store_true", help="保留 lr_history 详细轨迹（体积显著增加）")

    args = parser.parse_args()

    # 自动查找最新文件（若默认路径不存在）
    if not args.input.exists():
        fallback_dir = Path("extracted_scalars")
        if fallback_dir.exists():
            candidates = sorted(fallback_dir.glob("scalars_*.json"))
            if candidates:
                args.input = candidates[-1]
                print(f"Default file not found. Using latest: {args.input.name}")
            else:
                print("No scalars files found in extracted_scalars/")
                sys.exit(1)
        else:
            print(f"Input file not found: {args.input}")
            sys.exit(1)

    # 默认输出路径
    if args.output is None:
        args.output = args.input.parent / f"{args.input.stem}_lite{args.input.suffix}"

    compress_scalars(
        args.input, args.output,
        keep_losses=args.keep_losses,
        keep_mechanism=args.keep_mechanism,
        keep_lr_history=args.keep_lr_history
    )


if __name__ == "__main__":
    main()