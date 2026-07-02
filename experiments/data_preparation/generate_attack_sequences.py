"""
gen_attack_seq.py
=================
从 CSV 异常文件 生成系统调用序列文件（攻击/异常）

CSV 格式: time,proc,command,direct,content,time_proc
提取 command 列作为 syscall 名称

输出格式（每行一个样本，调用间空格分隔）:
  read write sendmsg connect write fsync ...
  socket recvfrom write __queue_work ...

用法:
  python gen_attack_seq.py <input.csv> <output.txt> [--window 200] [--stride 100]
  python experiments/data_preparation/generate_attack_sequences.py anorm_train.csv attack_seq.txt --window 200 --stride 100

"""

import csv
import argparse
import sys


def load_csv_commands(csv_path):
    """读取CSV，提取command列，跳过空值和通配符"""
    commands = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cmd = row.get("command", "").strip()
            if cmd and cmd != "*":
                commands.append(cmd)
    return commands


def segment(trace, window, stride):
    """滑窗切分成固定长度样本"""
    samples = []
    for i in range(0, max(1, len(trace) - window + 1), stride):
        seg = trace[i : i + window]
        if len(seg) >= 20:
            samples.append(seg)
    if not samples and len(trace) >= 20:
        samples.append(trace)
    return samples


def main():
    parser = argparse.ArgumentParser(description="CSV异常文件 → 系统调用序列")
    parser.add_argument("input", help="输入CSV文件路径")
    parser.add_argument("output", help="输出序列文件路径")
    parser.add_argument("--window", type=int, default=200, help="滑窗大小 (default: 200)")
    parser.add_argument("--stride", type=int, default=100, help="滑窗步长 (default: 100)")
    args = parser.parse_args()

    print(f"[1] 读取CSV: {args.input}")
    commands = load_csv_commands(args.input)
    unique = sorted(set(commands))
    print(f"    总调用数: {len(commands)}")
    print(f"    唯一调用: {len(unique)} 种 → {unique}")

    print(f"[2] 滑窗切分: window={args.window}, stride={args.stride}")
    samples = segment(commands, args.window, args.stride)
    print(f"    生成样本数: {len(samples)}")

    print(f"[3] 写入: {args.output}")
    with open(args.output, "w") as f:
        for sample in samples:
            f.write(" ".join(sample) + "\n")

    print(f"    完成！每行一个样本，调用间空格分隔")


if __name__ == "__main__":
    main()
