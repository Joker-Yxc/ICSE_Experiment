"""
gen_normal_seq.py
=================
从 pipe-delimited 的 .log 文件 生成系统调用序列文件（正常）

LOG 格式: 单行或多行，系统调用之间用 | 分隔
  execve|brk|arch_prctl|access|openat|newfstatat|mmap|close|...

输出格式（每行一个样本，调用间空格分隔）:
  execve brk arch_prctl access openat newfstatat mmap close ...
  mmap close openat read pread64 ...

用法:
  python gen_normal_seq.py <input.log> <output.txt> [--window 200] [--stride 100]
  python experiments/data_preparation/generate_normal_sequences.py sy_annexc.log normal_seq.txt --window 200 --stride 100
"""

import argparse


def load_log_syscalls(log_path):
    """读取pipe分隔的log文件，支持单行或多行"""
    with open(log_path, "r") as f:
        raw = f.read().strip()
    # 统一处理：换行也当作分隔符
    raw = raw.replace("\n", "|").replace("\r", "")
    calls = [c.strip() for c in raw.split("|") if c.strip()]
    return calls


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
    parser = argparse.ArgumentParser(description="LOG正常文件 → 系统调用序列")
    parser.add_argument("input", help="输入.log文件路径")
    parser.add_argument("output", help="输出序列文件路径")
    parser.add_argument("--window", type=int, default=200, help="滑窗大小 (default: 200)")
    parser.add_argument("--stride", type=int, default=100, help="滑窗步长 (default: 100)")
    args = parser.parse_args()

    print(f"[1] 读取LOG: {args.input}")
    calls = load_log_syscalls(args.input)
    unique = sorted(set(calls))
    print(f"    总调用数: {len(calls)}")
    print(f"    唯一调用: {len(unique)} 种 → {unique}")

    print(f"[2] 滑窗切分: window={args.window}, stride={args.stride}")
    samples = segment(calls, args.window, args.stride)
    print(f"    生成样本数: {len(samples)}")

    print(f"[3] 写入: {args.output}")
    with open(args.output, "w") as f:
        for sample in samples:
            f.write(" ".join(sample) + "\n")

    print(f"    完成！每行一个样本，调用间空格分隔")


if __name__ == "__main__":
    main()
