"""
EsCapturer-Inspired Syscall Sequence Anomaly Detection (v4 - Real Data)
========================================================================
改动:
- 删除所有硬编码的 SYSCALLS 列表、NORMAL_PATTERNS、ATTACK_PATTERNS
- Vocab 从输入序列文件动态构建
- Intention groups 从数据中自动发现（基于共现聚类）
- 读取由 gen_attack_seq.py / gen_normal_seq.py 生成的序列文件

输入文件格式（每行一个样本，调用间空格分隔）:
  read write sendmsg connect write fsync ...
  socket recvfrom write __queue_work ...

用法:
  python syscall_anomaly_detection.py \
      --attack attack_seq.txt \
      --normal normal_seq.txt \
      [--epochs 30] [--embed_dim 32] [--lr 5e-4]

 python escapture_true.py --attack attack_seq.txt  --normal normal_seq.txt --epochs 30  --embed_dim 32 --lr 5e-4
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter, defaultdict
import json
import math
import random
import argparse
import os
import csv
import re
from pathlib import Path

try:
    from escapture.llm_behavior_extractor import FrozenTemplateBehaviorExtractor
except ImportError:
    from llm_behavior_extractor import FrozenTemplateBehaviorExtractor


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 1. 数据加载 & 动态Vocab构建
# ============================================================

CALL_KEYS = {
    "api", "apis", "api_call", "api_calls", "apicall", "apicalls",
    "api_name",
    "call", "calls", "command", "commands", "syscall", "syscalls",
    "operation", "operations", "event", "events", "name", "function",
}


def _clean_call(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text or text == "*" or len(text) > 80:
        return None
    text = re.sub(r"\s+", "_", text)
    return text


def _extract_calls_from_json(obj):
    calls = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_norm = str(key).lower().replace(" ", "_")
            if key_norm in CALL_KEYS:
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, (dict, list)):
                            calls.extend(_extract_calls_from_json(item))
                        else:
                            call = _clean_call(item)
                            if call:
                                calls.append(call)
                elif isinstance(value, (dict, list)):
                    calls.extend(_extract_calls_from_json(value))
                else:
                    call = _clean_call(value)
                    if call:
                        calls.append(call)
            elif isinstance(value, (dict, list)):
                calls.extend(_extract_calls_from_json(value))
    elif isinstance(obj, list):
        for item in obj:
            calls.extend(_extract_calls_from_json(item))
    return calls


def _infer_label_from_path(path):
    parts = [p.lower() for p in Path(path).parts]
    joined = "/".join(parts)
    normal_words = ("benign", "normal", "goodware", "clean", "legit", "windows_syswow64")
    attack_words = (
        "malware", "malicious", "attack", "anorm", "virus", "trojan", "ransom",
        "ransomware", "rat", "coinminer", "keylogger", "dropper", "backdoor",
    )
    if any(word in joined for word in normal_words):
        return 0
    if any(word in joined for word in attack_words):
        return 1
    return None


def _segment(trace, window=200, stride=100):
    if len(trace) < 2:
        return []
    if len(trace) <= window:
        return [trace]
    samples = []
    for i in range(0, len(trace) - window + 1, stride):
        seg = trace[i:i + window]
        if len(seg) >= 2:
            samples.append(seg)
    return samples


def load_seq_file(path):
    """
    加载序列文件。每行一个样本，系统调用用空格分隔。
    返回: list of list[str]
    """
    path = Path(path)
    samples = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                calls = [_clean_call(x) for x in re.split(r"[\s|,]+", line)]
                calls = [x for x in calls if x]
                if len(calls) >= 2:
                    samples.append(calls)
    return samples


def load_samples_from_path(path, window=200, stride=100, max_attack_samples=0, max_normal_samples=0):
    """读取 Zenodo/Kaggle 解压目录，尽量从 txt/csv/json/jsonl 中抽取 API/syscall 序列。"""
    root = Path(path)
    files = [root] if root.is_file() else [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in {".txt", ".log", ".csv", ".json", ".jsonl"}
    ]
    attack_samples, normal_samples = [], []

    for file_path in files:
        suffix = file_path.suffix.lower()
        label = _infer_label_from_path(file_path)
        if label == 0 and max_normal_samples > 0 and len(normal_samples) >= max_normal_samples:
            continue
        if label == 1 and max_attack_samples > 0 and len(attack_samples) >= max_attack_samples:
            continue
        traces = []

        try:
            if suffix in {".txt", ".log"}:
                traces = load_seq_file(file_path)
            elif suffix == ".csv":
                rows_by_file = defaultdict(list)
                with open(file_path, "r", encoding="utf-8-sig", errors="ignore") as f:
                    reader = csv.DictReader(f)
                    for row_id, row in enumerate(reader):
                        call = None
                        for key, value in row.items():
                            if key and key.lower().replace(" ", "_") in CALL_KEYS:
                                call = _clean_call(value)
                                if call:
                                    break
                        if not call:
                            continue
                        sample_key = (
                            row.get("sha") or row.get("hash") or row.get("file") or
                            row.get("filename") or row.get("sample") or str(row_id // window)
                        )
                        rows_by_file[sample_key].append(call)
                for trace in rows_by_file.values():
                    traces.extend(_segment(trace, window, stride))
            elif suffix == ".jsonl":
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        calls = _extract_calls_from_json(json.loads(line))
                        traces.extend(_segment(calls, window, stride))
            elif suffix == ".json":
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    data = json.load(f)
                calls = _extract_calls_from_json(data)
                traces.extend(_segment(calls, window, stride))
        except (OSError, json.JSONDecodeError, csv.Error):
            continue

        if label == 0:
            normal_samples.extend(traces)
            if max_normal_samples > 0:
                normal_samples = normal_samples[:max_normal_samples]
        elif label == 1:
            attack_samples.extend(traces)
            if max_attack_samples > 0:
                attack_samples = attack_samples[:max_attack_samples]

        attack_done = max_attack_samples > 0 and len(attack_samples) >= max_attack_samples
        normal_done = max_normal_samples > 0 and len(normal_samples) >= max_normal_samples
        if attack_done and normal_done:
            break

    return attack_samples, normal_samples


class SyscallVocab:
    """从数据动态构建词汇表，不依赖任何硬编码列表"""

    def __init__(self, embed_dim=32):
        self.syscall2idx = {"<PAD>": 0}
        self.idx2syscall = {0: "<PAD>"}
        self.embed_dim = embed_dim
        self.vocab_size = 1

    def build_from_samples(self, *sample_lists):
        """从多个样本列表中收集所有唯一syscall，构建词表"""
        all_calls = set()
        for samples in sample_lists:
            for seq in samples:
                all_calls.update(seq)

        for sc in sorted(all_calls):
            if sc not in self.syscall2idx:
                idx = len(self.syscall2idx)
                self.syscall2idx[sc] = idx
                self.idx2syscall[idx] = sc

        self.vocab_size = len(self.syscall2idx)

    def encode(self, syscall):
        return self.syscall2idx.get(syscall, 0)

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"syscall2idx": self.syscall2idx, "embed_dim": self.embed_dim}, f)

    def load(self, path):
        with open(path, "r") as f:
            data = json.load(f)
        self.syscall2idx = data["syscall2idx"]
        self.idx2syscall = {v: k for k, v in self.syscall2idx.items()}
        self.embed_dim = data["embed_dim"]
        self.vocab_size = len(self.syscall2idx)


# ============================================================
# 2. 自动意图分组（从数据发现，不硬编码）
# ============================================================

def auto_discover_intention_groups(all_samples, n_groups=5):
    """
    基于 syscall 之间的共现关系自动聚类出意图组。
    统计 bigram 共现 -> 贪心聚类，不管输入什么数据都能自动发现行为模式。
    """
    cooccur = defaultdict(lambda: defaultdict(int))
    call_freq = Counter()

    for seq in all_samples:
        for i in range(len(seq) - 1):
            a, b = seq[i], seq[i + 1]
            cooccur[a][b] += 1
            cooccur[b][a] += 1
        for s in seq:
            call_freq[s] += 1

    all_calls = sorted(call_freq.keys())
    if len(all_calls) <= n_groups:
        return {f"group_{i}": {c} for i, c in enumerate(all_calls)}

    def similarity(a, b):
        co = cooccur[a].get(b, 0) + cooccur[b].get(a, 0)
        total = call_freq[a] + call_freq[b]
        return co / max(total, 1)

    # 贪心聚类：高频先做种子
    assigned = {}
    groups = {}
    group_id = 0
    sorted_calls = sorted(all_calls, key=lambda c: -call_freq[c])

    for call in sorted_calls:
        if call in assigned:
            continue

        best_group = None
        best_sim = -1
        for gname, members in groups.items():
            avg_sim = np.mean([similarity(call, m) for m in members])
            if avg_sim > best_sim:
                best_sim = avg_sim
                best_group = gname

        if best_group is not None and best_sim > 0.01 and len(groups[best_group]) < 15:
            groups[best_group].add(call)
            assigned[call] = best_group
        else:
            if group_id < n_groups:
                gname = f"group_{group_id}"
                groups[gname] = {call}
                assigned[call] = gname
                group_id += 1
            else:
                if best_group is not None:
                    groups[best_group].add(call)
                    assigned[call] = best_group
                else:
                    gname = f"group_{group_id - 1}"
                    groups[gname].add(call)
                    assigned[call] = gname

    return {k: v for k, v in groups.items()}


class IntentionMapper:
    """动态意图映射器"""

    def __init__(self):
        self.groups = {}
        self.call_to_group = {}
        self.group_names = []

    def build(self, all_samples, n_groups=5):
        self.groups = auto_discover_intention_groups(all_samples, n_groups)
        self.group_names = sorted(self.groups.keys())
        self.call_to_group = {}
        for gname, members in self.groups.items():
            for call in members:
                self.call_to_group[call] = gname

    def get_intention(self, syscall):
        return self.call_to_group.get(syscall, "other")

    def save(self, path):
        data = {k: sorted(v) for k, v in self.groups.items()}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path):
        with open(path, "r") as f:
            data = json.load(f)
        self.groups = {k: set(v) for k, v in data.items()}
        self.group_names = sorted(self.groups.keys())
        self.call_to_group = {}
        for gname, members in self.groups.items():
            for call in members:
                self.call_to_group[call] = gname


# ============================================================
# 3. 样本构造（序列 → groups 结构，兼容原模型接口）
# ============================================================

def seq_to_sample(seq, intention_mapper, label, n_groups_per_sample=5, semantic_extractor=None, sample_id="sample"):
    """
    将一条 syscall 序列转换为模型需要的 sample 格式:
    { "groups": [{"intention": ..., "syscalls": [...]}, ...], "label": 0/1 }
    按 intention 对序列进行连续分段。
    """
    if not seq:
        return {"groups": [{"intention": "other", "syscalls": ["<PAD>"]}], "label": label}

    if semantic_extractor is not None:
        semantic_units = semantic_extractor.build_units(seq, sample_id=sample_id, max_units=n_groups_per_sample)
        groups = [unit.to_group() for unit in semantic_units]
        if groups:
            return {
                "groups": groups,
                "label": label,
                "semantic_extractor": semantic_extractor.TEMPLATE_LIBRARY_VERSION,
            }

    intentions = [intention_mapper.get_intention(s) for s in seq]

    # 按 intention 连续分段
    raw_groups = []
    cur_intent = intentions[0]
    cur_calls = [seq[0]]

    for i in range(1, len(seq)):
        if intentions[i] == cur_intent:
            cur_calls.append(seq[i])
        else:
            raw_groups.append({"intention": cur_intent, "syscalls": cur_calls})
            cur_intent = intentions[i]
            cur_calls = [seq[i]]
    raw_groups.append({"intention": cur_intent, "syscalls": cur_calls})

    # 合并过短的相邻同类组
    merged = []
    for g in raw_groups:
        if merged and len(g["syscalls"]) < 3 and merged[-1]["intention"] == g["intention"]:
            merged[-1]["syscalls"].extend(g["syscalls"])
        else:
            merged.append(g)

    offset = 0
    for g in merged:
        g["start"] = offset
        offset += len(g["syscalls"])
        g["end"] = offset

    # 组太多时保留最长的，再按原始位置恢复顺序
    if len(merged) > n_groups_per_sample:
        merged.sort(key=lambda g: -len(g["syscalls"]))
        merged = merged[:n_groups_per_sample]
        merged.sort(key=lambda g: g["start"])

    if not merged:
        merged = [{"intention": "other", "syscalls": seq[:10]}]

    return {"groups": merged, "label": label}


def build_dataset(attack_samples, normal_samples, intention_mapper, semantic_extractor=None):
    dataset = []
    for idx, seq in enumerate(attack_samples):
        dataset.append(seq_to_sample(
            seq, intention_mapper, label=1,
            semantic_extractor=semantic_extractor,
            sample_id=f"attack_{idx}",
        ))
    for idx, seq in enumerate(normal_samples):
        dataset.append(seq_to_sample(
            seq, intention_mapper, label=0,
            semantic_extractor=semantic_extractor,
            sample_id=f"normal_{idx}",
        ))
    random.shuffle(dataset)
    return dataset


# ============================================================
# 4. Graph Construction
# ============================================================

def build_intention_graph(syscall_seq, vocab):
    unique_syscalls = list(set(syscall_seq))
    node_indices = {sc: i for i, sc in enumerate(unique_syscalls)}
    n_nodes = len(unique_syscalls)
    node_ids = [vocab.encode(sc) for sc in unique_syscalls]

    edge_features = {}
    for i in range(len(syscall_seq) - 1):
        src, dst = syscall_seq[i], syscall_seq[i + 1]
        si, di = node_indices[src], node_indices[dst]
        key = (si, di)
        if key not in edge_features:
            edge_features[key] = {"freq": 0, "positions": []}
        edge_features[key]["freq"] += 1
        edge_features[key]["positions"].append(i)

    edge_index, edge_attr = [], []
    for (s, d), feat in edge_features.items():
        edge_index.append([s, d])
        freq = feat["freq"] / max(len(syscall_seq), 1)
        mean_pos = np.mean(feat["positions"]) / max(len(syscall_seq), 1)
        std_pos = (np.std(feat["positions"]) / max(len(syscall_seq), 1)) if len(feat["positions"]) > 1 else 0.0
        edge_attr.append([freq, mean_pos, std_pos])

    adj = np.zeros((n_nodes, n_nodes))
    for (s, d) in edge_features.keys():
        adj[s][d] = 1
        adj[d][s] = 1

    spd = np.full((n_nodes, n_nodes), -1, dtype=np.int64)
    for i in range(n_nodes):
        visited = {i}
        queue = [(i, 0)]
        spd[i][i] = 0
        while queue:
            node, dist = queue.pop(0)
            for j in range(n_nodes):
                if adj[node][j] > 0 and j not in visited:
                    spd[i][j] = min(dist + 1, 5)
                    visited.add(j)
                    queue.append((j, dist + 1))

    if not edge_index:
        edge_index = [[0, 0]]
        edge_attr = [[0.0, 0.0, 0.0]]

    return {
        "node_ids": node_ids, "n_nodes": n_nodes,
        "edge_index": edge_index, "edge_attr": edge_attr, "spd": spd,
    }


# ============================================================
# 5. Sequential Feature Encoder
# ============================================================

class SequentialEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, nhead=4, num_layers=2, max_seq_len=200):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, embed_dim) * 0.02)
        self.freq_proj = nn.Linear(1, embed_dim)
        hidden = embed_dim * 2
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=nhead, dim_feedforward=hidden * 2,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out_proj = nn.Linear(hidden, embed_dim)

    def forward(self, token_ids, freq_features):
        d = self.embedding(token_ids)
        seq_len = d.size(1)
        d = d + self.pos_encoding[:, :seq_len, :]
        c = self.freq_proj(freq_features)
        x = torch.cat([c, d], dim=-1)
        h = self.transformer(x)
        h = h.mean(dim=1)
        return self.out_proj(h)


# ============================================================
# 6. Graph Transformer Encoder
# ============================================================

class GraphTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead=4, edge_dim=3, max_spd=7):
        super().__init__()
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_E = nn.Linear(edge_dim, nhead)
        self.O_h = nn.Linear(d_model, d_model)
        self.spd_bias = nn.Embedding(max_spd + 2, nhead)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(),
            nn.Linear(d_model * 4, d_model), nn.Dropout(0.1),
        )

    def forward(self, h, edge_index, edge_attr, spd_matrix):
        n = h.size(0)
        residual = h
        Q = self.W_Q(h).view(n, self.nhead, self.d_k)
        K = self.W_K(h).view(n, self.nhead, self.d_k)
        V = self.W_V(h).view(n, self.nhead, self.d_k)

        attn = torch.einsum("imk,jmk->ijm", Q, K) / math.sqrt(self.d_k)

        spd_clamped = spd_matrix.clamp(-1, 6) + 1
        spd_b = self.spd_bias(spd_clamped.long())
        attn = attn + spd_b

        if len(edge_index) > 0 and edge_attr.size(0) > 0:
            eb = self.W_E(edge_attr)
            for idx, (s, d) in enumerate(edge_index):
                attn[s, d] += eb[idx]

        attn = F.softmax(attn, dim=1)
        out = torch.einsum("ijm,jmk->imk", attn, V).reshape(n, -1)
        out = self.O_h(out)

        h = self.norm1(out + residual)
        h = self.norm2(self.ffn(h) + h)
        return h


class GraphEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, nhead=4, num_layers=2, edge_dim=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.layers = nn.ModuleList([
            GraphTransformerLayer(embed_dim, nhead, edge_dim) for _ in range(num_layers)
        ])

    def forward(self, node_ids, edge_index, edge_attr, spd_matrix):
        h = self.embedding(node_ids)
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr, spd_matrix)
        return h.mean(dim=0)


# ============================================================
# 7. Interleaving-prior Adaptive View Fusion
# ============================================================

def randomize_symmetric_binary_matrix(c, seed):
    """Preserve the upper-triangle density while randomizing pair locations."""
    n = c.size(0)
    if n <= 1:
        return c
    upper = torch.triu_indices(n, n, offset=1, device=c.device)
    values = c[upper[0], upper[1]]
    ones = int(values.sum().item())
    randomized = torch.zeros_like(c)
    if ones:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        selected = torch.randperm(values.numel(), generator=generator)[:ones].to(c.device)
        rows = upper[0][selected]
        cols = upper[1][selected]
        randomized[rows, cols] = 1.0
        randomized[cols, rows] = 1.0
    randomized.fill_diagonal_(float(c.diagonal().max().item()) if c.numel() else 0.0)
    return randomized


def build_relation_tensors(groups, device, prior_mode="real", random_seed=7):
    n = len(groups)
    c = torch.zeros(n, n, dtype=torch.float, device=device)
    r = torch.zeros(n, n, 7, dtype=torch.float, device=device)
    max_span = max((g.get("end", i + 1) for i, g in enumerate(groups)), default=1)

    for i, gi in enumerate(groups):
        si, ei = gi.get("start", i), gi.get("end", i + 1)
        calls_i = set(gi.get("syscalls", []))
        object_i = gi.get("object", gi.get("resource", gi.get("intention", "unknown")))
        subject_i = gi.get("subject", "process")
        type_i = gi.get("template_id", gi.get("intention", "unknown"))
        for j, gj in enumerate(groups):
            sj, ej = gj.get("start", j), gj.get("end", j + 1)
            calls_j = set(gj.get("syscalls", []))
            object_j = gj.get("object", gj.get("resource", gj.get("intention", "unknown")))
            subject_j = gj.get("subject", "process")
            type_j = gj.get("template_id", gj.get("intention", "unknown"))
            same_intention = float(gi.get("intention") == gj.get("intention"))
            overlap = max(0, min(ei, ej) - max(si, sj))
            union_span = max(ei, ej) - min(si, sj)
            overlap_ratio = overlap / max(union_span, 1)
            jaccard = len(calls_i & calls_j) / max(len(calls_i | calls_j), 1)
            same_object = float(object_i == object_j)
            same_process = float(subject_i == subject_j)
            same_type = float(type_i == type_j)
            non_adjacent_same_intention = same_intention and abs(i - j) > 1
            has_interleave = (
                (overlap_ratio > 0 and i != j)
                or non_adjacent_same_intention
                or (same_object and abs(i - j) > 1)
            )
            c[i, j] = float(has_interleave)
            r[i, j, 0] = abs(sj - ei) / max(max_span, 1)  # delta_t
            r[i, j, 1] = overlap_ratio
            r[i, j, 2] = jaccard
            r[i, j, 3] = same_object
            r[i, j, 4] = same_process
            r[i, j, 5] = float(has_interleave)
            r[i, j, 6] = same_type

    if prior_mode == "none":
        c.zero_()
        r[..., 5].zero_()
    elif prior_mode == "random":
        c = randomize_symmetric_binary_matrix(c, random_seed)
        r[..., 5] = c
    elif prior_mode != "real":
        raise ValueError(f"Unsupported prior_mode: {prior_mode}")

    return c, r


class InterleavingPriorAdaptiveFusion(nn.Module):
    def __init__(
        self,
        embed_dim=32,
        nhead=4,
        num_layers=1,
        beta=1.0,
        relation_dim=7,
        use_gating_prior=True,
        use_structural_bias=True,
        use_relation_features=True,
        hard_switching=False,
        weighting_mode="softmax",
        gating_temperature=1.0,
    ):
        super().__init__()
        self.beta = beta
        self.use_gating_prior = use_gating_prior
        self.use_structural_bias = use_structural_bias
        self.use_relation_features = use_relation_features
        self.hard_switching = hard_switching
        self.weighting_mode = weighting_mode
        if gating_temperature <= 0:
            raise ValueError("gating_temperature must be positive")
        self.gating_temperature = float(gating_temperature)
        if weighting_mode not in {"softmax", "sigmoid"}:
            raise ValueError(f"Unsupported weighting_mode: {weighting_mode}")
        seq_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=embed_dim * 4,
            dropout=0.1, batch_first=True
        )
        self.seq_transformer = nn.TransformerEncoder(seq_layer, num_layers=num_layers)

        hidden = max(embed_dim, 16)
        self.seq_score = nn.Sequential(
            nn.Linear(embed_dim * 2 + relation_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.graph_score = nn.Sequential(
            nn.Linear(embed_dim * 2 + relation_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.query = nn.Linear(embed_dim * 2, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.interleave_bias = nn.Embedding(2, 1)
        self.temporal_bias = nn.Linear(1, 1)
        self.object_bias = nn.Linear(2, 1)
        self.edge_type_bias = nn.Linear(2, 1)
        self.norm = nn.LayerNorm(embed_dim)
        self.out_norm = nn.LayerNorm(embed_dim * 2)
        self.last_pref_loss = None
        self.last_view_weights = None

    def forward(self, h_g_seq, h_s_seq, c_matrix, relation_features):
        h_s_seq = self.seq_transformer(h_s_seq)
        bsz, n_units, dim = h_s_seq.shape
        hs_i = h_s_seq.unsqueeze(2).expand(bsz, n_units, n_units, dim)
        hs_j = h_s_seq.unsqueeze(1).expand(bsz, n_units, n_units, dim)
        hg_i = h_g_seq.unsqueeze(2).expand(bsz, n_units, n_units, dim)
        hg_j = h_g_seq.unsqueeze(1).expand(bsz, n_units, n_units, dim)
        rel = relation_features.unsqueeze(0).expand(bsz, n_units, n_units, -1)
        score_rel = rel if self.use_relation_features else torch.zeros_like(rel)
        c = c_matrix.unsqueeze(0).expand(bsz, n_units, n_units)

        a_s = self.seq_score(torch.cat([hs_i, hs_j, score_rel], dim=-1)).squeeze(-1)
        a_g = self.graph_score(torch.cat([hg_i, hg_j, score_rel], dim=-1)).squeeze(-1)
        if self.use_gating_prior:
            a_s = a_s + self.beta * (1.0 - c)
            a_g = a_g + self.beta * c
        if self.weighting_mode == "sigmoid":
            w_g_scalar = torch.sigmoid(
                (a_g - a_s) / self.gating_temperature
            )
            view_weights = torch.stack([1.0 - w_g_scalar, w_g_scalar], dim=-1)
        else:
            view_weights = F.softmax(
                torch.stack([a_s, a_g], dim=-1) / self.gating_temperature,
                dim=-1,
            )
        if self.hard_switching:
            selected = view_weights.argmax(dim=-1)
            view_weights = F.one_hot(selected, num_classes=2).to(view_weights.dtype)
        w_s = view_weights[..., 0].unsqueeze(-1)
        w_g = view_weights[..., 1].unsqueeze(-1)
        v_ij = w_s * hs_j + w_g * hg_j

        q = self.query(torch.cat([h_s_seq, h_g_seq], dim=-1)).unsqueeze(2)
        k = self.key(v_ij)
        score = (q * k).sum(dim=-1) / math.sqrt(dim)
        if self.use_structural_bias:
            score = score + self.interleave_bias(c.long()).squeeze(-1)
            score = score + self.temporal_bias(rel[..., 0:1]).squeeze(-1)
            score = score + self.object_bias(rel[..., 3:5]).squeeze(-1)
            score = score + self.edge_type_bias(rel[..., 5:7]).squeeze(-1)
        alpha = F.softmax(score, dim=-1)

        aggregated = torch.sum(alpha.unsqueeze(-1) * self.value_proj(v_ij), dim=2)
        aggregated = self.norm(aggregated + h_g_seq)
        pooled = aggregated.mean(dim=1)
        global_context = 0.5 * (h_s_seq.mean(dim=1) + h_g_seq.mean(dim=1))
        out = self.out_norm(torch.cat([pooled, global_context], dim=-1))

        if self.hard_switching:
            preference_prob = F.softmax(
                torch.stack([a_s, a_g], dim=-1) / self.gating_temperature,
                dim=-1,
            )[..., 1]
        else:
            preference_prob = view_weights[..., 1]
        self.last_pref_loss = F.binary_cross_entropy(preference_prob, c)
        self.last_view_weights = view_weights.detach()
        return out


# ============================================================
# 8. DSVDD Head（LayerNorm）
# ============================================================

class DSVDDHead(nn.Module):
    def __init__(self, input_dim, output_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(output_dim * 2, output_dim),
        )
        self.register_buffer("center", torch.zeros(output_dim))

    def forward(self, x):
        return self.net(x)

    def dist_to_center(self, z):
        return torch.sum((z - self.center) ** 2, dim=-1)


# ============================================================
# 9. Full Model
# ============================================================

class EsCapturer(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=32,
        nhead=4,
        output_dim=32,
        max_seq_len=200,
        beta=1.0,
        use_sequence_view=True,
        use_graph_view=True,
        prior_mode="real",
        hard_switching=False,
        use_gating_prior=True,
        use_structural_bias=True,
        use_relation_features=True,
        weighting_mode="softmax",
        gating_temperature=1.0,
        random_seed=7,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab = None
        self.use_sequence_view = use_sequence_view
        self.use_graph_view = use_graph_view
        self.prior_mode = prior_mode
        self.random_seed = random_seed
        if not use_sequence_view and not use_graph_view:
            raise ValueError("At least one representation view must be enabled")

        self.seq_encoder = SequentialEncoder(vocab_size, embed_dim, nhead, num_layers=2, max_seq_len=max_seq_len)
        self.graph_encoder = GraphEncoder(vocab_size, embed_dim, nhead, num_layers=2)
        self.sequence_replacement = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.graph_replacement = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.fusion = InterleavingPriorAdaptiveFusion(
            embed_dim,
            nhead,
            beta=beta,
            use_gating_prior=use_gating_prior,
            use_structural_bias=use_structural_bias,
            use_relation_features=use_relation_features,
            hard_switching=hard_switching,
            weighting_mode=weighting_mode,
            gating_temperature=gating_temperature,
        )
        self.dsvdd = DSVDDHead(embed_dim * 2, output_dim)
        self.classifier = nn.Linear(embed_dim * 2, 1)

    def encode_sample(self, sample, device, return_pref_loss=False):
        groups = sample["groups"]
        h_s_list, h_g_list = [], []

        for group in groups:
            syscalls = group["syscalls"]

            token_ids = [self.vocab.encode(s) for s in syscalls]
            freq_counter = Counter(syscalls)
            total = len(syscalls)
            freq_features = [freq_counter[s] / total for s in syscalls]

            tok_t = torch.tensor([token_ids], dtype=torch.long, device=device)
            freq_t = torch.tensor([[f] for f in freq_features], dtype=torch.float, device=device).unsqueeze(0)
            graph = build_intention_graph(syscalls, self.vocab)
            nids = torch.tensor(graph["node_ids"], dtype=torch.long, device=device)
            eattr = torch.tensor(graph["edge_attr"], dtype=torch.float, device=device)
            spd = torch.tensor(graph["spd"], dtype=torch.long, device=device)
            h_s = (
                self.seq_encoder(tok_t, freq_t).squeeze(0)
                if self.use_sequence_view
                else None
            )
            h_g = (
                self.graph_encoder(nids, graph["edge_index"], eattr, spd)
                if self.use_graph_view
                else None
            )
            if h_s is None:
                h_s = self.sequence_replacement(h_g)
            if h_g is None:
                h_g = self.graph_replacement(h_s)
            h_s_list.append(h_s)
            h_g_list.append(h_g)

        h_s_seq = torch.stack(h_s_list).unsqueeze(0)
        h_g_seq = torch.stack(h_g_list).unsqueeze(0)
        c_matrix, relation_features = build_relation_tensors(
            groups,
            device,
            prior_mode=self.prior_mode,
            random_seed=self.random_seed,
        )
        h_d = self.fusion(h_g_seq, h_s_seq, c_matrix, relation_features)
        if return_pref_loss:
            return h_d, self.fusion.last_pref_loss
        return h_d

    def forward_dsvdd(self, sample, device, return_pref_loss=False):
        encoded = self.encode_sample(sample, device, return_pref_loss=return_pref_loss)
        if return_pref_loss:
            h_d, pref_loss = encoded
        else:
            h_d, pref_loss = encoded, None
        z = self.dsvdd(h_d)
        if return_pref_loss:
            return z, self.dsvdd.dist_to_center(z), pref_loss
        return z, self.dsvdd.dist_to_center(z)

    def forward_classifier(self, sample, device, return_pref_loss=False):
        encoded = self.encode_sample(
            sample, device, return_pref_loss=return_pref_loss
        )
        if return_pref_loss:
            h_d, pref_loss = encoded
        else:
            h_d, pref_loss = encoded, None
        logit = self.classifier(h_d).squeeze(-1)
        if return_pref_loss:
            return logit, pref_loss
        return logit


# ============================================================
# 10. Training
# ============================================================

def train_model(model, train_data, val_data, device, epochs=30, lr=5e-4, lambda_pref=0.1):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    attack_train = [s for s in train_data if s["label"] == 1]
    normal_train = [s for s in train_data if s["label"] == 0]

    print("=" * 90)
    print(f"  Train: {len(attack_train)} attack, {len(normal_train)} normal")
    print("=" * 90)

    # Phase 1: Initialize center
    print("\n[Phase 1] Initializing DSVDD center...")
    model.eval()
    vecs = []
    center_seed = attack_train[:40] if attack_train else train_data[:40]
    with torch.no_grad():
        for s in center_seed:
            h_d = model.encode_sample(s, device)
            z = model.dsvdd(h_d)
            vecs.append(z.squeeze(0))
    if not vecs:
        raise ValueError("训练集为空，无法初始化 DSVDD center")
    center = torch.stack(vecs).mean(dim=0)
    center[(center.abs() < 0.01)] = 0.01
    model.dsvdd.center.copy_(center)
    print(f"  Center norm: {center.norm().item():.4f}")

    # Phase 2: Train
    print("\n[Phase 2] Semi-supervised DSVDD training...\n")
    header = (f"{'Ep':>4} | {'Loss':>8} | {'L_atk':>7} | {'L_nrm':>7} | "
              f"{'Pref':>7} | {'F1':>6} | {'Acc':>6} | {'P':>6} | {'R':>6} | "
              f"{'NrmDist':>8} | {'AtkDist':>8}")
    print(header)
    print("-" * len(header))

    history = {"epoch": [], "train_loss": [], "pref_loss": [], "val_f1": [], "val_acc": [],
               "val_precision": [], "val_recall": []}

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_atk, total_nrm, total_pref = 0.0, 0.0, 0.0, 0.0
        n = 0

        random.shuffle(attack_train)
        random.shuffle(normal_train)
        n_pairs = min(len(attack_train), len(normal_train))

        for i in range(n_pairs):
            optimizer.zero_grad()

            z_atk, dist_atk, pref_atk = model.forward_dsvdd(attack_train[i], device, return_pref_loss=True)
            loss_atk = dist_atk.mean()

            z_nrm, dist_nrm, pref_nrm = model.forward_dsvdd(normal_train[i], device, return_pref_loss=True)
            margin = max(dist_atk.detach().mean().item() + 2.0, 3.0)
            loss_nrm = F.relu(margin - dist_nrm).mean()
            loss_pref = 0.5 * (pref_atk + pref_nrm)

            loss = loss_atk + loss_nrm + lambda_pref * loss_pref
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_atk += loss_atk.item()
            total_nrm += loss_nrm.item()
            total_pref += loss_pref.item()
            n += 1

        scheduler.step()
        avg_loss = total_loss / max(n, 1)
        avg_atk = total_atk / max(n, 1)
        avg_nrm = total_nrm / max(n, 1)
        avg_pref = total_pref / max(n, 1)

        metrics = evaluate(model, val_data, device)
        history["epoch"].append(epoch)
        history["train_loss"].append(avg_loss)
        history["pref_loss"].append(avg_pref)
        history["val_f1"].append(metrics["f1"])
        history["val_acc"].append(metrics["accuracy"])
        history["val_precision"].append(metrics["precision"])
        history["val_recall"].append(metrics["recall"])

        print(f"{epoch:4d} | {avg_loss:8.4f} | {avg_atk:7.4f} | {avg_nrm:7.4f} | "
              f"{avg_pref:7.4f} | "
              f"{metrics['f1']:6.4f} | {metrics['accuracy']:6.4f} | "
              f"{metrics['precision']:6.4f} | {metrics['recall']:6.4f} | "
              f"{metrics['normal_avg_dist']:8.4f} | {metrics['attack_avg_dist']:8.4f}")

    print("=" * 90)
    return history


def evaluate(model, data, device, threshold=None):
    model.eval()
    scores, labels = [], []

    with torch.no_grad():
        for sample in data:
            _, dist = model.forward_dsvdd(sample, device)
            scores.append(dist.item())
            labels.append(sample["label"])

    scores = np.array(scores)
    labels = np.array(labels)

    normal_dists = scores[labels == 0]
    attack_dists = scores[labels == 1]

    if threshold is None:
        best_f1, best_t = 0.0, np.median(scores)
        for pct in range(5, 96):
            t = np.percentile(scores, pct)
            preds = (scores < t).astype(int)
            f1 = _f1(labels, preds)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        threshold = best_t

    preds = (scores < threshold).astype(int)
    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    accuracy = (tp + tn) / len(labels)

    return {
        "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy,
        "threshold": float(threshold),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "scores": scores.tolist(), "labels": labels.tolist(),
        "normal_avg_dist": float(normal_dists.mean()) if len(normal_dists) > 0 else 0.0,
        "attack_avg_dist": float(attack_dists.mean()) if len(attack_dists) > 0 else 0.0,
    }


def _f1(labels, preds):
    tp = np.sum((preds == 1) & (labels == 1))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return 2 * p * r / max(p + r, 1e-8)


# ============================================================
# 11. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="EsCapturer Syscall Anomaly Detection with interleaving-prior view selection")
    parser.add_argument("--attack", help="攻击/异常序列文件路径")
    parser.add_argument("--normal", help="正常序列文件路径")
    parser.add_argument("--dataset_root", action="append", default=[],
                        help="Zenodo/Kaggle 解压目录，可重复传入；路径中含 benign/normal 视为正常，含 malware/malicious 视为异常")
    parser.add_argument("--window", type=int, default=200, help="目录数据滑窗大小")
    parser.add_argument("--stride", type=int, default=100, help="目录数据滑窗步长")
    parser.add_argument("--max_attack_samples", type=int, default=0, help="只取前 N 个异常样本，0 表示不限制")
    parser.add_argument("--max_normal_samples", type=int, default=0, help="只取前 N 个正常样本，0 表示不限制")
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数")
    parser.add_argument("--embed_dim", type=int, default=32, help="嵌入维度")
    parser.add_argument("--lr", type=float, default=5e-4, help="学习率")
    parser.add_argument("--n_groups", type=int, default=5, help="自动意图分组数")
    parser.add_argument("--beta", type=float, default=1.0, help="交织先验偏置强度")
    parser.add_argument("--lambda_pref", type=float, default=0.1, help="视图选择偏好损失权重")
    parser.add_argument("--seed", type=int, default=7, help="随机种子")
    parser.add_argument(
        "--semantic_extractor",
        choices=["template", "none"],
        default="template",
        help="使用可复现的 LLM-assisted 行为要素模板抽取层；none 为消融版本",
    )
    parser.add_argument("--output", default="results.json", help="结果输出路径")
    args = parser.parse_args()

    set_random_seed(args.seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output = str(output_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # --- 加载数据 ---
    print("[1] 加载序列文件...")
    attack_samples, normal_samples = [], []
    if args.attack:
        attack_samples.extend(load_seq_file(args.attack))
    if args.normal:
        normal_samples.extend(load_seq_file(args.normal))
    for root in args.dataset_root:
        remaining_attack = max(args.max_attack_samples - len(attack_samples), 0) if args.max_attack_samples > 0 else 0
        remaining_normal = max(args.max_normal_samples - len(normal_samples), 0) if args.max_normal_samples > 0 else 0
        root_attack, root_normal = load_samples_from_path(
            root, args.window, args.stride,
            max_attack_samples=remaining_attack,
            max_normal_samples=remaining_normal,
        )
        attack_samples.extend(root_attack)
        normal_samples.extend(root_normal)
    random.shuffle(attack_samples)
    random.shuffle(normal_samples)
    if args.max_attack_samples > 0:
        attack_samples = attack_samples[:args.max_attack_samples]
    if args.max_normal_samples > 0:
        normal_samples = normal_samples[:args.max_normal_samples]
    print(f"    攻击样本: {len(attack_samples)}")
    print(f"    正常样本: {len(normal_samples)}")
    if not attack_samples or not normal_samples:
        raise ValueError("需要同时提供异常和正常样本。可使用 --attack/--normal，或使用带 benign/malware 路径标记的 --dataset_root。")

    # --- 动态构建词表 ---
    print("\n[2] 动态构建词汇表...")
    vocab = SyscallVocab(embed_dim=args.embed_dim)
    vocab.build_from_samples(attack_samples, normal_samples)
    print(f"    词表大小: {vocab.vocab_size}")
    print(f"    包含: {list(vocab.syscall2idx.keys())[:20]}...")

    # --- 自动发现意图分组 ---
    print(f"\n[3] 自动发现意图分组 (n_groups={args.n_groups})...")
    intention_mapper = IntentionMapper()
    all_seqs = attack_samples + normal_samples
    intention_mapper.build(all_seqs, n_groups=args.n_groups)
    for gname, members in intention_mapper.groups.items():
        print(f"    {gname}: {sorted(members)}")

    semantic_extractor = None
    if args.semantic_extractor == "template":
        cache_path = args.output.replace(".json", "_llm_template_cache.json")
        semantic_extractor = FrozenTemplateBehaviorExtractor(cache_path)
        print("\n[3.5] LLM-assisted 行为要素抽取层启用")
        print(f"    mode: frozen template library ({semantic_extractor.TEMPLATE_LIBRARY_VERSION})")
        print(f"    cache: {cache_path}")
        print("    输出字段: subject, operation, object, resource, context, goal, template_id")
    else:
        print("\n[3.5] LLM-assisted 行为要素抽取层关闭（w/o LLM elements 消融）")

    # --- 构建数据集 ---
    print("\n[4] 构建数据集...")
    dataset = build_dataset(attack_samples, normal_samples, intention_mapper, semantic_extractor)
    if semantic_extractor is not None:
        semantic_extractor.save_cache()
    n_attack = sum(1 for d in dataset if d["label"] == 1)
    n_normal = sum(1 for d in dataset if d["label"] == 0)
    print(f"    总样本: {len(dataset)} (攻击: {n_attack}, 正常: {n_normal})")

    split1 = int(0.7 * len(dataset))
    split2 = int(0.85 * len(dataset))
    train_data = dataset[:split1]
    val_data = dataset[split1:split2]
    test_data = dataset[split2:]
    print(f"    Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}\n")

    # --- 模型 ---
    max_seq_len = 20
    for s in dataset:
        for g in s["groups"]:
            max_seq_len = max(max_seq_len, len(g["syscalls"]))
    max_seq_len = min(max_seq_len, 512)

    model = EsCapturer(vocab.vocab_size, embed_dim=args.embed_dim, nhead=4,
                       output_dim=args.embed_dim, max_seq_len=max_seq_len,
                       beta=args.beta).to(device)
    model.vocab = vocab

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
    print(f"Max seq len: {max_seq_len}\n")

    # --- 训练 ---
    history = train_model(model, train_data, val_data, device,
                          epochs=args.epochs, lr=args.lr,
                          lambda_pref=args.lambda_pref)

    # --- 测试 ---
    print("\n[Test Results]")
    print("-" * 50)
    test_m = evaluate(model, test_data, device)
    print(f"  Accuracy:  {test_m['accuracy']:.4f}")
    print(f"  Precision: {test_m['precision']:.4f}")
    print(f"  Recall:    {test_m['recall']:.4f}")
    print(f"  F1-Score:  {test_m['f1']:.4f}")
    print(f"  Threshold: {test_m['threshold']:.4f}")
    print(f"  TP={test_m['tp']} FP={test_m['fp']} FN={test_m['fn']} TN={test_m['tn']}")
    print(f"  Normal avg dist:  {test_m['normal_avg_dist']:.4f}")
    print(f"  Attack avg dist:  {test_m['attack_avg_dist']:.4f}")

    # --- 保存 ---
    results = {
        "history": history,
        "test_metrics": {k: v for k, v in test_m.items() if k not in ("scores", "labels")},
        "score_distribution": {"scores": test_m["scores"], "labels": test_m["labels"]},
        "model_info": {
            "total_params": total_params,
            "vocab_size": vocab.vocab_size,
            "embed_dim": args.embed_dim,
            "n_groups": args.n_groups,
            "beta": args.beta,
            "lambda_pref": args.lambda_pref,
            "seed": args.seed,
            "semantic_extractor": args.semantic_extractor,
            "max_seq_len": max_seq_len,
            "n_train": len(train_data),
            "n_val": len(val_data),
            "n_test": len(test_data),
        },
        "intention_groups": {k: sorted(v) for k, v in intention_mapper.groups.items()},
    }
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    vocab.save(args.output.replace(".json", "_vocab.json"))
    intention_mapper.save(args.output.replace(".json", "_intentions.json"))
    if semantic_extractor is not None:
        semantic_extractor.save_cache()

    print(f"\nResults saved to {args.output}")
    print(f"Vocab saved to {args.output.replace('.json', '_vocab.json')}")
    print(f"Intentions saved to {args.output.replace('.json', '_intentions.json')}")
    print("\nDone!")


if __name__ == "__main__":
    main()
