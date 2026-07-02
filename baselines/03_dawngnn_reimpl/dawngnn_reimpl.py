"""
DawnGNN-reimpl baseline for the EsCapturer experiment setup.

This is a minimal, reproducible implementation of the DawnGNN core idea:
API sequence -> transition graph -> documentation-style node semantics ->
2-layer Graph Attention Network -> binary classifier.

The data loading path intentionally reuses escapture_true.py so the dataset
scale and splits can match the existing EsCapturer runs.
"""

import argparse
import gzip
import json
import random
import re
import sys
import tarfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score

BASELINE_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = BASELINE_ROOT.parents[1]
ESCAPTURE_ROOT = WORKSPACE_ROOT / "escapture"
if str(ESCAPTURE_ROOT) not in sys.path:
    sys.path.insert(0, str(ESCAPTURE_ROOT))

from escapture_true import load_samples_from_path, load_seq_file


ZENODO_CALL_KEYS = ("vmi_FunctionName", "vmi_Function")


KNOWN_API_DESCRIPTIONS = {
    "CreateFile": "Opens or creates a file, device, directory, pipe, or other I/O object.",
    "ReadFile": "Reads data from a file or input/output device.",
    "WriteFile": "Writes data to a file or input/output device.",
    "CloseHandle": "Closes an open object handle.",
    "CreateProcess": "Creates a new process and its primary thread.",
    "VirtualAlloc": "Reserves or commits memory in the virtual address space of a process.",
    "VirtualProtect": "Changes protection on a region of committed memory.",
    "LoadLibrary": "Loads a dynamic-link library module into the process.",
    "GetProcAddress": "Retrieves the address of an exported function from a module.",
    "RegOpenKey": "Opens a registry key.",
    "RegSetValue": "Sets data for a registry value.",
    "NtCreateFile": "Creates or opens a file, directory, device, or volume object.",
    "NtOpenFile": "Opens an existing file, directory, device, or volume object.",
    "NtReadFile": "Reads data from an open file handle.",
    "NtWriteFile": "Writes data to an open file handle.",
    "NtClose": "Closes an object handle.",
    "NtCreateProcess": "Creates a process object.",
    "NtAllocateVirtualMemory": "Allocates virtual memory in a process address space.",
    "NtProtectVirtualMemory": "Changes virtual memory page protection.",
    "NtMapViewOfSection": "Maps a section object into a process address space.",
    "NtQueryInformationProcess": "Retrieves information about a process.",
    "NtSetInformationFile": "Sets information for a file object.",
}

ACTION_HINTS = {
    "create": "create or initialize an operating system object",
    "open": "open an existing operating system object",
    "read": "read data from an object",
    "write": "write data to an object",
    "close": "close or release a handle",
    "delete": "delete or remove an object",
    "query": "query metadata or status information",
    "set": "set metadata, configuration, or state",
    "get": "retrieve metadata, configuration, or state",
    "allocate": "allocate memory or resources",
    "free": "free memory or resources",
    "protect": "change access protection",
    "map": "map memory, files, or sections",
    "unmap": "unmap memory, files, or sections",
    "load": "load a module or resource",
    "connect": "connect to a local or network endpoint",
    "send": "send data through an endpoint",
    "recv": "receive data through an endpoint",
    "receive": "receive data through an endpoint",
    "socket": "create or operate on a network socket",
    "reg": "operate on the Windows registry",
    "process": "operate on a process",
    "thread": "operate on a thread",
    "file": "operate on a file system object",
}

OBJECT_HINTS = {
    "file": "file system object",
    "directory": "directory object",
    "process": "process object",
    "thread": "thread object",
    "memory": "virtual memory",
    "section": "memory section",
    "registry": "Windows registry",
    "key": "registry key or object key",
    "socket": "network socket",
    "service": "Windows service",
    "token": "security token",
    "event": "synchronization event",
    "mutant": "mutex synchronization object",
    "semaphore": "semaphore synchronization object",
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_api_name(name):
    text = re.sub(r"^(Nt|Zw|Rtl|Ldr|Api|Win32|kernel32|ntdll)[_.-]?", "", str(name), flags=re.I)
    text = re.sub(r"[^A-Za-z0-9]+", " ", text)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return [p.lower() for p in text.split() if p]


def describe_api(name, include_api_name=False):
    if name in KNOWN_API_DESCRIPTIONS:
        desc = KNOWN_API_DESCRIPTIONS[name]
        return f"{name}: {desc}" if include_api_name else desc
    parts = split_api_name(name)
    hints = []
    for part in parts:
        if part in ACTION_HINTS:
            hints.append(ACTION_HINTS[part])
        if part in OBJECT_HINTS:
            hints.append(OBJECT_HINTS[part])
    if not hints:
        hints = [f"Windows API or native system call named {' '.join(parts) or name}."]
    desc = "; ".join(dict.fromkeys(hints)) + "."
    return f"{name}: {desc}" if include_api_name else desc


def load_balanced_sequences(args):
    if args.compact_data:
        return load_compact_sequences(args.compact_data)

    if args.zenodo_archive:
        samples, attack_samples, normal_samples = load_zenodo_tar_sequences(args)
        random.shuffle(samples)
        return samples, attack_samples, normal_samples

    attack_samples, normal_samples = [], []
    if args.attack:
        attack_samples.extend(load_seq_file(args.attack))
    if args.normal:
        normal_samples.extend(load_seq_file(args.normal))
    for root in args.dataset_root:
        remaining_attack = max(args.max_attack_samples - len(attack_samples), 0) if args.max_attack_samples > 0 else 0
        remaining_normal = max(args.max_normal_samples - len(normal_samples), 0) if args.max_normal_samples > 0 else 0
        root_attack, root_normal = load_samples_from_path(
            root,
            window=args.window,
            stride=args.stride,
            max_attack_samples=remaining_attack,
            max_normal_samples=remaining_normal,
        )
        attack_samples.extend(root_attack)
        normal_samples.extend(root_normal)
    random.shuffle(attack_samples)
    random.shuffle(normal_samples)
    if args.max_attack_samples > 0:
        attack_samples = attack_samples[: args.max_attack_samples]
    if args.max_normal_samples > 0:
        normal_samples = normal_samples[: args.max_normal_samples]
    if not attack_samples or not normal_samples:
        raise ValueError("Need both attack and normal samples.")
    samples = [(seq, 1) for seq in attack_samples] + [(seq, 0) for seq in normal_samples]
    random.shuffle(samples)
    return samples, attack_samples, normal_samples


def load_compact_sequences(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    split_samples = {"train": [], "val": [], "test": []}
    attack_samples, normal_samples = [], []
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            seq = [str(api) for api in row.get("api_seq", []) if api]
            split = row.get("split")
            if len(seq) < 2 or split not in split_samples:
                continue
            raw_label = row.get("label")
            label = int(raw_label == "malware") if isinstance(raw_label, str) else int(raw_label)
            split_samples[split].append((seq, label))
            (attack_samples if label else normal_samples).append(seq)
    if any(not split_samples[name] for name in split_samples):
        raise ValueError(f"Compact dataset must contain non-empty train/val/test splits: {path}")
    return split_samples, attack_samples, normal_samples


def segment_trace(trace, window=200, stride=100):
    if len(trace) < 2:
        return []
    if len(trace) <= window:
        return [trace]
    samples = []
    for i in range(0, len(trace) - window + 1, stride):
        seg = trace[i : i + window]
        if len(seg) >= 2:
            samples.append(seg)
    return samples


def load_zenodo_metadata(metadata_path, seed):
    with open(metadata_path, "r", encoding="utf-8") as f:
        by_family = json.load(f)
    benign = list(dict.fromkeys(by_family.get("benign", [])))
    malware = []
    for family, shas in by_family.items():
        if family == "benign":
            continue
        malware.extend(shas)
    malware = list(dict.fromkeys(malware))
    rng = random.Random(seed)
    rng.shuffle(benign)
    rng.shuffle(malware)
    labels = {sha: 0 for sha in benign}
    labels.update({sha: 1 for sha in malware})
    return benign, malware, labels


def extract_zenodo_calls(fileobj):
    calls = []
    for raw in fileobj:
        try:
            event = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            continue
        if event.get("msg") and event.get("msg") != "Monitored function called":
            continue
        call = None
        for key in ZENODO_CALL_KEYS:
            value = event.get(key)
            if value:
                call = str(value).strip()
                break
        if call and call != "*" and len(call) <= 80:
            calls.append(re.sub(r"\s+", "_", call))
    return calls


def load_zenodo_tar_sequences(args):
    archive = Path(args.zenodo_archive)
    metadata = Path(args.zenodo_metadata)
    benign_shas, malware_shas, labels = load_zenodo_metadata(metadata, args.seed)
    candidates = set(benign_shas + malware_shas)
    normal_samples, attack_samples = [], []
    seen_files = {0: 0, 1: 0}

    print(f"    Zenodo archive: {archive}")
    print(f"    Zenodo metadata benign files: {len(benign_shas)}")
    print(f"    Zenodo metadata malware files: {len(malware_shas)}")

    with tarfile.open(archive, "r:xz") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".json"):
                continue
            sha = Path(member.name).stem
            if sha not in candidates:
                continue
            label = labels.get(sha)
            if label == 0 and len(normal_samples) >= args.max_normal_samples:
                continue
            if label == 1 and len(attack_samples) >= args.max_attack_samples:
                continue

            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            calls = extract_zenodo_calls(extracted)
            segments = segment_trace(calls, args.window, args.stride)
            if not segments:
                continue

            seen_files[label] += 1
            if label == 0:
                normal_samples.extend(segments)
                normal_samples = normal_samples[: args.max_normal_samples]
            else:
                attack_samples.extend(segments)
                attack_samples = attack_samples[: args.max_attack_samples]

            if len(normal_samples) >= args.max_normal_samples and len(attack_samples) >= args.max_attack_samples:
                break

    if len(normal_samples) < args.max_normal_samples or len(attack_samples) < args.max_attack_samples:
        raise ValueError(
            "Zenodo archive did not yield enough segmented samples: "
            f"normal={len(normal_samples)}/{args.max_normal_samples}, "
            f"attack={len(attack_samples)}/{args.max_attack_samples}."
        )

    print(f"    Zenodo files used: benign={seen_files[0]}, malware={seen_files[1]}")
    samples = [(seq, 1) for seq in attack_samples] + [(seq, 0) for seq in normal_samples]
    return samples, attack_samples, normal_samples


def build_doc_embeddings(vocab, dim, analyzer="word", include_api_name=False):
    descriptions = [describe_api(api, include_api_name=include_api_name) for api in vocab]
    if analyzer == "char_wb":
        vectorizer_kwargs = {"analyzer": "char_wb", "ngram_range": (3, 5)}
    else:
        vectorizer_kwargs = {"analyzer": "word", "ngram_range": (1, 2), "token_pattern": r"(?u)\b\w+\b"}
    vectorizer = HashingVectorizer(
        n_features=dim,
        alternate_sign=False,
        norm="l2",
        lowercase=True,
        **vectorizer_kwargs,
    )
    matrix = vectorizer.transform(descriptions).toarray().astype(np.float32)
    return torch.tensor(matrix, dtype=torch.float32), dict(zip(vocab, descriptions))


def sequence_to_graph(seq, api_to_idx):
    nodes = sorted(set(seq))
    local = {api: i for i, api in enumerate(nodes)}
    x_idx = torch.tensor([api_to_idx[api] for api in nodes], dtype=torch.long)

    counts = Counter()
    for src, dst in zip(seq[:-1], seq[1:]):
        counts[(local[src], local[dst])] += 1
    for i in range(len(nodes)):
        counts[(i, i)] += 1

    edge_pairs = list(counts.keys())
    edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor([counts[p] for p in edge_pairs], dtype=torch.float32)
    edge_weight = torch.log1p(edge_weight)
    return {"x_idx": x_idx, "edge_index": edge_index, "edge_weight": edge_weight, "label": int(seq is not None)}


class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, labeled_sequences, api_to_idx):
        self.graphs = []
        for seq, label in labeled_sequences:
            g = sequence_to_graph(seq, api_to_idx)
            g["label"] = label
            self.graphs.append(g)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


def collate_graphs(graphs):
    x_idx, edge_indices, edge_weights, batch, labels = [], [], [], [], []
    offset = 0
    for gid, g in enumerate(graphs):
        n = g["x_idx"].numel()
        x_idx.append(g["x_idx"])
        edge_indices.append(g["edge_index"] + offset)
        edge_weights.append(g["edge_weight"])
        batch.append(torch.full((n,), gid, dtype=torch.long))
        labels.append(g["label"])
        offset += n
    return {
        "x_idx": torch.cat(x_idx),
        "edge_index": torch.cat(edge_indices, dim=1),
        "edge_weight": torch.cat(edge_weights),
        "batch": torch.cat(batch),
        "y": torch.tensor(labels, dtype=torch.long),
    }


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4, dropout=0.2):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.dropout = dropout
        self.lin = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.att_src = nn.Parameter(torch.empty(heads, out_dim))
        self.att_dst = nn.Parameter(torch.empty(heads, out_dim))
        self.bias = nn.Parameter(torch.zeros(heads * out_dim))
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, x, edge_index, edge_weight):
        n = x.size(0)
        h = self.lin(x).view(n, self.heads, self.out_dim)
        src, dst = edge_index
        score = (h[src] * self.att_src).sum(-1) + (h[dst] * self.att_dst).sum(-1)
        score = F.leaky_relu(score, negative_slope=0.2) + edge_weight.unsqueeze(-1)

        alpha = torch.zeros_like(score)
        for head in range(self.heads):
            alpha[:, head] = torch.zeros_like(score[:, head]).scatter_reduce(
                0, dst, score[:, head], reduce="amax", include_self=False
            )
        exp_score = torch.exp(score - alpha[dst])
        denom = torch.zeros(n, self.heads, device=x.device).index_add_(0, dst, exp_score)
        attn = exp_score / (denom[dst] + 1e-9)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        out = torch.zeros(n, self.heads, self.out_dim, device=x.device)
        out.index_add_(0, dst, h[src] * attn.unsqueeze(-1))
        return out.reshape(n, self.heads * self.out_dim) + self.bias


class DawnGNN(nn.Module):
    def __init__(self, doc_features, hidden_dim=64, heads=4, dropout=0.2, readout="mean"):
        super().__init__()
        self.readout = readout
        self.register_buffer("doc_features", doc_features)
        in_dim = doc_features.size(1)
        self.input_proj = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU())
        self.gat1 = GATLayer(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout)
        self.gat2 = GATLayer(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout)
        classifier_dim = hidden_dim * 2 if readout == "mean_max" else hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(classifier_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, batch):
        x = self.doc_features[batch["x_idx"]]
        x = self.input_proj(x)
        x = F.elu(self.gat1(x, batch["edge_index"], batch["edge_weight"]))
        x = F.elu(self.gat2(x, batch["edge_index"], batch["edge_weight"]))

        graph_count = int(batch["batch"].max().item()) + 1
        mean_pool = torch.zeros(graph_count, x.size(1), device=x.device).index_add_(0, batch["batch"], x)
        counts = torch.bincount(batch["batch"], minlength=graph_count).float().unsqueeze(1).to(x.device)
        mean_pool = mean_pool / counts.clamp_min(1.0)
        if self.readout == "mean":
            return self.classifier(mean_pool)

        max_pool = torch.full((graph_count, x.size(1)), -1e9, device=x.device)
        max_pool = max_pool.scatter_reduce(
            0, batch["batch"].view(-1, 1).expand(-1, x.size(1)), x, reduce="amax", include_self=True
        )
        return self.classifier(torch.cat([mean_pool, max_pool], dim=1))


def move_batch(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def best_threshold(labels, probs):
    labels_np = np.array(labels)
    probs_np = np.array(probs)
    best_f1, best_t = -1.0, 0.5
    for t in np.linspace(0.05, 0.95, 181):
        preds = (probs_np >= t).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(labels_np, preds, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = float(f1), float(t)
    return best_t


def evaluate(model, loader, device, threshold=0.5, return_scores=False):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            logits = model(batch)
            prob = torch.softmax(logits, dim=1)[:, 1]
            probs.extend(prob.cpu().numpy().tolist())
            labels.extend(batch["y"].cpu().numpy().tolist())
    probs_np = np.array(probs)
    labels_np = np.array(labels)
    preds = (probs_np >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(labels_np, preds, average="binary", zero_division=0)
    acc = accuracy_score(labels_np, preds)
    tn, fp, fn, tp = confusion_matrix(labels_np, preds, labels=[0, 1]).ravel()
    try:
        auc = roc_auc_score(labels_np, probs_np)
    except ValueError:
        auc = 0.0
    result = {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "threshold": float(threshold),
    }
    if return_scores:
        result["scores"] = probs
        result["labels"] = labels
    return result


def main():
    parser = argparse.ArgumentParser(description="DawnGNN reimplementation baseline")
    parser.add_argument("--compact_data", help="Compact jsonl/jsonl.gz with existing train/val/test split fields")
    parser.add_argument("--attack")
    parser.add_argument("--normal")
    parser.add_argument("--zenodo_archive", help="Read Zenodo API_traces_malware_detection.tar.xz directly without extracting it")
    parser.add_argument(
        "--zenodo_metadata",
        default=str(WORKSPACE_ROOT / "datasets_50k/zenodo_11079764/data/raw/shas_by_families.json"),
        help="Zenodo shas_by_families.json used to assign benign/malware labels",
    )
    parser.add_argument(
        "--dataset_root",
        action="append",
        default=[str(WORKSPACE_ROOT / "datasets_50k/quo_vadis/data/raw")],
    )
    parser.add_argument("--window", type=int, default=200)
    parser.add_argument("--stride", type=int, default=100)
    parser.add_argument("--max_attack_samples", type=int, default=25000)
    parser.add_argument("--max_normal_samples", type=int, default=25000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--doc_dim", type=int, default=128)
    parser.add_argument("--doc_analyzer", choices=["word", "char_wb"], default="word")
    parser.add_argument("--include_api_name", action="store_true")
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--readout", choices=["mean", "mean_max"], default="mean")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--use_val_threshold", action="store_true")
    parser.add_argument(
        "--output",
        default=str(
            WORKSPACE_ROOT
            / "datasets_50k/quo_vadis/artifacts/dawngnn_reimpl/raw_result.json"
        ),
    )
    args = parser.parse_args()

    started = time.time()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("[1] Loading sequences with EsCapturer data loader...")
    loaded, attack_samples, normal_samples = load_balanced_sequences(args)
    if args.compact_data:
        split_samples = loaded
        samples = split_samples["train"] + split_samples["val"] + split_samples["test"]
    else:
        samples = loaded
    print(f"    Attack samples: {len(attack_samples)}")
    print(f"    Normal samples: {len(normal_samples)}")

    vocab = sorted({api for seq, _ in samples for api in seq})
    api_to_idx = {api: i for i, api in enumerate(vocab)}
    doc_features, descriptions = build_doc_embeddings(
        vocab,
        args.doc_dim,
        analyzer=args.doc_analyzer,
        include_api_name=args.include_api_name,
    )
    print(f"[2] API vocabulary: {len(vocab)}")
    print(
        f"    Documentation embedding: HashingVectorizer({args.doc_analyzer}), "
        f"dim={args.doc_dim}, include_api_name={args.include_api_name}"
    )

    if args.compact_data:
        train_set = GraphDataset(split_samples["train"], api_to_idx)
        val_set = GraphDataset(split_samples["val"], api_to_idx)
        test_set = GraphDataset(split_samples["test"], api_to_idx)
    else:
        dataset = GraphDataset(samples, api_to_idx)
        split1 = int(0.7 * len(dataset))
        split2 = int(0.85 * len(dataset))
        train_set = torch.utils.data.Subset(dataset, range(0, split1))
        val_set = torch.utils.data.Subset(dataset, range(split1, split2))
        test_set = torch.utils.data.Subset(dataset, range(split2, len(dataset)))
    print(f"[3] Split: train={len(train_set)} val={len(val_set)} test={len(test_set)}")

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_graphs)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_graphs)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_graphs)

    model = DawnGNN(
        doc_features,
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        dropout=args.dropout,
        readout=args.readout,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    print(f"[4] Parameters: {total_params:,}")

    history = {"epoch": [], "train_loss": [], "val_accuracy": [], "val_precision": [], "val_recall": [], "val_f1": [], "val_auc": []}
    best_state = None
    best_val_f1 = -1.0

    print("\n  Ep |     Loss |    Acc |      P |      R |     F1 |    AUC")
    print("-" * 62)
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad()
            logits = model(batch)
            loss = F.cross_entropy(logits, batch["y"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()

        val_m = evaluate(model, val_loader, device)
        avg_loss = float(np.mean(losses)) if losses else 0.0
        history["epoch"].append(epoch)
        history["train_loss"].append(avg_loss)
        history["val_accuracy"].append(val_m["accuracy"])
        history["val_precision"].append(val_m["precision"])
        history["val_recall"].append(val_m["recall"])
        history["val_f1"].append(val_m["f1"])
        history["val_auc"].append(val_m["auc"])
        if val_m["f1"] > best_val_f1:
            best_val_f1 = val_m["f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"{epoch:4d} | {avg_loss:8.4f} | {val_m['accuracy']:6.4f} | {val_m['precision']:6.4f} | {val_m['recall']:6.4f} | {val_m['f1']:6.4f} | {val_m['auc']:6.4f}")

    if best_state:
        model.load_state_dict(best_state)
    val_scores = evaluate(model, val_loader, device, return_scores=True)
    val_best_threshold = best_threshold(val_scores["labels"], val_scores["scores"])
    threshold = val_best_threshold if args.use_val_threshold else 0.5
    test_m = evaluate(model, test_loader, device, threshold=threshold, return_scores=True)

    print("\n[Test Results]")
    print(f"  Accuracy:  {test_m['accuracy']:.4f}")
    print(f"  Precision: {test_m['precision']:.4f}")
    print(f"  Recall:    {test_m['recall']:.4f}")
    print(f"  F1-Score:  {test_m['f1']:.4f}")
    print(f"  AUC:       {test_m['auc']:.4f}")
    print(f"  Threshold: {test_m['threshold']:.4f}")
    print(f"  TP={test_m['tp']} FP={test_m['fp']} FN={test_m['fn']} TN={test_m['tn']}")

    results = {
        "history": history,
        "test_metrics": {k: v for k, v in test_m.items() if k not in ("scores", "labels")},
        "score_distribution": {"scores": test_m["scores"], "labels": test_m["labels"]},
        "model_info": {
            "baseline": "DawnGNN-reimpl",
            "note": "API transition graph + fixed documentation-style API descriptions + hashed semantic node embeddings + 2-layer GAT.",
            "doc_embedding_backend": "sklearn HashingVectorizer over API documentation-style descriptions; deterministic fallback for BERT documentation embeddings",
            "doc_analyzer": args.doc_analyzer,
            "include_api_name": args.include_api_name,
            "use_val_threshold": args.use_val_threshold,
            "val_best_threshold": val_best_threshold,
            "total_params": total_params,
            "vocab_size": len(vocab),
            "doc_dim": args.doc_dim,
            "hidden_dim": args.hidden_dim,
            "heads": args.heads,
            "dropout": args.dropout,
            "readout": args.readout,
            "epochs": args.epochs,
            "window": args.window,
            "stride": args.stride,
            "max_attack_samples": args.max_attack_samples,
            "max_normal_samples": args.max_normal_samples,
            "n_train": len(train_set),
            "n_val": len(val_set),
            "n_test": len(test_set),
            "seed": args.seed,
            "runtime_seconds": round(time.time() - started, 3),
        },
        "api_descriptions": descriptions,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
