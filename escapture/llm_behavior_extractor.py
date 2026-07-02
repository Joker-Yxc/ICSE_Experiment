#!/usr/bin/env python3
"""Reproducible LLM-assisted behavior element extraction.

The paper revision plan uses an LLM as a semantic parser, not as the final
malware classifier. For repeatable experiments this module implements a frozen
template library distilled from the LLM extraction schema and exposes the same
structured fields that an online LLM call would return:

    subject, operation, object, resource, context, goal, template_id

The implementation is deterministic, privacy-preserving, and compact. It keeps
the original API-name sequence as the primary signal while adding semantic
tokens/behavior units that can be consumed by our method and by ablation code.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class BehaviorElement:
    sample_id: str
    index: int
    api_name: str
    subject: str
    operation: str
    object: str
    resource: str
    context: str
    goal: str
    template_id: str

    def semantic_tokens(self) -> list[str]:
        return [
            f"SUBJ={self.subject}",
            f"OP={self.operation}",
            f"OBJ={self.object}",
            f"RES={self.resource}",
            f"CTX={self.context}",
            f"GOAL={self.goal}",
            f"TPL={self.template_id}",
        ]


@dataclass(frozen=True)
class BehaviorUnit:
    unit_id: str
    start: int
    end: int
    subject: str
    operation: str
    object: str
    resource: str
    context: str
    goal: str
    template_id: str
    api_seq: list[str]

    def to_group(self) -> dict:
        return {
            "unit_id": self.unit_id,
            "start": self.start,
            "end": self.end,
            "intention": self.goal,
            "subject": self.subject,
            "operation": self.operation,
            "object": self.object,
            "resource": self.resource,
            "context": self.context,
            "goal": self.goal,
            "template_id": self.template_id,
            "syscalls": self.api_seq,
        }


class FrozenTemplateBehaviorExtractor:
    """Local, deterministic extractor following the LLM schema.

    The keyword templates act as the reproducible template library requested in
    the revision plan. An online LLM can be added later behind this interface,
    but the default experiment stays stable and does not leak paths, addresses,
    or labels.
    """

    TEMPLATE_LIBRARY_VERSION = "llm_template_v1"

    _RULES = [
        ("file_write", ("write", "copy", "move", "replace", "setfile", "flush", "createfile", "createdirectory", "tempfilename"), "file", "file_update", "filesystem_modification"),
        ("file_read", ("read", "findresource", "loadresource", "lockresource", "getfile", "getmodulefilename", "querydirectory", "path"), "file", "file_read", "data_collection"),
        ("file_delete", ("delete", "remove"), "file", "file_delete", "filesystem_modification"),
        ("registry_modify", ("regset", "regcreate", "regdelete", "regopen", "regquery", "shgetfolderpath"), "registry", "registry_access", "persistence"),
        ("network_connect", ("connect", "send", "recv", "internet", "http", "dns", "socket", "wsastartup", "url"), "network", "network_communication", "c2_or_exfiltration"),
        ("process_exec", ("createprocess", "shellexecute", "winexec", "system", "commandline", "exitprocess"), "process", "process_execution", "payload_execution"),
        ("process_control", ("openprocess", "terminat", "thread", "remote", "virtualallocex", "writeprocessmemory", "createremotethread"), "process", "process_manipulation", "privilege_or_injection"),
        ("memory_exec", ("virtualalloc", "virtualprotect", "mapview", "mmap", "heapalloc", "heapfree", "loadlibrary", "getprocaddress"), "memory", "memory_or_library", "payload_preparation"),
        ("permission_change", ("chmod", "setsecurity", "adjusttoken", "privilege", "acl", "token"), "permission", "permission_change", "privilege_operation"),
        ("ipc_sync", ("pipe", "mutex", "event", "semaphore", "criticalsection", "waitforsingleobject", "fls", "tls"), "ipc", "ipc_or_sync", "execution_coordination"),
        ("environment_probe", ("getsystem", "getcurrent", "queryperformance", "isprocessor", "getversion", "verifyversion", "getstartup", "getstdhandle", "getacp", "getcpinfo"), "system", "environment_probe", "anti_analysis_or_setup"),
        ("encoding_transform", ("multibytetowidechar", "widechartomultibyte", "lcmapstring", "getstringtype"), "system", "encoding_transform", "data_preparation"),
        ("error_state", ("getlasterror", "setlasterror"), "system", "error_state", "control_flow"),
    ]

    def __init__(self, cache_path: str | Path | None = None):
        self.cache_path = Path(cache_path) if cache_path else None
        self._cache: dict[str, dict] = {}
        if self.cache_path and self.cache_path.exists():
            try:
                self._cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self._cache = {}

    def save_cache(self) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self._cache, indent=2, sort_keys=True), encoding="utf-8")

    @staticmethod
    def normalize_api(api_name: str) -> tuple[str, str]:
        text = str(api_name or "").strip()
        if "." in text:
            module, func = text.rsplit(".", 1)
        else:
            module, func = "unknown", text
        module = re.sub(r"[^A-Za-z0-9_]+", "_", module).strip("_").lower() or "unknown"
        func = re.sub(r"[^A-Za-z0-9_]+", "_", func).strip("_")
        return module, func or "unknown_call"

    @staticmethod
    def desensitize(value: str) -> str:
        text = str(value or "")
        text = re.sub(r"[A-Za-z]:\\[^\\/:*?\"<>|\s]+(?:\\[^\\/:*?\"<>|\s]+)*", "<PATH>", text)
        text = re.sub(r"/(?:[^/\s]+/)+[^/\s]+", "<PATH>", text)
        text = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "<IP_ADDR>", text)
        text = re.sub(r"\b[0-9a-fA-F]{32,64}\b", "<HASH>", text)
        return text

    def extract_api(self, api_name: str, sample_id: str = "sample", index: int = 0) -> BehaviorElement:
        api_name = self.desensitize(api_name)
        cache_key = api_name.lower()
        if cache_key in self._cache:
            cached = dict(self._cache[cache_key])
            return BehaviorElement(sample_id=sample_id, index=index, api_name=api_name, **cached)

        module, func = self.normalize_api(api_name)
        func_l = func.lower()
        template_id, resource, operation, goal = "generic_call", "system", "system_call", "execution_context"
        for rule_id, keywords, rule_resource, rule_operation, rule_goal in self._RULES:
            if any(keyword in func_l for keyword in keywords):
                template_id, resource, operation, goal = rule_id, rule_resource, rule_operation, rule_goal
                break

        context = self._context_for(module, func_l, resource, operation)
        obj = self._object_for(resource, func_l)
        payload = {
            "subject": f"process:{module}",
            "operation": operation,
            "object": obj,
            "resource": resource,
            "context": context,
            "goal": goal,
            "template_id": template_id,
        }
        self._cache[cache_key] = payload
        return BehaviorElement(sample_id=sample_id, index=index, api_name=api_name, **payload)

    def extract_sequence(self, api_seq: Iterable[str], sample_id: str = "sample", max_len: int = 5000) -> list[BehaviorElement]:
        return [
            self.extract_api(api, sample_id=sample_id, index=i)
            for i, api in enumerate(list(api_seq)[:max_len])
        ]

    @staticmethod
    def _context_for(module: str, func_l: str, resource: str, operation: str) -> str:
        if "ex" in func_l or "remote" in func_l:
            return "extended_or_remote_context"
        if resource == "network":
            return "external_object_context"
        if resource in {"file", "registry"}:
            return "persistent_object_context"
        if operation in {"memory_or_library", "payload_preparation"}:
            return "runtime_loading_context"
        if module in {"ntdll", "advapi32"}:
            return "native_or_privileged_context"
        return "local_process_context"

    @staticmethod
    def _object_for(resource: str, func_l: str) -> str:
        if resource == "file":
            return "<FILE_OBJECT>"
        if resource == "network":
            return "<NETWORK_ENDPOINT>"
        if resource == "process":
            return "<PROCESS_OR_THREAD>"
        if resource == "memory":
            return "<MEMORY_REGION_OR_DLL>"
        if resource == "registry":
            return "<REGISTRY_KEY>"
        if resource == "permission":
            return "<SECURITY_OBJECT>"
        if resource == "ipc":
            return "<IPC_OBJECT>"
        if "error" in func_l:
            return "<ERROR_STATE>"
        return "<SYSTEM_OBJECT>"

    def build_units(
        self,
        api_seq: Iterable[str],
        sample_id: str = "sample",
        max_len: int = 5000,
        max_unit_len: int = 64,
        max_units: int = 128,
        unit_selection: str = "prefix",
    ) -> list[BehaviorUnit]:
        elements = self.extract_sequence(api_seq, sample_id=sample_id, max_len=max_len)
        return self.build_units_from_elements(
            elements,
            sample_id=sample_id,
            max_unit_len=max_unit_len,
            max_units=max_units,
            unit_selection=unit_selection,
        )

    def build_units_from_elements(
        self,
        elements: list[BehaviorElement],
        sample_id: str = "sample",
        max_unit_len: int = 64,
        max_units: int = 128,
        unit_selection: str = "prefix",
    ) -> list[BehaviorUnit]:
        if not elements:
            return []
        if max_units < 1:
            raise ValueError("max_units must be at least 1")
        if unit_selection not in {"prefix", "uniform-cover"}:
            raise ValueError(f"Unsupported unit_selection: {unit_selection}")

        if unit_selection == "uniform-cover":
            # Divide the complete trace into contiguous, near-equal spans. This
            # guarantees that increasing semantic diversity cannot silently
            # discard the tail of a trace when the unit budget is reached.
            unit_count = min(max_units, len(elements))
            boundaries = [
                round(index * len(elements) / unit_count)
                for index in range(unit_count + 1)
            ]
            return [
                self._make_unit(
                    sample_id,
                    unit_index,
                    elements[boundaries[unit_index] : boundaries[unit_index + 1]],
                )
                for unit_index in range(unit_count)
                if boundaries[unit_index + 1] > boundaries[unit_index]
            ]

        units: list[BehaviorUnit] = []
        current = [elements[0]]
        for element in elements[1:]:
            prev = current[-1]
            same_semantic_run = (
                element.template_id == prev.template_id
                or (element.resource == prev.resource and element.goal == prev.goal)
            )
            if same_semantic_run and len(current) < max_unit_len:
                current.append(element)
            else:
                units.append(self._make_unit(sample_id, len(units), current))
                current = [element]
                if len(units) >= max_units - 1:
                    break
        if current and len(units) < max_units:
            units.append(self._make_unit(sample_id, len(units), current))
        return units

    @staticmethod
    def _majority(values: Iterable[str]) -> str:
        counter = Counter(values)
        return counter.most_common(1)[0][0] if counter else "unknown"

    def _make_unit(self, sample_id: str, unit_index: int, elements: list[BehaviorElement]) -> BehaviorUnit:
        unit_hash = hashlib.sha1(f"{sample_id}:{unit_index}:{elements[0].index}".encode("utf-8")).hexdigest()[:12]
        return BehaviorUnit(
            unit_id=f"u{unit_index}_{unit_hash}",
            start=elements[0].index,
            end=elements[-1].index + 1,
            subject=self._majority(e.subject for e in elements),
            operation=self._majority(e.operation for e in elements),
            object=self._majority(e.object for e in elements),
            resource=self._majority(e.resource for e in elements),
            context=self._majority(e.context for e in elements),
            goal=self._majority(e.goal for e in elements),
            template_id=self._majority(e.template_id for e in elements),
            api_seq=[e.api_name for e in elements],
        )


def semantic_tokens_for_row(row: dict, extractor: FrozenTemplateBehaviorExtractor, max_len: int = 5000) -> list[str]:
    sample_id = str(row.get("sample_id", "sample"))
    tokens: list[str] = []
    for element in extractor.extract_sequence(row.get("api_seq", [])[:max_len], sample_id=sample_id, max_len=max_len):
        tokens.extend(element.semantic_tokens())
    return tokens


def semantic_text_for_row(row: dict, extractor: FrozenTemplateBehaviorExtractor, max_len: int = 5000) -> str:
    return " ".join(semantic_tokens_for_row(row, extractor, max_len=max_len))


def build_extraction_summary(
    rows: list[dict],
    extractor: FrozenTemplateBehaviorExtractor,
    max_rows: int | None = None,
    max_len: int = 5000,
) -> dict:
    started = time.time()
    resource_counts: Counter[str] = Counter()
    goal_counts: Counter[str] = Counter()
    template_counts: Counter[str] = Counter()
    unit_counts: list[int] = []
    element_count = 0
    selected = rows[:max_rows] if max_rows else rows

    for row in selected:
        sample_id = str(row.get("sample_id", "sample"))
        elements = extractor.extract_sequence(row.get("api_seq", []), sample_id=sample_id, max_len=max_len)
        units = extractor.build_units(row.get("api_seq", []), sample_id=sample_id, max_len=max_len)
        element_count += len(elements)
        unit_counts.append(len(units))
        resource_counts.update(e.resource for e in elements)
        goal_counts.update(e.goal for e in elements)
        template_counts.update(e.template_id for e in elements)

    elapsed = time.time() - started
    return {
        "extractor": "FrozenTemplateBehaviorExtractor",
        "schema": ["subject", "operation", "object", "resource", "context", "goal", "template_id"],
        "template_library_version": extractor.TEMPLATE_LIBRARY_VERSION,
        "llm_role": "semantic parser only; no label or malware verdict is produced",
        "privacy": "API names are normalized/desensitized; paths, IPs, and hashes are replaced by placeholders",
        "rows_profiled": len(selected),
        "elements_profiled": element_count,
        "avg_units_per_sample": float(np_mean(unit_counts)),
        "resource_distribution": dict(resource_counts.most_common()),
        "goal_distribution": dict(goal_counts.most_common()),
        "template_distribution": dict(template_counts.most_common()),
        "latency_seconds": round(elapsed, 6),
        "latency_ms_per_sample": round(1000.0 * elapsed / max(len(selected), 1), 6),
    }


def np_mean(values: list[int]) -> float:
    return float(sum(values) / max(len(values), 1))


def dump_units_jsonl_gz(rows: list[dict], extractor: FrozenTemplateBehaviorExtractor, out_path: str | Path, max_len: int = 5000) -> None:
    """Optional compact unit dump for interpretability/ablation.

    This is intentionally separate from the main runner so large intermediate
    behavior-unit files are not created by default.
    """
    import gzip

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt", encoding="utf-8") as f:
        for row in rows:
            sample_id = str(row.get("sample_id", "sample"))
            units = extractor.build_units(row.get("api_seq", []), sample_id=sample_id, max_len=max_len)
            f.write(json.dumps({
                "sample_id": sample_id,
                "source": row.get("source"),
                "label": row.get("label"),
                "family": row.get("family", "unknown"),
                "split": row.get("split"),
                "behavior_units": [asdict(unit) for unit in units],
            }, ensure_ascii=False, separators=(",", ":")) + "\n")
