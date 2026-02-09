import sys
import os

sys.path.insert(0, "src")
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "dummy")

import argparse
import json
import queue
import re
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any

import tot.models as models
from tot.models import request_extra_headers
from tot.tasks.game24 import Game24Task


_env_vllm_base_url = os.environ.get("VLLM_BASE_URL")
if _env_vllm_base_url:
    models.VLLM_BASE_URL = _env_vllm_base_url

_env_vllm_api_key = os.environ.get("VLLM_API_KEY")
if _env_vllm_api_key:
    models.VLLM_API_KEY = _env_vllm_api_key

_env_vllm_model = os.environ.get("VLLM_MODEL")
if _env_vllm_model:
    models.VLLM_MODEL = _env_vllm_model


def _task_is_solved(task: Game24Task, idx: int, y: str) -> bool:
    try:
        # Game24: reaching a single remaining number 24 is already a valid solution
        # even if the final "Answer:" line is not produced.
        left = _get_current_numbers(y).strip()
        if left:
            parts = left.replace(",", " ").split()
            if len(parts) == 1:
                try:
                    if abs(float(parts[0]) - 24.0) <= 1e-6:
                        return True
                except Exception:
                    pass
    except Exception:
        pass
    is_solved_fn = getattr(task, "is_solved", None)
    if callable(is_solved_fn):
        try:
            return bool(is_solved_fn(idx, y))
        except Exception:
            pass
    try:
        r = task.test_output(idx, y)
        if isinstance(r, dict) and "r" in r:
            return bool(r["r"])
    except Exception:
        pass
    try:
        last_line = (y or "").strip().split("\n")[-1].lower()
        return bool("answer" in last_line and "24" in last_line)
    except Exception:
        return False


def _get_current_numbers(y: str) -> str:
    last_line = (y or "").strip().split("\n")[-1]
    if "left: " not in last_line:
        return ""
    return last_line.split("left: ")[-1].split(")")[0]


def _parse_proposals(parent_y: str, response: str) -> list[str]:
    if not response:
        return []

    current_numbers = ""
    if parent_y and "left: " in parent_y:
        current_numbers = _get_current_numbers(parent_y)

    is_left_24 = False
    if current_numbers:
        parts = current_numbers.replace(",", " ").split()
        if len(parts) == 1:
            try:
                is_left_24 = abs(float(parts[0]) - 24.0) <= 1e-6
            except Exception:
                is_left_24 = False

    if is_left_24:
        answer_lines = [line for line in response.split("\n") if "answer" in line.lower() and "=" in line]
        if answer_lines:
            return [parent_y + answer_lines[0] + "\n"]
        return []

    lines = response.split("\n")
    cleaned: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        line_clean = re.sub(r"\*\*", "", line)
        line_clean = re.sub(r"#+", "", line_clean)
        line_clean = re.sub(r"→", ":", line_clean)
        line_clean = re.sub(r"^\d+\.\s*", "", line_clean)
        line_clean = re.sub(r"^-\s+", "", line_clean)
        line_clean = line_clean.strip()

        expr_match = re.search(r"(\-?\d+\.?\d*\s*[\+\-\*\/]\s*\-?\d+\.?\d*\s*=\s*\-?[\d\.]+)", line_clean)
        if expr_match:
            expr = expr_match.group(1).strip()
            left_nums = None

            left_match = re.search(r"(?:left|remaining|Left|Remaining):\s*(.+?)(?:\)|$)", line_clean, re.IGNORECASE)
            if left_match:
                left_nums = left_match.group(1).strip()
            elif "(" in line_clean and ")" in line_clean:
                paren_match = re.search(r"\((?:left|remaining):\s*([^\)]+)\)", line_clean, re.IGNORECASE)
                if paren_match:
                    left_nums = paren_match.group(1).strip()
            elif i + 1 < len(lines):
                next_line = lines[i + 1]
                next_clean = re.sub(r"\*\*", "", next_line).strip()
                left_match = re.search(r"(?:left|remaining|Left|Remaining):\s*(.+?)(?:\)|$)", next_clean, re.IGNORECASE)
                if left_match:
                    left_nums = left_match.group(1).strip()
                    i += 1

            if left_nums:
                left_nums = re.sub(r",\s*", " ", left_nums)
                left_nums = left_nums.replace("`", "")
                left_nums = re.sub(r"\s+", " ", left_nums)
                cleaned.append(f"{expr} (left: {left_nums})")
        i += 1

    return [parent_y + c + "\n" for c in cleaned]


def _dedup_proposals(proposals: list[str]) -> list[str]:
    seen = set()
    out = []
    for p in proposals:
        key = re.sub(r"\s+", " ", (p or "").strip())
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _get_value(task: Game24Task, x: str, y: str, n_evaluate_sample: int, model_name: str, task_id: int, step: int, parent_id: Any, branch_id: Any) -> float:
    prompt = task.value_prompt_wrap(x, y)
    extra_headers = {
        "X-TOT-Task-Id": str(task_id),
        "X-TOT-Depth": str(step),
        "X-TOT-Parent-Idx": str(parent_id),
        "X-TOT-Branch-Idx": str(branch_id),
        "X-TOT-Call-Type": "value",
    }
    with request_extra_headers(extra_headers):
        outs = models.gpt(
            prompt,
            n=max(1, int(n_evaluate_sample)),
            stop=getattr(task, "value_stop", None),
            max_tokens=getattr(task, "value_max_tokens", 1000),
            temperature=getattr(task, "value_temperature", 0.0),
            model=model_name,
        )
    try:
        return float(task.value_outputs_unwrap(x, y, outs))
    except Exception:
        return 0.0


def _run_step0(
    task: Game24Task,
    x: str,
    args,
    task_id: int,
    time_trace=None,
    t0: float | None = None,
    mode: str | None = None,
) -> tuple[list[str], float, dict[str, Any]]:
    print(f"[STEP0] Starting for task_id={task_id}")
    start_t = time.time()
    prompt = task.propose_prompt_wrap(x, "")
    extra_headers = {
        "X-TOT-Task-Id": str(task_id),
        "X-TOT-Depth": "0",
        "X-TOT-Parent-Idx": "root",
        "X-TOT-Branch-Idx": "root",
        "X-TOT-Call-Type": "propose",
    }
    _ts = time.time()
    with request_extra_headers(extra_headers):
        responses = models.gpt(
            prompt,
            n=1,
            stop=None,
            max_tokens=256,
            temperature=0.7,
            model=args.backend,
        )
    _te = time.time()
    if callable(time_trace) and t0 is not None and mode:
        try:
            time_trace(
                {
                    "type": "span",
                    "mode": str(mode),
                    "op": "propose",
                    "t_start": float(_ts - float(t0)),
                    "t_end": float(_te - float(t0)),
                    "task_id": int(task_id),
                    "depth": 0,
                    "parent_id": "root",
                    "branch_id": "root",
                    "node_id": None,
                    "thread": int(threading.get_ident()),
                }
            )
        except Exception:
            pass

    proposals: list[str] = []
    for r in responses or []:
        proposals.extend(_parse_proposals("", r))
    proposals = _dedup_proposals(proposals)

    step0_max_proposals = getattr(args, "step0_max_proposals", None)
    if step0_max_proposals == 0:
        step0_max_proposals = None
    if step0_max_proposals is None:
        step0_max_proposals = max(int(args.n_select_sample) * int(args.max_children_per_parent), int(args.max_children_per_parent))
    if step0_max_proposals and len(proposals) > int(step0_max_proposals):
        proposals = proposals[: int(step0_max_proposals)]

    propose_time = time.time() - start_t
    print(f"[STEP0] Propose done: {len(proposals)} proposals in {propose_time:.2f}s")

    values = [0.0] * len(proposals)
    if proposals:
        print(f"[STEP0] Evaluating {len(proposals)} proposals with concurrency={int(args.value_concurrency)}")
        def _do_step0_value(i: int, p: str) -> tuple[int, float]:
            _vts = time.time()
            v = _get_value(task, x, p, args.n_evaluate_sample, args.backend, task_id, 0, "root", i)
            _vte = time.time()
            if callable(time_trace) and t0 is not None and mode:
                try:
                    time_trace(
                        {
                            "type": "span",
                            "mode": str(mode),
                            "op": "value",
                            "t_start": float(_vts - float(t0)),
                            "t_end": float(_vte - float(t0)),
                            "task_id": int(task_id),
                            "depth": 0,
                            "parent_id": "root",
                            "branch_id": int(i),
                            "node_id": None,
                            "thread": int(threading.get_ident()),
                        }
                    )
                except Exception:
                    pass
            return i, float(v)

        with ThreadPoolExecutor(max_workers=int(args.value_concurrency)) as ex:
            futs = []
            for i, p in enumerate(proposals):
                futs.append(
                    (
                        i,
                        ex.submit(_do_step0_value, i, p),
                    )
                )
            for i, fut in futs:
                _i, _v = fut.result()
                values[int(_i)] = float(_v)

    ids = list(range(len(proposals)))
    k = int(getattr(args, "n_select_sample", 0) or 0)
    if k <= 0:
        select_ids = sorted(ids, key=lambda i: values[i], reverse=True)
    else:
        select_ids = sorted(ids, key=lambda i: values[i], reverse=True)[:k]
    selected = [proposals[i] for i in select_ids]
    step0_debug = {
        "proposals": proposals,
        "values": values,
        "selected_ids": select_ids,
        "selected": selected,
    }
    return selected, time.time() - start_t, step0_debug


def _run_serial(task: Game24Task, idx: int, args, log_dir: str) -> dict[str, Any]:
    task_id = idx + 1
    x = task.get_input(idx)

    models.completion_tokens = 0
    models.prompt_tokens = 0

    time_trace_fp = None
    time_trace_lock = threading.Lock()
    if bool(getattr(args, "write_time_trace", False)):
        time_trace_fp = open(os.path.join(log_dir, f"time_trace_task_{task_id}.jsonl"), "w", encoding="utf-8")

    t0 = time.time()

    def _time_trace(event: dict[str, Any]) -> None:
        if time_trace_fp is None:
            return

        try:
            with time_trace_lock:
                time_trace_fp.write(json.dumps(event, ensure_ascii=False) + "\n")
                time_trace_fp.flush()
        except Exception:
            return

    step0_candidates, step0_time, step0_debug = _run_step0(task, x, args, task_id, time_trace=_time_trace, t0=t0, mode="serial")

    start_t = time.time()
    ys = step0_candidates
    best = {"y": "", "value": None}
    solved_y = ""

    try:
        for step in range(1, int(task.steps)):
            new_ys: list[str] = []
            new_vs: list[float] = []

            for parent_idx, parent_y in enumerate(ys):
                prompt = task.propose_prompt_wrap(x, parent_y)
                extra_headers = {
                    "X-TOT-Task-Id": str(task_id),
                    "X-TOT-Depth": str(step),
                    "X-TOT-Parent-Idx": str(parent_idx),
                    "X-TOT-Branch-Idx": str(parent_idx),
                    "X-TOT-Call-Type": "propose",
                }
                _ts = time.time()
                with request_extra_headers(extra_headers):
                    responses = models.gpt(
                        prompt,
                        n=1,
                        stop=None,
                        max_tokens=256,
                        temperature=0.7,
                        model=args.backend,
                    )
                _te = time.time()
                _time_trace(
                    {
                        "type": "span",
                        "mode": "serial",
                        "op": "propose",
                        "t_start": float(_ts - t0),
                        "t_end": float(_te - t0),
                        "task_id": int(task_id),
                        "depth": int(step),
                        "parent_id": int(parent_idx),
                        "node_id": None,
                        "thread": int(threading.get_ident()),
                    }
                )

                proposals: list[str] = []
                for r in responses or []:
                    proposals.extend(_parse_proposals(parent_y, r))
                proposals = _dedup_proposals(proposals)
                if int(args.max_children_per_parent) and len(proposals) > int(args.max_children_per_parent):
                    proposals = proposals[: int(args.max_children_per_parent)]

                for proposal_idx, p in enumerate(proposals):
                    _vts = time.time()
                    v = _get_value(task, x, p, args.n_evaluate_sample, args.backend, task_id, step, parent_idx, proposal_idx)
                    _vte = time.time()
                    _time_trace(
                        {
                            "type": "span",
                            "mode": "serial",
                            "op": "value",
                            "t_start": float(_vts - t0),
                            "t_end": float(_vte - t0),
                            "task_id": int(task_id),
                            "depth": int(step),
                            "parent_id": int(parent_idx),
                            "branch_id": int(proposal_idx),
                            "node_id": None,
                            "thread": int(threading.get_ident()),
                        }
                    )
                    new_ys.append(p)
                    new_vs.append(float(v))
                    if not solved_y and _task_is_solved(task, idx, p):
                        solved_y = p

            if not new_ys:
                break

            ids = list(range(len(new_ys)))
            k = int(getattr(args, "n_select_sample", 0) or 0)
            if k <= 0:
                select_ids = sorted(ids, key=lambda i: new_vs[i], reverse=True)
            else:
                select_ids = sorted(ids, key=lambda i: new_vs[i], reverse=True)[:k]
            ys = [new_ys[i] for i in select_ids]

            top_i = select_ids[0] if select_ids else None
            if top_i is not None:
                best["y"] = new_ys[top_i]
                best["value"] = float(new_vs[top_i])

            if args.early_stop:
                for y in ys:
                    if _task_is_solved(task, idx, y):
                        best["y"] = y
                        solved_y = y
                        break
                if best.get("y") and _task_is_solved(task, idx, best["y"]):
                    break
    finally:
        if time_trace_fp is not None:
            try:
                time_trace_fp.close()
            except Exception:
                pass

    search_time = time.time() - start_t
    total_time = step0_time + search_time
    tokens = int(models.completion_tokens + models.prompt_tokens)
    out = solved_y or best.get("y") or (ys[0] if ys else "")
    success = bool(out) and _task_is_solved(task, idx, out)

    result = {
        "task_id": task_id,
        "idx": idx,
        "input": x,
        "step0_time": step0_time,
        "tot": {"step0": step0_debug},
        "baseline": {
            "mode": "serial",
            "output": out,
            "success": success,
            "tokens": tokens,
            "total_time": total_time,
            "search_time": search_time,
        },
    }

    with open(os.path.join(log_dir, f"task_{task_id}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


def _run_scheme_h(task: Game24Task, idx: int, args, log_dir: str) -> dict[str, Any]:
    task_id = idx + 1
    x = task.get_input(idx)

    models.completion_tokens = 0
    models.prompt_tokens = 0

    trace_fp = None
    trace_lock = threading.Lock()
    trace_path = None
    if bool(getattr(args, "write_trace", False)):
        trace_path = os.path.join(log_dir, f"trace_task_{task_id}.jsonl")
        trace_fp = open(trace_path, "w", encoding="utf-8")

    time_trace_fp = None
    time_trace_lock = threading.Lock()
    if bool(getattr(args, "write_time_trace", False)):
        time_trace_fp = open(os.path.join(log_dir, f"time_trace_task_{task_id}.jsonl"), "w", encoding="utf-8")

    t0 = time.time()

    def _trace(event: dict[str, Any]) -> None:
        if trace_fp is None:
            return
        try:
            with trace_lock:
                trace_fp.write(json.dumps(event, ensure_ascii=False) + "\n")
                trace_fp.flush()
        except Exception:
            return

    def _time_trace(event: dict[str, Any]) -> None:
        if time_trace_fp is None:
            return
        try:
            with time_trace_lock:
                time_trace_fp.write(json.dumps(event, ensure_ascii=False) + "\n")
                time_trace_fp.flush()
        except Exception:
            return

    step0_time = 0.0
    step0_debug = None

    tot_lock = threading.Lock()
    max_events_per_depth = int(getattr(args, "log_max_events_per_depth", 50))
    propose_done_by_depth: dict[int, int] = {}
    value_done_by_depth: dict[int, int] = {}
    propose_samples_by_depth: dict[int, list[dict[str, Any]]] = {}
    value_samples_by_depth: dict[int, list[dict[str, Any]]] = {}

    class Node:
        __slots__ = (
            "node_id",
            "depth",
            "y",
            "parent_id",
            "score",
            "expanded",
            "children",
            "value_enqueued",
            "propose_enqueued",
            "invalid",
        )

        def __init__(self, node_id: int, depth: int, y: str, parent_id: int | None):
            self.node_id = node_id
            self.depth = depth
            self.y = y
            self.parent_id = parent_id
            self.score: float | None = None
            self.expanded = False
            self.children: list[int] = []
            self.value_enqueued = False
            self.propose_enqueued = False
            self.invalid = False

    node_lock = threading.Lock()
    stop_event = threading.Event()

    next_node_id = 0

    def _alloc_node_id() -> int:
        nonlocal next_node_id
        with node_lock:
            nid = next_node_id
            next_node_id += 1
            return nid

    depth_state: dict[int, dict[str, Any]] = {}

    def _get_depth_state(depth: int) -> dict[str, Any]:
        st = depth_state.get(depth)
        if st is None:
            st = {
                "all_ids": [],
                "evaluated_ids": set(),
                "pruned": False,
                "closed": False,
                "valid_ids": set(),
                "expand_target": None,
                "propose_submitted": 0,
                "propose_done": 0,
            }
            depth_state[depth] = st
        return st

    nodes: dict[int, Node] = {}

    def _invalidate_subtree(node_id: int) -> None:
        stack = [int(node_id)]
        while stack:
            nid = stack.pop()
            with node_lock:
                n = nodes.get(nid)
                if n is None:
                    continue
                if n.invalid:
                    continue
                n.invalid = True
                stack.extend(list(n.children or []))

    def _enqueue_value(node_id: int) -> None:
        with node_lock:
            n = nodes.get(node_id)
            if n is None or n.invalid or n.value_enqueued:
                return
            if n.score is not None:
                return
            n.value_enqueued = True
        task_q.append(("value", int(node_id)))

    def _enqueue_propose(node_id: int) -> None:
        with node_lock:
            n = nodes.get(node_id)
            if n is None or n.invalid:
                return
            if n.propose_enqueued or n.expanded:
                return
            if n.score is None:
                return
            if int(n.depth) >= int(task.steps) - 1:
                return
            st = _get_depth_state(int(n.depth))
            if not bool(st.get("pruned")):
                return
            expand_target = st.get("expand_target")
            if expand_target is not None and int(st.get("propose_submitted", 0)) >= int(expand_target):
                return
            if bool(st.get("closed")):
                return
            n.propose_enqueued = True
        task_q.append(("propose", int(node_id)))

    def _maybe_prune(depth: int) -> None:
        if depth < 0:
            return
        if int(depth) >= int(task.steps):
            return
        with node_lock:
            st = _get_depth_state(int(depth))
            if bool(st.get("pruned")):
                return
            parent_st = _get_depth_state(int(depth) - 1)
            if not bool(parent_st.get("closed")):
                return
            ids = list(st.get("all_ids") or [])
            if not ids:
                st["pruned"] = True
                st["valid_ids"] = set()
                st["expand_target"] = 0
                st["closed"] = True
                return
            valid_ids = []
            for nid in ids:
                n = nodes.get(int(nid))
                if n is None:
                    continue
                if n.invalid:
                    continue
                if n.score is None:
                    return
                valid_ids.append(int(nid))

            k = int(getattr(args, "n_select_sample", 0) or 0)
            if k <= 0:
                keep_ids = set(valid_ids)
            else:
                keep_ids = {
                    nid
                    for nid in sorted(valid_ids, key=lambda _nid: float(nodes[_nid].score or 0.0), reverse=True)[:k]
                }
            prune_ids = [nid for nid in valid_ids if nid not in keep_ids]

        for nid in prune_ids:
            _invalidate_subtree(nid)

        with node_lock:
            st = _get_depth_state(int(depth))
            st["pruned"] = True
            st["valid_ids"] = set(keep_ids)
            expandable = []
            for nid in keep_ids:
                n = nodes.get(int(nid))
                if n is None or n.invalid:
                    continue
                if int(n.depth) < int(task.steps) - 1:
                    expandable.append(int(nid))
            st["expand_target"] = int(len(expandable))
            if int(st["expand_target"]) <= 0:
                st["closed"] = True

        for nid in sorted(expandable):
            _enqueue_propose(nid)

        if int(st.get("expand_target") or 0) <= 0:
            next_depth = int(depth) + 1
            if next_depth < int(task.steps):
                with node_lock:
                    st_next = depth_state.get(int(next_depth))
                    next_has_nodes = bool(st_next and (st_next.get("all_ids") or []))
                if next_has_nodes:
                    _maybe_prune(int(next_depth))

    def _do_propose(node_id: int) -> dict[str, Any]:
        with node_lock:
            node = nodes[node_id]
            parent_y = node.y
            parent_depth = node.depth
            child_depth = parent_depth + 1
            parent_idx_header = "root" if int(node_id) == int(root_id) else str(node_id)

        prompt = task.propose_prompt_wrap(x, parent_y)
        extra_headers = {
            "X-TOT-Task-Id": str(task_id),
            "X-TOT-Depth": str(child_depth),
            "X-TOT-Parent-Idx": parent_idx_header,
            "X-TOT-Branch-Idx": "",
            "X-TOT-Call-Type": "propose",
        }
        _ts = time.time()
        with request_extra_headers(extra_headers):
            responses = models.gpt(
                prompt,
                n=1,
                stop=None,
                max_tokens=256,
                temperature=0.7,
                model=args.backend,
            )
        _te = time.time()
        _time_trace(
            {
                "type": "span",
                "mode": "scheme_h",
                "op": "propose",
                "t_start": float(_ts - t0),
                "t_end": float(_te - t0),
                "task_id": int(task_id),
                "depth": int(child_depth),
                "parent_id": "root" if int(node_id) == int(root_id) else int(node_id),
                "node_id": int(node_id),
                "thread": int(threading.get_ident()),
            }
        )

        proposals: list[str] = []
        for r in responses or []:
            proposals.extend(_parse_proposals(parent_y, r))
        proposals = _dedup_proposals(proposals)
        if int(args.max_children_per_parent) and len(proposals) > int(args.max_children_per_parent):
            proposals = proposals[: int(args.max_children_per_parent)]

        with tot_lock:
            propose_done_by_depth[child_depth] = int(propose_done_by_depth.get(child_depth, 0)) + 1
            if max_events_per_depth and len(propose_samples_by_depth.get(child_depth, [])) < max_events_per_depth:
                propose_samples_by_depth.setdefault(child_depth, []).append(
                    {
                        "parent_node_id": node_id,
                        "parent_depth": parent_depth,
                        "n_proposals": len(proposals),
                        "proposals": proposals,
                    }
                )

        _trace(
            {
                "type": "propose_done",
                "task_id": task_id,
                "parent_node_id": node_id,
                "parent_depth": parent_depth,
                "child_depth": child_depth,
                "n_proposals": len(proposals),
                "proposals": proposals,
            }
        )
        return {"proposals": proposals, "depth": child_depth}

    def _do_value(node_id: int) -> float:
        with node_lock:
            node = nodes[node_id]
            y = node.y
            depth = node.depth
            parent_id = node.parent_id
        _ts = time.time()
        v = _get_value(task, x, y, args.n_evaluate_sample, args.backend, task_id, depth, parent_id, 0)
        _te = time.time()
        _time_trace(
            {
                "type": "span",
                "mode": "scheme_h",
                "op": "value",
                "t_start": float(_ts - t0),
                "t_end": float(_te - t0),
                "task_id": int(task_id),
                "depth": int(depth),
                "parent_id": int(parent_id) if parent_id is not None else None,
                "node_id": int(node_id),
                "thread": int(threading.get_ident()),
            }
        )
        return float(v)

    done_q: queue.Queue = queue.Queue()
    inflight_lock = threading.Lock()
    inflight = 0

    def _inflight_inc() -> None:
        nonlocal inflight
        with inflight_lock:
            inflight += 1

    def _inflight_dec() -> None:
        nonlocal inflight
        with inflight_lock:
            inflight -= 1

    def _submit(ex: ThreadPoolExecutor, fn, typ: str, node_id: int) -> None:
        _inflight_inc()
        fut = ex.submit(fn, node_id)

        def _cb(f):
            try:
                done_q.put((typ, node_id, f.result(), None))
            except Exception as e:
                done_q.put((typ, node_id, None, e))

        fut.add_done_callback(_cb)

    best_solution = {"y": "", "node_id": None, "depth": None, "value": None}
    first_solution = {"time_s": None, "tokens": None}

    propose_done_cnt = 0
    value_done_cnt = 0

    task_q: deque[tuple[str, int]] = deque()

    root_id = _alloc_node_id()
    root = Node(root_id, -1, "", None)
    root.score = 20.0
    nodes[root_id] = root
    st_root = _get_depth_state(-1)
    st_root["pruned"] = True
    st_root["valid_ids"] = {int(root_id)}
    st_root["expand_target"] = 1
    st_root["closed"] = False
    st_root["all_ids"].append(int(root_id))
    st_root["evaluated_ids"].add(int(root_id))
    _enqueue_propose(root_id)

    propose_exec = ThreadPoolExecutor(max_workers=int(args.propose_concurrency))
    value_exec = ThreadPoolExecutor(max_workers=int(args.value_concurrency))

    propose_inflight = 0
    value_inflight = 0
    type_inflight_lock = threading.Lock()

    start_t = time.time()
    last_stats = time.time()

    def _mark_depth_closed(depth: int) -> None:
        with node_lock:
            st = _get_depth_state(int(depth))
            st["closed"] = True

    try:
        while True:
            with inflight_lock:
                inflight_now = inflight

            if stop_event.is_set():
                if inflight_now <= 0:
                    break

            submitted_any = False

            if not stop_event.is_set():
                batch = []
                while task_q and len(batch) < int(args.batch_size):
                    batch.append(task_q.popleft())

                for typ, nid in batch:
                    if stop_event.is_set():
                        break

                    if typ == "propose":
                        with type_inflight_lock:
                            if propose_inflight >= max(1, int(args.propose_concurrency)):
                                task_q.append((typ, nid))
                                continue
                        with node_lock:
                            node = nodes.get(int(nid))
                            if node is None or node.invalid:
                                continue
                            st = _get_depth_state(int(node.depth))
                            expand_target = st.get("expand_target")
                            if not bool(st.get("pruned")):
                                node.propose_enqueued = False
                                continue
                            if bool(st.get("closed")):
                                node.propose_enqueued = False
                                continue
                            if expand_target is not None and int(st.get("propose_submitted", 0)) >= int(expand_target):
                                node.propose_enqueued = False
                                continue
                            if node.expanded or node.score is None or int(node.depth) >= int(task.steps) - 1:
                                node.propose_enqueued = False
                                continue
                            node.propose_enqueued = False
                            node.expanded = True
                            st["propose_submitted"] = int(st.get("propose_submitted", 0)) + 1

                        with type_inflight_lock:
                            propose_inflight += 1
                        _submit(propose_exec, _do_propose, "propose", int(nid))
                        submitted_any = True

                    else:
                        with type_inflight_lock:
                            if value_inflight >= max(1, int(args.value_concurrency)):
                                task_q.append((typ, nid))
                                continue
                        with node_lock:
                            node = nodes.get(int(nid))
                            if node is None or node.invalid:
                                continue
                            node.value_enqueued = False
                            if node.score is not None:
                                continue
                        with type_inflight_lock:
                            value_inflight += 1
                        _submit(value_exec, _do_value, "value", int(nid))
                        submitted_any = True

            processed_any = False
            while True:
                try:
                    typ, nid, res, err = done_q.get_nowait()
                except queue.Empty:
                    break
                processed_any = True
                _inflight_dec()

                if typ == "propose":
                    with type_inflight_lock:
                        propose_inflight = max(0, propose_inflight - 1)
                elif typ == "value":
                    with type_inflight_lock:
                        value_inflight = max(0, value_inflight - 1)

                if err is not None:
                    continue

                if typ == "value":
                    v = float(res)
                    value_done_cnt += 1
                    with node_lock:
                        node = nodes.get(int(nid))
                        if node is None:
                            continue
                        if node.invalid:
                            continue
                        node.score = v
                        depth = int(node.depth)
                        y_local = node.y
                        _get_depth_state(depth)["evaluated_ids"].add(int(nid))

                    with tot_lock:
                        value_done_by_depth[depth] = int(value_done_by_depth.get(depth, 0)) + 1
                        if max_events_per_depth and len(value_samples_by_depth.get(depth, [])) < max_events_per_depth:
                            value_samples_by_depth.setdefault(depth, []).append(
                                {
                                    "node_id": nid,
                                    "parent_id": nodes.get(nid).parent_id if nid in nodes else None,
                                    "value": v,
                                    "y": y_local,
                                }
                            )

                    _trace(
                        {
                            "type": "value_done",
                            "task_id": task_id,
                            "node_id": nid,
                            "depth": depth,
                            "parent_id": nodes.get(nid).parent_id if nid in nodes else None,
                            "value": v,
                            "y": y_local,
                        }
                    )

                    if _task_is_solved(task, idx, y_local):
                        if not best_solution.get("y"):
                            best_solution.update({"y": y_local, "node_id": nid, "depth": depth, "value": v})
                        if first_solution.get("time_s") is None:
                            first_solution["time_s"] = float(time.time() - start_t)
                            first_solution["tokens"] = int(models.completion_tokens + models.prompt_tokens)
                        if args.early_stop:
                            stop_event.set()

                    _maybe_prune(depth)

                else:
                    propose_done_cnt += 1
                    with node_lock:
                        parent = nodes.get(int(nid))
                        if parent is None:
                            continue
                        parent_depth = int(parent.depth)
                        st = _get_depth_state(parent_depth)
                        st["propose_done"] = int(st.get("propose_done", 0)) + 1
                        propose_done_at_depth = int(st.get("propose_done", 0))
                        expand_target = st.get("expand_target")

                    if not stop_event.is_set():
                        proposals = list((res or {}).get("proposals", []))
                        with node_lock:
                            parent = nodes.get(int(nid))
                            if parent is not None and (not parent.invalid):
                                child_depth = int(parent.depth) + 1
                                st_child = _get_depth_state(child_depth)
                            else:
                                child_depth = parent_depth + 1
                                st_child = _get_depth_state(child_depth)

                        for child_y in proposals:
                            child_id = _alloc_node_id()
                            child = Node(child_id, child_depth, child_y, int(nid))
                            with node_lock:
                                nodes[child_id] = child
                                st_child["all_ids"].append(int(child_id))
                                parent = nodes.get(int(nid))
                                if parent is not None:
                                    parent.children.append(int(child_id))
                            _enqueue_value(child_id)

                    if expand_target is not None and propose_done_at_depth >= int(expand_target):
                        _mark_depth_closed(parent_depth)
                        _maybe_prune(parent_depth + 1)

            now = time.time()
            if float(args.log_interval_s) and now - last_stats >= float(args.log_interval_s):
                with inflight_lock:
                    inflight_now = inflight
                print(
                    f"[STATS] inflight={inflight_now} qsize={len(task_q)} first_solution_time={first_solution.get('time_s')} best_depth={best_solution.get('depth')}"
                )
                last_stats = now

            if stop_event.is_set():
                if not processed_any:
                    time.sleep(0.01)
                continue

            if not submitted_any and not processed_any:
                with inflight_lock:
                    inflight_now = inflight
                if inflight_now == 0 and (not task_q):
                    break
                time.sleep(0.01)

    finally:
        propose_exec.shutdown(wait=False)
        value_exec.shutdown(wait=False)
        if trace_fp is not None:
            try:
                trace_fp.close()
            except Exception:
                pass
        if time_trace_fp is not None:
            try:
                time_trace_fp.close()
            except Exception:
                pass

    search_time = time.time() - start_t
    total_time = step0_time + search_time
    tokens = int(models.completion_tokens + models.prompt_tokens)
    out = best_solution.get("y") or ""
    success = bool(out) and _task_is_solved(task, idx, out)

    depth_nodes: dict[int, int] = {}
    with node_lock:
        for n in nodes.values():
            if n.invalid:
                continue
            depth_nodes[n.depth] = int(depth_nodes.get(n.depth, 0)) + 1

    result = {
        "task_id": task_id,
        "idx": idx,
        "input": x,
        "step0_time": step0_time,
        "tot": {
            "step0": step0_debug,
            "scheme_h": {
                "trace_path": trace_path,
                "n_nodes": len(nodes),
                "propose_done": propose_done_cnt,
                "value_done": value_done_cnt,
                "depth_nodes": depth_nodes,
                "propose_done_by_depth": propose_done_by_depth,
                "value_done_by_depth": value_done_by_depth,
                "propose_samples_by_depth": propose_samples_by_depth,
                "value_samples_by_depth": value_samples_by_depth,
                "first_solution": first_solution,
            },
        },
        "baseline": {
            "mode": "scheme_h",
            "output": out,
            "success": success,
            "tokens": tokens,
            "total_time": total_time,
            "search_time": search_time,
            "best_solution": best_solution,
            "first_solution": first_solution,
        },
    }

    with open(os.path.join(log_dir, f"task_{task_id}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


def _run_dfs(task: Game24Task, idx: int, args, log_dir: str) -> dict[str, Any]:
    task_id = idx + 1
    x = task.get_input(idx)

    models.completion_tokens = 0
    models.prompt_tokens = 0

    step0_candidates, step0_time, step0_debug = _run_step0(task, x, args, task_id)

    start_t = time.time()
    dfs_start_t = start_t
    time_limit_s = float(getattr(args, "dfs_time_limit_s", 1200) or 1200)
    value_threshold = float(getattr(args, "dfs_value_threshold", 1.0) or 1.0)

    dfs_stats = {
        "value_threshold": value_threshold,
        "time_limit_s": time_limit_s,
        "nodes_visited": 0,
        "nodes_expanded": 0,
        "children_evaluated": 0,
        "pruned_by_threshold": 0,
        "max_depth_reached": 0,
        "timeout": False,
        "path": [],
        "best_solution": {"y": "", "value": None, "depth": None},
    }

    best_solution = {"y": "", "value": None, "depth": None}

    def _timed_out() -> bool:
        if time_limit_s <= 0:
            return False
        return (time.time() - dfs_start_t) >= time_limit_s

    def _dfs(parent_y: str, depth: int) -> tuple[bool, str]:
        nonlocal best_solution
        dfs_stats["nodes_visited"] = int(dfs_stats.get("nodes_visited", 0)) + 1
        dfs_stats["max_depth_reached"] = max(int(dfs_stats.get("max_depth_reached", 0)), int(depth))

        if _timed_out():
            dfs_stats["timeout"] = True
            return False, ""

        if parent_y and _task_is_solved(task, idx, parent_y):
            return True, parent_y

        if depth >= int(task.steps) - 1:
            return False, ""

        step = depth + 1
        dfs_stats["nodes_expanded"] = int(dfs_stats.get("nodes_expanded", 0)) + 1

        prompt = task.propose_prompt_wrap(x, parent_y)
        extra_headers = {
            "X-TOT-Task-Id": str(task_id),
            "X-TOT-Depth": str(step),
            "X-TOT-Parent-Idx": str(depth),
            "X-TOT-Branch-Idx": "0",
            "X-TOT-Call-Type": "propose",
        }
        with request_extra_headers(extra_headers):
            responses = models.gpt(
                prompt,
                n=1,
                stop=None,
                max_tokens=256,
                temperature=0.7,
                model=args.backend,
            )

        proposals: list[str] = []
        for r in responses or []:
            proposals.extend(_parse_proposals(parent_y, r))
        proposals = _dedup_proposals(proposals)
        if int(args.max_children_per_parent) and len(proposals) > int(args.max_children_per_parent):
            proposals = proposals[: int(args.max_children_per_parent)]

        if not proposals:
            return False, ""

        vals: list[float] = []
        dfs_stats["children_evaluated"] = int(dfs_stats.get("children_evaluated", 0)) + int(len(proposals))
        for proposal_idx, child_y in enumerate(proposals):
            if _timed_out():
                dfs_stats["timeout"] = True
                return False, ""
            v = _get_value(task, x, child_y, args.n_evaluate_sample, args.backend, task_id, step, depth, proposal_idx)
            vals.append(float(v))

        order = sorted(range(len(proposals)), key=lambda i: vals[i], reverse=True)
        for i in order:
            if _timed_out():
                dfs_stats["timeout"] = True
                return False, ""

            child_y = proposals[i]
            v = float(vals[i])

            if best_solution.get("value") is None or v > float(best_solution.get("value") or 0.0):
                best_solution = {"y": child_y, "value": v, "depth": step}
                dfs_stats["best_solution"] = best_solution

            if v < value_threshold:
                dfs_stats["pruned_by_threshold"] = int(dfs_stats.get("pruned_by_threshold", 0)) + 1
                continue

            dfs_stats["path"].append({"depth": step, "value": v, "y": child_y})
            ok, sol = _dfs(child_y, depth + 1)
            if ok:
                return True, sol
            dfs_stats["path"].pop()

        return False, ""

    solved_y = ""
    for y0 in step0_candidates:
        if _timed_out():
            dfs_stats["timeout"] = True
            break
        dfs_stats["path"] = [{"depth": 0, "value": None, "y": y0}]
        ok, sol = _dfs(y0, 0)
        if ok:
            solved_y = sol
            break

    search_time = time.time() - start_t
    total_time = step0_time + search_time
    tokens = int(models.completion_tokens + models.prompt_tokens)
    out = solved_y or best_solution.get("y") or (step0_candidates[0] if step0_candidates else "")
    success = bool(out) and _task_is_solved(task, idx, out)

    result = {
        "task_id": task_id,
        "idx": idx,
        "input": x,
        "step0_time": step0_time,
        "tot": {"step0": step0_debug, "dfs": dfs_stats},
        "baseline": {
            "mode": "dfs",
            "output": out,
            "success": success,
            "tokens": tokens,
            "total_time": total_time,
            "search_time": search_time,
        },
    }

    with open(os.path.join(log_dir, f"task_{task_id}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


def _run_layered(task: Game24Task, idx: int, args, log_dir: str) -> dict[str, Any]:
    task_id = idx + 1
    x = task.get_input(idx)

    models.completion_tokens = 0
    models.prompt_tokens = 0

    time_trace_fp = None
    time_trace_lock = threading.Lock()
    if bool(getattr(args, "write_time_trace", False)):
        time_trace_fp = open(os.path.join(log_dir, f"time_trace_task_{task_id}.jsonl"), "w", encoding="utf-8")

    t0 = time.time()

    def _time_trace(event: dict[str, Any]) -> None:
        if time_trace_fp is None:
            return
        try:
            with time_trace_lock:
                time_trace_fp.write(json.dumps(event, ensure_ascii=False) + "\n")
                time_trace_fp.flush()
        except Exception:
            return

    step0_candidates, step0_time, step0_debug = _run_step0(task, x, args, task_id, time_trace=_time_trace, t0=t0, mode="layered")

    start_t = time.time()
    ys = step0_candidates
    best = {"y": "", "value": None}
    solved_y = ""

    def _do_propose(parent_idx: int, parent_y: str, step: int) -> tuple[int, str, list[str]]:
        prompt = task.propose_prompt_wrap(x, parent_y)
        extra_headers = {
            "X-TOT-Task-Id": str(task_id),
            "X-TOT-Depth": str(step),
            "X-TOT-Parent-Idx": str(parent_idx),
            "X-TOT-Branch-Idx": str(parent_idx),
            "X-TOT-Call-Type": "propose",
        }
        _ts = time.time()
        with request_extra_headers(extra_headers):
            responses = models.gpt(
                prompt,
                n=1,
                stop=None,
                max_tokens=256,
                temperature=0.7,
                model=args.backend,
            )
        _te = time.time()
        _time_trace(
            {
                "type": "span",
                "mode": "layered",
                "op": "propose",
                "t_start": float(_ts - t0),
                "t_end": float(_te - t0),
                "task_id": int(task_id),
                "depth": int(step),
                "parent_id": int(parent_idx),
                "node_id": None,
                "thread": int(threading.get_ident()),
            }
        )

        proposals: list[str] = []
        for r in responses or []:
            proposals.extend(_parse_proposals(parent_y, r))
        proposals = _dedup_proposals(proposals)
        if int(args.max_children_per_parent) and len(proposals) > int(args.max_children_per_parent):
            proposals = proposals[: int(args.max_children_per_parent)]
        return parent_idx, parent_y, proposals

    def _timed_get_value(parent_idx: int, proposal_idx: int, p: str, step: int) -> float:
        _ts = time.time()
        v = _get_value(task, x, p, args.n_evaluate_sample, args.backend, task_id, step, parent_idx, proposal_idx)
        _te = time.time()
        _time_trace(
            {
                "type": "span",
                "mode": "layered",
                "op": "value",
                "t_start": float(_ts - t0),
                "t_end": float(_te - t0),
                "task_id": int(task_id),
                "depth": int(step),
                "parent_id": int(parent_idx),
                "branch_id": int(proposal_idx),
                "node_id": None,
                "thread": int(threading.get_ident()),
            }
        )
        return float(v)

    try:
        for step in range(1, int(task.steps)):
            parent_results = []
            if ys and int(args.propose_concurrency) > 1 and len(ys) > 1:
                with ThreadPoolExecutor(max_workers=int(args.propose_concurrency)) as ex:
                    futs = [ex.submit(_do_propose, i, y, step) for i, y in enumerate(ys)]
                    for fut in futs:
                        parent_results.append(fut.result())
            else:
                for i, y in enumerate(ys):
                    parent_results.append(_do_propose(i, y, step))

            all_candidates: list[tuple[int, int, str]] = []
            for parent_idx, _parent_y, proposals in parent_results:
                for proposal_idx, p in enumerate(proposals):
                    all_candidates.append((parent_idx, proposal_idx, p))

            if not all_candidates:
                break

            new_ys: list[str] = []
            new_vs: list[float] = []
            if int(args.value_concurrency) > 1 and len(all_candidates) > 1:
                with ThreadPoolExecutor(max_workers=int(args.value_concurrency)) as ex:
                    futs = []
                    for parent_idx, proposal_idx, p in all_candidates:
                        futs.append((p, ex.submit(_timed_get_value, parent_idx, proposal_idx, p, step)))
                    for p, fut in futs:
                        new_ys.append(p)
                        new_vs.append(float(fut.result()))
                        if not solved_y and _task_is_solved(task, idx, p):
                            solved_y = p
            else:
                for parent_idx, proposal_idx, p in all_candidates:
                    v = _timed_get_value(parent_idx, proposal_idx, p, step)
                    new_ys.append(p)
                    new_vs.append(float(v))
                    if not solved_y and _task_is_solved(task, idx, p):
                        solved_y = p

            ids = list(range(len(new_ys)))
            select_ids = sorted(ids, key=lambda i: new_vs[i], reverse=True)[: int(args.n_select_sample)]
            ys = [new_ys[i] for i in select_ids]

            top_i = select_ids[0] if select_ids else None
            if top_i is not None:
                best["y"] = new_ys[top_i]
                best["value"] = float(new_vs[top_i])

            if args.early_stop:
                for y in ys:
                    if _task_is_solved(task, idx, y):
                        best["y"] = y
                        solved_y = y
                        break
                if best.get("y") and _task_is_solved(task, idx, best["y"]):
                    break

    finally:
        if time_trace_fp is not None:
            try:
                time_trace_fp.close()
            except Exception:
                pass

    search_time = time.time() - start_t
    total_time = step0_time + search_time
    tokens = int(models.completion_tokens + models.prompt_tokens)
    out = solved_y or best.get("y") or (ys[0] if ys else "")
    success = bool(out) and _task_is_solved(task, idx, out)

    result = {
        "task_id": task_id,
        "idx": idx,
        "input": x,
        "step0_time": step0_time,
        "tot": {"step0": step0_debug},
        "baseline": {
            "mode": "layered",
            "output": out,
            "success": success,
            "tokens": tokens,
            "total_time": total_time,
            "search_time": search_time,
        },
    }

    with open(os.path.join(log_dir, f"task_{task_id}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


def _run_scheme_b(task: Game24Task, idx: int, args, log_dir: str) -> dict[str, Any]:
    task_id = idx + 1
    x = task.get_input(idx)

    models.completion_tokens = 0
    models.prompt_tokens = 0

    trace_fp = None
    trace_lock = threading.Lock()
    trace_path = None
    if bool(getattr(args, "write_trace", False)):
        trace_path = os.path.join(log_dir, f"trace_task_{task_id}.jsonl")
        trace_fp = open(trace_path, "w", encoding="utf-8")

    time_trace_fp = None
    time_trace_lock = threading.Lock()
    if bool(getattr(args, "write_time_trace", False)):
        time_trace_fp = open(os.path.join(log_dir, f"time_trace_task_{task_id}.jsonl"), "w", encoding="utf-8")

    t0 = time.time()

    def _trace(event: dict[str, Any]) -> None:
        if trace_fp is None:
            return
        try:
            with trace_lock:
                trace_fp.write(json.dumps(event, ensure_ascii=False) + "\n")
                trace_fp.flush()
        except Exception:
            return

    def _time_trace(event: dict[str, Any]) -> None:
        if time_trace_fp is None:
            return
        try:
            with time_trace_lock:
                time_trace_fp.write(json.dumps(event, ensure_ascii=False) + "\n")
                time_trace_fp.flush()
        except Exception:
            return

    step0_time = 0.0
    step0_debug = None

    tot_lock = threading.Lock()
    max_events_per_depth = int(getattr(args, "log_max_events_per_depth", 50))
    propose_done_by_depth: dict[int, int] = {}
    value_done_by_depth: dict[int, int] = {}
    propose_samples_by_depth: dict[int, list[dict[str, Any]]] = {}
    value_samples_by_depth: dict[int, list[dict[str, Any]]] = {}

    class Node:
        __slots__ = ("node_id", "depth", "y", "parent_id", "score", "expanded", "children", "value_enqueued", "propose_enqueued")

        def __init__(self, node_id: int, depth: int, y: str, parent_id: int | None):
            self.node_id = node_id
            self.depth = depth
            self.y = y
            self.parent_id = parent_id
            self.score: float | None = None
            self.expanded = False
            self.children: list[int] = []
            self.value_enqueued = False
            self.propose_enqueued = False

    node_lock = threading.Lock()
    stop_event = threading.Event()
    next_node_id = 0

    def _alloc_node_id() -> int:
        nonlocal next_node_id
        with node_lock:
            nid = next_node_id
            next_node_id += 1
            return nid

    depth_state: dict[int, dict[str, Any]] = {}

    def _get_depth_state(depth: int) -> dict[str, Any]:
        st = depth_state.get(depth)
        if st is None:
            st = {"evaluated_ids": set(), "expanded_count": 0}
            depth_state[depth] = st
        return st

    def _topk_ids(depth: int) -> set[int]:
        with node_lock:
            st = _get_depth_state(depth)
            eval_ids = list(st["evaluated_ids"])
            eval_nodes = [nodes[nid] for nid in eval_ids if nid in nodes and nodes[nid].score is not None]
        eval_nodes.sort(key=lambda n: float(n.score or 0.0), reverse=True)
        return {n.node_id for n in eval_nodes[: int(args.n_select_sample)]}

    def can_propose(node_id: int) -> bool:
        with node_lock:
            node = nodes.get(node_id)
            if node is None:
                return False
            if node.score is None:
                return False
            if node.expanded or node.depth >= int(task.steps) - 1:
                return False
            st = _get_depth_state(node.depth)
            if st["expanded_count"] >= int(args.n_select_sample):
                return False

            evaluated_cnt = len(st["evaluated_ids"])
            min_expand_value = float(getattr(args, "min_expand_value", 20.0))
            score = float(node.score)
            total_nodes_at_depth = 0
            for _n in nodes.values():
                if _n.depth == node.depth:
                    total_nodes_at_depth += 1

        topk = _topk_ids(nodes[node_id].depth)
        if node_id not in topk:
            return False

        # Early exploration: before all nodes at this depth are evaluated,
        # only allow proposing (expanding) full-score nodes.
        if evaluated_cnt < total_nodes_at_depth and score < min_expand_value:
            return False

        return True

    task_q: queue.PriorityQueue = queue.PriorityQueue()
    seq_lock = threading.Lock()
    seq = 0

    def _next_seq() -> int:
        nonlocal seq
        with seq_lock:
            s = seq
            seq += 1
            return s

    done_q: queue.Queue = queue.Queue()
    inflight_lock = threading.Lock()
    inflight = 0

    def _inflight_inc() -> None:
        nonlocal inflight
        with inflight_lock:
            inflight += 1

    def _inflight_dec() -> None:
        nonlocal inflight
        with inflight_lock:
            inflight -= 1

    nodes: dict[int, Node] = {}

    def _enqueue_value(node_id: int) -> None:
        with node_lock:
            node = nodes.get(node_id)
            if node is None or node.value_enqueued:
                return
            node.value_enqueued = True
        task_q.put((0.0, _next_seq(), "value", node_id))

    def _enqueue_propose(node_id: int, priority: float) -> None:
        with node_lock:
            node = nodes.get(node_id)
            if node is None or node.expanded or node.propose_enqueued:
                return
            node.propose_enqueued = True
        task_q.put((float(priority), _next_seq(), "propose", node_id))

    def _refresh_depth(depth: int) -> None:
        st = _get_depth_state(depth)
        with node_lock:
            eval_ids = list(st["evaluated_ids"])
        for nid in eval_ids:
            with node_lock:
                node = nodes.get(nid)
                score = None if node is None else node.score
            if score is None:
                continue
            if can_propose(nid):
                _enqueue_propose(nid, priority=-float(score))

    def _do_propose(node_id: int) -> dict[str, Any]:
        with node_lock:
            node = nodes[node_id]
            parent_y = node.y
            parent_depth = node.depth
            child_depth = parent_depth + 1

        prompt = task.propose_prompt_wrap(x, parent_y)
        parent_idx_header = "root" if int(node_id) == int(root_id) else str(node_id)
        extra_headers = {
            "X-TOT-Task-Id": str(task_id),
            "X-TOT-Depth": str(child_depth),
            "X-TOT-Parent-Idx": parent_idx_header,
            "X-TOT-Branch-Idx": "",
            "X-TOT-Call-Type": "propose",
        }
        _ts = time.time()
        with request_extra_headers(extra_headers):
            responses = models.gpt(
                prompt,
                n=1,
                stop=None,
                max_tokens=256,
                temperature=0.7,
                model=args.backend,
            )
        _te = time.time()
        _time_trace(
            {
                "type": "span",
                "mode": "scheme_b",
                "op": "propose",
                "t_start": float(_ts - t0),
                "t_end": float(_te - t0),
                "task_id": int(task_id),
                "depth": int(child_depth),
                "parent_id": "root" if int(node_id) == int(root_id) else int(node_id),
                "node_id": int(node_id),
                "thread": int(threading.get_ident()),
            }
        )

        proposals: list[str] = []
        for r in responses or []:
            proposals.extend(_parse_proposals(parent_y, r))
        proposals = _dedup_proposals(proposals)
        if int(args.max_children_per_parent) and len(proposals) > int(args.max_children_per_parent):
            proposals = proposals[: int(args.max_children_per_parent)]

        with tot_lock:
            propose_done_by_depth[child_depth] = int(propose_done_by_depth.get(child_depth, 0)) + 1
            if max_events_per_depth and len(propose_samples_by_depth.get(child_depth, [])) < max_events_per_depth:
                propose_samples_by_depth.setdefault(child_depth, []).append(
                    {
                        "parent_node_id": node_id,
                        "parent_depth": parent_depth,
                        "n_proposals": len(proposals),
                        "proposals": proposals,
                    }
                )
        _trace(
            {
                "type": "propose_done",
                "task_id": task_id,
                "parent_node_id": node_id,
                "parent_depth": parent_depth,
                "child_depth": child_depth,
                "n_proposals": len(proposals),
                "proposals": proposals,
            }
        )
        return {"proposals": proposals, "depth": child_depth}

    def _do_value(node_id: int) -> float:
        with node_lock:
            node = nodes[node_id]
            y = node.y
            depth = node.depth
            parent_id = node.parent_id
        _ts = time.time()
        v = _get_value(task, x, y, args.n_evaluate_sample, args.backend, task_id, depth, parent_id, 0)
        _te = time.time()
        _time_trace(
            {
                "type": "span",
                "mode": "scheme_b",
                "op": "value",
                "t_start": float(_ts - t0),
                "t_end": float(_te - t0),
                "task_id": int(task_id),
                "depth": int(depth),
                "parent_id": int(parent_id) if parent_id is not None else None,
                "node_id": int(node_id),
                "thread": int(threading.get_ident()),
            }
        )
        return float(v)

    def _submit(ex: ThreadPoolExecutor, fn, typ: str, node_id: int) -> None:
        _inflight_inc()
        fut = ex.submit(fn, node_id)

        def _cb(f):
            try:
                done_q.put((typ, node_id, f.result(), None))
            except Exception as e:
                done_q.put((typ, node_id, None, e))

        fut.add_done_callback(_cb)

    best_solution = {"y": "", "node_id": None, "depth": None, "value": None}

    propose_done_cnt = 0
    value_done_cnt = 0

    root_id = _alloc_node_id()
    root = Node(root_id, -1, "", None)
    root.score = 20.0
    nodes[root_id] = root
    _enqueue_propose(root_id, priority=0.0)

    propose_exec = ThreadPoolExecutor(max_workers=int(args.propose_concurrency))
    value_exec = ThreadPoolExecutor(max_workers=int(args.value_concurrency))

    propose_inflight = 0
    value_inflight = 0
    type_inflight_lock = threading.Lock()

    start_t = time.time()
    last_stats = time.time()

    try:
        while True:
            if stop_event.is_set():
                break

            batch: list[tuple[float, int, str, int]] = []
            while len(batch) < int(args.batch_size) and not task_q.empty():
                batch.append(task_q.get())

            for pri, _s, typ, nid in batch:
                if typ == "propose":
                    with type_inflight_lock:
                        if propose_inflight >= max(1, int(args.propose_concurrency)):
                            task_q.put((pri, _s, typ, nid))
                            continue
                    with node_lock:
                        node = nodes.get(nid)
                        if node is None or node.expanded:
                            continue
                        node.expanded = True
                        st = _get_depth_state(node.depth)
                        st["expanded_count"] += 1
                    with type_inflight_lock:
                        propose_inflight += 1
                    _submit(propose_exec, _do_propose, "propose", nid)
                else:
                    with type_inflight_lock:
                        if value_inflight >= max(1, int(args.value_concurrency)):
                            task_q.put((pri, _s, typ, nid))
                            continue
                    with type_inflight_lock:
                        value_inflight += 1
                    _submit(value_exec, _do_value, "value", nid)

            processed_any = False
            while True:
                try:
                    typ, nid, res, err = done_q.get_nowait()
                except queue.Empty:
                    break
                processed_any = True
                _inflight_dec()

                if typ == "propose":
                    with type_inflight_lock:
                        propose_inflight = max(0, propose_inflight - 1)
                elif typ == "value":
                    with type_inflight_lock:
                        value_inflight = max(0, value_inflight - 1)

                if err is not None:
                    continue

                if typ == "value":
                    v = float(res)
                    value_done_cnt += 1
                    with node_lock:
                        node = nodes.get(nid)
                        if node is None:
                            continue
                        node.score = v
                        depth = node.depth
                        y_local = node.y
                        _get_depth_state(depth)["evaluated_ids"].add(nid)

                    with tot_lock:
                        value_done_by_depth[depth] = int(value_done_by_depth.get(depth, 0)) + 1
                        if max_events_per_depth and len(value_samples_by_depth.get(depth, [])) < max_events_per_depth:
                            value_samples_by_depth.setdefault(depth, []).append(
                                {
                                    "node_id": nid,
                                    "parent_id": nodes.get(nid).parent_id if nid in nodes else None,
                                    "value": v,
                                    "y": y_local,
                                }
                            )

                    _trace(
                        {
                            "type": "value_done",
                            "task_id": task_id,
                            "node_id": nid,
                            "depth": depth,
                            "parent_id": nodes.get(nid).parent_id if nid in nodes else None,
                            "value": v,
                            "y": y_local,
                        }
                    )

                    if _task_is_solved(task, idx, y_local):
                        best_solution.update({"y": y_local, "node_id": nid, "depth": depth, "value": v})
                        if args.early_stop:
                            stop_event.set()

                    if can_propose(nid):
                        _enqueue_propose(nid, priority=-v)
                    _refresh_depth(depth)

                else:
                    proposals = list((res or {}).get("proposals", []))
                    propose_done_cnt += 1
                    with node_lock:
                        parent = nodes.get(nid)
                        if parent is None:
                            continue
                        parent_depth = parent.depth
                        child_depth = parent_depth + 1

                    for child_y in proposals:
                        child_id = _alloc_node_id()
                        child = Node(child_id, child_depth, child_y, nid)
                        with node_lock:
                            nodes[child_id] = child
                            parent = nodes.get(nid)
                            if parent is not None:
                                parent.children.append(child_id)
                        _enqueue_value(child_id)

            now = time.time()
            if float(args.log_interval_s) and now - last_stats >= float(args.log_interval_s):
                with inflight_lock:
                    inflight_now = inflight
                try:
                    qsize = task_q.qsize()
                except NotImplementedError:
                    qsize = -1
                print(f"[STATS] inflight={inflight_now} qsize={qsize} best_depth={best_solution.get('depth')} best_value={best_solution.get('value')}")
                last_stats = now

            if not batch and not processed_any:
                with inflight_lock:
                    inflight_now = inflight
                if inflight_now == 0 and task_q.empty():
                    break
                time.sleep(0.05)

    finally:
        propose_exec.shutdown(wait=False)
        value_exec.shutdown(wait=False)
        if trace_fp is not None:
            try:
                trace_fp.close()
            except Exception:
                pass
        if time_trace_fp is not None:
            try:
                time_trace_fp.close()
            except Exception:
                pass

    search_time = time.time() - start_t
    total_time = step0_time + search_time
    tokens = int(models.completion_tokens + models.prompt_tokens)
    out = best_solution.get("y") or ""
    success = bool(out) and _task_is_solved(task, idx, out)

    depth_nodes: dict[int, int] = {}
    with node_lock:
        for n in nodes.values():
            depth_nodes[n.depth] = int(depth_nodes.get(n.depth, 0)) + 1

    result = {
        "task_id": task_id,
        "idx": idx,
        "input": x,
        "step0_time": step0_time,
        "tot": {
            "step0": step0_debug,
            "scheme_b": {
                "trace_path": trace_path,
                "n_nodes": len(nodes),
                "propose_done": propose_done_cnt,
                "value_done": value_done_cnt,
                "depth_nodes": depth_nodes,
                "propose_done_by_depth": propose_done_by_depth,
                "value_done_by_depth": value_done_by_depth,
                "propose_samples_by_depth": propose_samples_by_depth,
                "value_samples_by_depth": value_samples_by_depth,
            },
        },
        "baseline": {
            "mode": "scheme_b",
            "output": out,
            "success": success,
            "tokens": tokens,
            "total_time": total_time,
            "search_time": search_time,
            "best_solution": best_solution,
        },
    }

    with open(os.path.join(log_dir, f"task_{task_id}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="scheme_b", choices=["serial", "layered", "scheme_b", "scheme_h", "dfs"])

    parser.add_argument("--backend", type=str, default="qwen3-32b-vllm")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=3)

    parser.add_argument("--n_propose_sample", type=int, default=1)
    parser.add_argument("--n_select_sample", type=int, default=5)
    parser.add_argument("--n_evaluate_sample", type=int, default=3)

    parser.add_argument("--propose_concurrency", type=int, default=4)
    parser.add_argument("--value_concurrency", type=int, default=32)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_children_per_parent", type=int, default=12)
    parser.add_argument("--step0_max_proposals", type=int, default=0)

    parser.add_argument("--early_stop", type=int, default=1)
    parser.add_argument("--stop_threshold", type=float, default=6.5)

    parser.add_argument("--min_expand_value", type=float, default=20.0)

    parser.add_argument("--log_interval_s", type=float, default=5.0)
    parser.add_argument("--log_batch_mix", type=int, default=0)
    parser.add_argument("--write_trace", type=int, default=0)
    parser.add_argument("--write_time_trace", type=int, default=0)
    parser.add_argument("--log_max_events_per_depth", type=int, default=50)

    parser.add_argument("--dfs_value_threshold", type=float, default=1.0)
    parser.add_argument("--dfs_time_limit_s", type=float, default=1200.0)

    args = parser.parse_args()
    args.early_stop = bool(args.early_stop)
    args.log_batch_mix = bool(args.log_batch_mix)
    args.write_trace = bool(args.write_trace)
    args.write_time_trace = bool(args.write_time_trace)

    task = Game24Task()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/game24/{args.mode}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)

    print(f"logs_dir: {log_dir}")

    results = []
    for task_id in range(int(args.start), int(args.end) + 1):
        idx = task_id - 1
        try:
            if args.mode == "serial":
                r = _run_serial(task, idx, args, log_dir)
            elif args.mode == "layered":
                r = _run_layered(task, idx, args, log_dir)
            elif args.mode == "dfs":
                r = _run_dfs(task, idx, args, log_dir)
            elif args.mode == "scheme_h":
                r = _run_scheme_h(task, idx, args, log_dir)
            else:
                r = _run_scheme_b(task, idx, args, log_dir)
            results.append(r)
            b = r["baseline"]
            print(
                f"Task {task_id} done: total={b['total_time']:.2f}s step0={r['step0_time']:.2f}s search={b['search_time']:.2f}s tokens={b['tokens']} success={'✓' if b['success'] else '✗'}"
            )
        except Exception as e:
            print(f"Task {task_id} failed: {e}")
            import traceback

            traceback.print_exc()

    with open(os.path.join(log_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"timestamp": datetime.now().isoformat(), "results": results}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
