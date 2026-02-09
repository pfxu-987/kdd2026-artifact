import sys
import os

sys.path.insert(0, "src")
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "dummy")

import argparse
import json
import queue
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any

import tot.models as models
from tot.models import request_extra_headers
from tot.tasks.gsm8k import GSM8KTask


_env_vllm_base_url = os.environ.get("VLLM_BASE_URL")
if _env_vllm_base_url:
    models.VLLM_BASE_URL = _env_vllm_base_url

_env_vllm_api_key = os.environ.get("VLLM_API_KEY")
if _env_vllm_api_key:
    models.VLLM_API_KEY = _env_vllm_api_key

_env_vllm_model = os.environ.get("VLLM_MODEL")
if _env_vllm_model:
    models.VLLM_MODEL = _env_vllm_model


def _task_is_solved(task: GSM8KTask, idx: int, y: str) -> bool:
    try:
        return bool(task.is_solved(idx, y))
    except Exception:
        return False


def _parse_proposals(task: GSM8KTask, parent_y: str, response: str, step: int) -> list[str]:
    if not response:
        return []
    try:
        return task.parse_proposals(parent_y, response, step)
    except TypeError:
        return task.parse_proposals(parent_y, response)


def _dedup_proposals(task: GSM8KTask, proposals: list[str]) -> list[str]:
    if hasattr(task, "dedup_proposals"):
        return task.dedup_proposals(proposals)
    seen = set()
    out = []
    for p in proposals:
        k = " ".join((p or "").split())
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(p)
    return out


def _get_value(task: GSM8KTask, x: str, y: str, n_evaluate_sample: int, model_name: str, task_id: int, step: int, parent_id: Any, branch_id: Any) -> float:
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


def _run_step0(task: GSM8KTask, x: str, args, task_id: int) -> tuple[list[str], float, dict[str, Any]]:
    start_t = time.time()
    prompt = task.propose_prompt_wrap(x, "", 0)
    extra_headers = {
        "X-TOT-Task-Id": str(task_id),
        "X-TOT-Depth": "0",
        "X-TOT-Parent-Idx": "root",
        "X-TOT-Branch-Idx": "root",
        "X-TOT-Call-Type": "propose",
    }
    with request_extra_headers(extra_headers):
        responses = models.gpt(
            prompt,
            n=max(1, int(args.n_propose_sample)),
            stop=getattr(task, "propose_stop", None),
            max_tokens=getattr(task, "propose_max_tokens", 1000),
            temperature=getattr(task, "propose_temperature", 0.7),
            model=args.backend,
        )

    proposals: list[str] = []
    for r in responses or []:
        proposals.extend(_parse_proposals(task, "", r, 0))
    proposals = _dedup_proposals(task, proposals)
    if int(args.max_children_per_parent) and len(proposals) > int(args.max_children_per_parent):
        proposals = proposals[: int(args.max_children_per_parent)]

    values = [0.0] * len(proposals)
    if proposals:
        with ThreadPoolExecutor(max_workers=int(args.value_concurrency)) as ex:
            futs = []
            for i, p in enumerate(proposals):
                futs.append(
                    (
                        i,
                        ex.submit(
                            _get_value,
                            task,
                            x,
                            p,
                            args.n_evaluate_sample,
                            args.backend,
                            task_id,
                            0,
                            "root",
                            i,
                        ),
                    )
                )
            for i, fut in futs:
                values[i] = float(fut.result())

    ids = list(range(len(proposals)))
    select_ids = sorted(ids, key=lambda i: values[i], reverse=True)[: int(args.n_select_sample)]
    selected = [proposals[i] for i in select_ids]
    step0_debug = {
        "proposals": proposals,
        "values": values,
        "selected_ids": select_ids,
        "selected": selected,
    }
    return selected, time.time() - start_t, step0_debug


def _run_serial(task: GSM8KTask, idx: int, args, log_dir: str) -> dict[str, Any]:
    task_id = idx + 1
    x = task.get_input(idx)

    models.completion_tokens = 0
    models.prompt_tokens = 0

    step0_candidates, step0_time, step0_debug = _run_step0(task, x, args, task_id)

    start_t = time.time()
    ys = step0_candidates
    best_y = ys[0] if ys else ""
    step_traces: list[dict[str, Any]] = []

    for step in range(1, int(task.steps)):
        new_ys: list[str] = []
        new_vs: list[float] = []

        for parent_idx, parent_y in enumerate(ys):
            prompt = task.propose_prompt_wrap(x, parent_y, step)
            extra_headers = {
                "X-TOT-Task-Id": str(task_id),
                "X-TOT-Depth": str(step),
                "X-TOT-Parent-Idx": str(parent_idx),
                "X-TOT-Branch-Idx": str(parent_idx),
                "X-TOT-Call-Type": "propose",
            }
            with request_extra_headers(extra_headers):
                responses = models.gpt(
                    prompt,
                    n=max(1, int(args.n_propose_sample)),
                    stop=getattr(task, "propose_stop", None),
                    max_tokens=getattr(task, "propose_max_tokens", 1000),
                    temperature=getattr(task, "propose_temperature", 0.7),
                    model=args.backend,
                )

            proposals: list[str] = []
            for r in responses or []:
                proposals.extend(_parse_proposals(task, parent_y, r, step))
            proposals = _dedup_proposals(task, proposals)
            if int(args.max_children_per_parent) and len(proposals) > int(args.max_children_per_parent):
                proposals = proposals[: int(args.max_children_per_parent)]

            for proposal_idx, p in enumerate(proposals):
                v = _get_value(task, x, p, args.n_evaluate_sample, args.backend, task_id, step, parent_idx, proposal_idx)
                new_ys.append(p)
                new_vs.append(float(v))

        if not new_ys:
            break

        ids = list(range(len(new_ys)))
        select_ids = sorted(ids, key=lambda i: new_vs[i], reverse=True)[: int(args.n_select_sample)]
        ys = [new_ys[i] for i in select_ids]
        best_y = ys[0] if ys else best_y

        step_traces.append(
            {
                "step": step,
                "n_candidates": len(new_ys),
                "selected_ids": select_ids,
                "selected": [new_ys[i] for i in select_ids],
                "selected_values": [float(new_vs[i]) for i in select_ids],
                "best": best_y,
            }
        )

        if args.early_stop and best_y and _task_is_solved(task, idx, best_y):
            break

    search_time = time.time() - start_t
    total_time = step0_time + search_time
    tokens = int(models.completion_tokens + models.prompt_tokens)
    out = best_y or ""
    success = bool(out) and _task_is_solved(task, idx, out)

    result = {
        "task_id": task_id,
        "idx": idx,
        "input": x,
        "step0_time": step0_time,
        "tot": {"step0": step0_debug, "steps": step_traces},
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


def _run_scheme_h(task: GSM8KTask, idx: int, args, log_dir: str) -> dict[str, Any]:
    task_id = idx + 1
    x = task.get_input(idx)

    models.completion_tokens = 0
    models.prompt_tokens = 0

    step0_time = 0.0
    step0_debug = None

    trace_fp = None
    trace_lock = threading.Lock()
    trace_path = None
    if bool(getattr(args, "write_trace", False)):
        trace_path = os.path.join(log_dir, f"trace_task_{task_id}.jsonl")
        trace_fp = open(trace_path, "w", encoding="utf-8")

    def _trace(event: dict[str, Any]) -> None:
        if trace_fp is None:
            return
        try:
            with trace_lock:
                trace_fp.write(json.dumps(event, ensure_ascii=False) + "\n")
                trace_fp.flush()
        except Exception:
            return

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

    task_q: deque[tuple[str, int]] = deque()

    def _enqueue_value(node_id: int) -> None:
        with node_lock:
            n = nodes.get(int(node_id))
            if n is None or n.invalid or n.value_enqueued:
                return
            if stop_depth is not None and int(n.depth) > int(stop_depth):
                return
            if n.score is not None:
                return
            n.value_enqueued = True
        task_q.append(("value", int(node_id)))

    def _enqueue_propose(node_id: int) -> None:
        with node_lock:
            n = nodes.get(int(node_id))
            if n is None or n.invalid:
                return
            if n.propose_enqueued or n.expanded:
                return
            if n.score is None:
                return
            if stop_depth is not None and int(n.depth) >= int(stop_depth):
                return
            if int(n.depth) >= int(task.steps) - 1:
                return
            st = _get_depth_state(int(n.depth))
            if not bool(st.get("pruned")):
                return
            if bool(st.get("closed")):
                return
            expand_target = st.get("expand_target")
            if expand_target is not None and int(st.get("propose_submitted", 0)) >= int(expand_target):
                return
            n.propose_enqueued = True
        task_q.append(("propose", int(node_id)))

    def _maybe_prune(depth: int) -> None:
        if int(depth) < 0:
            return
        if int(depth) >= int(task.steps):
            return
        if stop_depth is not None and int(depth) > int(stop_depth):
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
            valid_ids: list[int] = []
            for nid in ids:
                n = nodes.get(int(nid))
                if n is None or n.invalid:
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
            _invalidate_subtree(int(nid))

        with node_lock:
            st = _get_depth_state(int(depth))
            st["pruned"] = True
            st["valid_ids"] = set(keep_ids)
            expandable: list[int] = []
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
            _enqueue_propose(int(nid))

        if int(st.get("expand_target") or 0) <= 0:
            next_depth = int(depth) + 1
            if next_depth < int(task.steps):
                if stop_depth is not None and int(next_depth) > int(stop_depth):
                    return
                with node_lock:
                    st_next = depth_state.get(int(next_depth))
                    next_has_nodes = bool(st_next and (st_next.get("all_ids") or []))
                if next_has_nodes:
                    _maybe_prune(int(next_depth))

    def _do_propose(node_id: int) -> dict[str, Any]:
        with node_lock:
            node = nodes[int(node_id)]
            parent_y = node.y
            parent_depth = int(node.depth)
            child_depth = parent_depth + 1

        prompt = task.propose_prompt_wrap(x, parent_y, child_depth)
        extra_headers = {
            "X-TOT-Task-Id": str(task_id),
            "X-TOT-Depth": str(child_depth),
            "X-TOT-Parent-Idx": str(node_id),
            "X-TOT-Branch-Idx": "",
            "X-TOT-Call-Type": "propose",
        }
        with request_extra_headers(extra_headers):
            responses = models.gpt(
                prompt,
                n=max(1, int(args.n_propose_sample)),
                stop=getattr(task, "propose_stop", None),
                max_tokens=getattr(task, "propose_max_tokens", 1000),
                temperature=getattr(task, "propose_temperature", 0.7),
                model=args.backend,
            )

        proposals: list[str] = []
        for r in responses or []:
            proposals.extend(_parse_proposals(task, parent_y, r, child_depth))
        proposals = _dedup_proposals(task, proposals)
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
            node = nodes[int(node_id)]
            y = node.y
            depth = int(node.depth)
            parent_id = node.parent_id
        return _get_value(task, x, y, args.n_evaluate_sample, args.backend, task_id, depth, parent_id, 0)

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
        fut = ex.submit(fn, int(node_id))

        def _cb(f):
            try:
                done_q.put((typ, int(node_id), f.result(), None))
            except Exception as e:
                done_q.put((typ, int(node_id), None, e))

        fut.add_done_callback(_cb)

    best_solution = {"y": "", "node_id": None, "depth": None, "value": None}
    first_solution = {"time_s": None, "tokens": None}
    stop_depth: int | None = None

    propose_done_cnt = 0
    value_done_cnt = 0

    root_id = _alloc_node_id()
    root = Node(root_id, -1, "", None)
    root.score = 1.0
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

    def _stop_layer_complete(depth: int) -> bool:
        with node_lock:
            parent_depth = int(depth) - 1
            if parent_depth >= -1:
                st_parent = depth_state.get(int(parent_depth))
                if not (st_parent and bool(st_parent.get("closed"))):
                    return False
            st_cur = depth_state.get(int(depth))
            if st_cur is None:
                return False
            ids = list(st_cur.get("all_ids") or [])
            for nid in ids:
                n = nodes.get(int(nid))
                if n is None:
                    continue
                if n.invalid:
                    continue
                if n.score is None:
                    return False
        return True

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
                batch: list[tuple[str, int]] = []
                while task_q and len(batch) < int(args.batch_size):
                    batch.append(task_q.popleft())

                for typ, nid in batch:
                    if stop_event.is_set():
                        break

                    if typ == "propose":
                        with type_inflight_lock:
                            if propose_inflight >= max(1, int(args.propose_concurrency)):
                                task_q.append((typ, int(nid)))
                                continue
                        with node_lock:
                            node = nodes.get(int(nid))
                            if node is None or node.invalid:
                                continue
                            if stop_depth is not None and int(node.depth) >= int(stop_depth):
                                node.propose_enqueued = False
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
                                task_q.append((typ, int(nid)))
                                continue
                        with node_lock:
                            node = nodes.get(int(nid))
                            if node is None or node.invalid:
                                continue
                            node.value_enqueued = False
                            if stop_depth is not None and int(node.depth) > int(stop_depth):
                                continue
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
                        if node is None or node.invalid:
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
                        if first_solution.get("time_s") is None:
                            first_solution["time_s"] = float(time.time() - start_t)
                            first_solution["tokens"] = int(models.completion_tokens + models.prompt_tokens)

                        if args.early_stop and stop_depth is None:
                            stop_depth = int(depth)

                        if not best_solution.get("y"):
                            best_solution.update({"y": y_local, "node_id": nid, "depth": depth, "value": v})
                        elif (not args.early_stop) and float(v) > float(best_solution.get("value") or 0.0):
                            best_solution.update({"y": y_local, "node_id": nid, "depth": depth, "value": v})
                        if args.early_stop:
                            pass

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

                        if stop_depth is not None and int(child_depth) > int(stop_depth):
                            proposals = []

                        for child_y in proposals:
                            child_id = _alloc_node_id()
                            child = Node(child_id, int(child_depth), child_y, int(nid))
                            with node_lock:
                                nodes[child_id] = child
                                st_child["all_ids"].append(int(child_id))
                                parent = nodes.get(int(nid))
                                if parent is not None:
                                    parent.children.append(int(child_id))
                            _enqueue_value(child_id)

                    if expand_target is not None and propose_done_at_depth >= int(expand_target):
                        _mark_depth_closed(parent_depth)
                        if stop_depth is None or int(parent_depth + 1) <= int(stop_depth):
                            _maybe_prune(parent_depth + 1)

            now = time.time()
            if float(args.log_interval_s) and now - last_stats >= float(args.log_interval_s):
                with inflight_lock:
                    inflight_now = inflight
                print(
                    f"[STATS] inflight={inflight_now} qsize={len(task_q)} first_solution_time={first_solution.get('time_s')} best_depth={best_solution.get('depth')}"
                )
                last_stats = now

            if args.early_stop and stop_depth is not None:
                if _stop_layer_complete(int(stop_depth)):
                    stop_event.set()

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
                "stop_depth": stop_depth,
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
            "stop_depth": stop_depth,
        },
    }

    with open(os.path.join(log_dir, f"task_{task_id}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


def _run_dfs(task: GSM8KTask, idx: int, args, log_dir: str) -> dict[str, Any]:
    task_id = idx + 1
    x = task.get_input(idx)

    models.completion_tokens = 0
    models.prompt_tokens = 0

    step0_candidates, step0_time, step0_debug = _run_step0(task, x, args, task_id)

    start_t = time.time()
    dfs_start_t = start_t
    time_limit_s = float(getattr(args, "dfs_time_limit_s", 600.0) or 600.0)
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

        child_step = depth + 1
        dfs_stats["nodes_expanded"] = int(dfs_stats.get("nodes_expanded", 0)) + 1

        prompt = task.propose_prompt_wrap(x, parent_y, child_step)
        extra_headers = {
            "X-TOT-Task-Id": str(task_id),
            "X-TOT-Depth": str(child_step),
            "X-TOT-Parent-Idx": str(depth),
            "X-TOT-Branch-Idx": "0",
            "X-TOT-Call-Type": "propose",
        }
        with request_extra_headers(extra_headers):
            responses = models.gpt(
                prompt,
                n=max(1, int(args.n_propose_sample)),
                stop=getattr(task, "propose_stop", None),
                max_tokens=getattr(task, "propose_max_tokens", 1000),
                temperature=getattr(task, "propose_temperature", 0.7),
                model=args.backend,
            )

        proposals: list[str] = []
        for r in responses or []:
            proposals.extend(_parse_proposals(task, parent_y, r, child_step))
        proposals = _dedup_proposals(task, proposals)
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
            v = _get_value(task, x, child_y, args.n_evaluate_sample, args.backend, task_id, child_step, depth, proposal_idx)
            vals.append(float(v))

        order = sorted(range(len(proposals)), key=lambda i: vals[i], reverse=True)
        for i in order:
            if _timed_out():
                dfs_stats["timeout"] = True
                return False, ""

            child_y = proposals[i]
            v = float(vals[i])

            if best_solution.get("value") is None or v > float(best_solution.get("value") or 0.0):
                best_solution = {"y": child_y, "value": v, "depth": child_step}
                dfs_stats["best_solution"] = best_solution

            if v < value_threshold:
                dfs_stats["pruned_by_threshold"] = int(dfs_stats.get("pruned_by_threshold", 0)) + 1
                continue

            dfs_stats["path"].append({"depth": child_step, "value": v, "y": child_y})
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


def _run_layered(task: GSM8KTask, idx: int, args, log_dir: str) -> dict[str, Any]:
    task_id = idx + 1
    x = task.get_input(idx)

    models.completion_tokens = 0
    models.prompt_tokens = 0

    step0_candidates, step0_time, step0_debug = _run_step0(task, x, args, task_id)

    start_t = time.time()
    ys = step0_candidates
    best_y = ys[0] if ys else ""
    step_traces: list[dict[str, Any]] = []

    def _do_propose(parent_idx: int, parent_y: str, step: int) -> tuple[int, list[str]]:
        prompt = task.propose_prompt_wrap(x, parent_y, step)
        extra_headers = {
            "X-TOT-Task-Id": str(task_id),
            "X-TOT-Depth": str(step),
            "X-TOT-Parent-Idx": str(parent_idx),
            "X-TOT-Branch-Idx": str(parent_idx),
            "X-TOT-Call-Type": "propose",
        }
        with request_extra_headers(extra_headers):
            responses = models.gpt(
                prompt,
                n=max(1, int(args.n_propose_sample)),
                stop=getattr(task, "propose_stop", None),
                max_tokens=getattr(task, "propose_max_tokens", 1000),
                temperature=getattr(task, "propose_temperature", 0.7),
                model=args.backend,
            )

        proposals: list[str] = []
        for r in responses or []:
            proposals.extend(_parse_proposals(task, parent_y, r, step))
        proposals = _dedup_proposals(task, proposals)
        if int(args.max_children_per_parent) and len(proposals) > int(args.max_children_per_parent):
            proposals = proposals[: int(args.max_children_per_parent)]
        return parent_idx, proposals

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
        for parent_idx, proposals in parent_results:
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
                    futs.append(
                        (
                            p,
                            ex.submit(
                                _get_value,
                                task,
                                x,
                                p,
                                args.n_evaluate_sample,
                                args.backend,
                                task_id,
                                step,
                                parent_idx,
                                proposal_idx,
                            ),
                        )
                    )
                for p, fut in futs:
                    new_ys.append(p)
                    new_vs.append(float(fut.result()))
        else:
            for parent_idx, proposal_idx, p in all_candidates:
                v = _get_value(task, x, p, args.n_evaluate_sample, args.backend, task_id, step, parent_idx, proposal_idx)
                new_ys.append(p)
                new_vs.append(float(v))

        ids = list(range(len(new_ys)))
        select_ids = sorted(ids, key=lambda i: new_vs[i], reverse=True)[: int(args.n_select_sample)]
        ys = [new_ys[i] for i in select_ids]
        best_y = ys[0] if ys else best_y

        step_traces.append(
            {
                "step": step,
                "n_candidates": len(new_ys),
                "selected_ids": select_ids,
                "selected": [new_ys[i] for i in select_ids],
                "selected_values": [float(new_vs[i]) for i in select_ids],
                "best": best_y,
            }
        )

        if args.early_stop and best_y and _task_is_solved(task, idx, best_y):
            break

    search_time = time.time() - start_t
    total_time = step0_time + search_time
    tokens = int(models.completion_tokens + models.prompt_tokens)
    out = best_y or ""
    success = bool(out) and _task_is_solved(task, idx, out)

    result = {
        "task_id": task_id,
        "idx": idx,
        "input": x,
        "step0_time": step0_time,
        "tot": {"step0": step0_debug, "steps": step_traces},
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


def _run_scheme_b(task: GSM8KTask, idx: int, args, log_dir: str) -> dict[str, Any]:
    task_id = idx + 1
    x = task.get_input(idx)

    models.completion_tokens = 0
    models.prompt_tokens = 0

    step0_time = 0.0
    step0_debug = None

    trace_fp = None
    trace_lock = threading.Lock()
    trace_path = None
    if bool(getattr(args, "write_trace", False)):
        trace_path = os.path.join(log_dir, f"trace_task_{task_id}.jsonl")
        trace_fp = open(trace_path, "w", encoding="utf-8")

    def _trace(event: dict[str, Any]) -> None:
        if trace_fp is None:
            return
        try:
            with trace_lock:
                trace_fp.write(json.dumps(event, ensure_ascii=False) + "\n")
                trace_fp.flush()
        except Exception:
            return

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
            if node.expanded or node.depth >= int(task.steps) - 1:
                return False
            st = _get_depth_state(node.depth)
            if st["expanded_count"] >= int(args.n_select_sample):
                return False
        topk = _topk_ids(nodes[node_id].depth)
        return node_id in topk

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

        prompt = task.propose_prompt_wrap(x, parent_y, child_depth)
        extra_headers = {
            "X-TOT-Task-Id": str(task_id),
            "X-TOT-Depth": str(child_depth),
            "X-TOT-Parent-Idx": str(node_id),
            "X-TOT-Branch-Idx": "",
            "X-TOT-Call-Type": "propose",
        }
        with request_extra_headers(extra_headers):
            responses = models.gpt(
                prompt,
                n=max(1, int(args.n_propose_sample)),
                stop=getattr(task, "propose_stop", None),
                max_tokens=getattr(task, "propose_max_tokens", 1000),
                temperature=getattr(task, "propose_temperature", 0.7),
                model=args.backend,
            )

        proposals: list[str] = []
        for r in responses or []:
            proposals.extend(_parse_proposals(task, parent_y, r, child_depth))
        proposals = _dedup_proposals(task, proposals)
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
        return _get_value(task, x, y, args.n_evaluate_sample, args.backend, task_id, depth, parent_id, 0)

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
    root.score = 1.0
    nodes[root_id] = root
    _enqueue_propose(root_id, priority=0.0)

    propose_exec = ThreadPoolExecutor(max_workers=int(args.propose_concurrency))
    value_exec = ThreadPoolExecutor(max_workers=int(args.value_concurrency))

    start_t = time.time()
    last_stats = time.time()

    try:
        while True:
            if stop_event.is_set():
                break

            batch: list[tuple[float, int, str, int]] = []
            while len(batch) < int(args.batch_size) and not task_q.empty():
                batch.append(task_q.get())

            for _pri, _s, typ, nid in batch:
                if typ == "propose":
                    with node_lock:
                        node = nodes.get(nid)
                        if node is None or node.expanded:
                            continue
                        node.expanded = True
                        st = _get_depth_state(node.depth)
                        st["expanded_count"] += 1
                    _submit(propose_exec, _do_propose, "propose", nid)
                else:
                    _submit(value_exec, _do_value, "value", nid)

            processed_any = False
            while True:
                try:
                    typ, nid, res, err = done_q.get_nowait()
                except queue.Empty:
                    break
                processed_any = True
                _inflight_dec()

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
    parser.add_argument("--gsm8k_file", type=str, default="test.jsonl")
    parser.add_argument("--gsm8k_steps", type=int, default=6)

    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=3)

    parser.add_argument("--n_propose_sample", type=int, default=8)
    parser.add_argument("--n_select_sample", type=int, default=5)
    parser.add_argument("--n_evaluate_sample", type=int, default=3)

    parser.add_argument("--propose_concurrency", type=int, default=4)
    parser.add_argument("--value_concurrency", type=int, default=32)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_children_per_parent", type=int, default=24)

    parser.add_argument("--early_stop", type=int, default=1)

    parser.add_argument("--log_interval_s", type=float, default=5.0)

    parser.add_argument("--write_trace", type=int, default=0)
    parser.add_argument("--log_max_events_per_depth", type=int, default=50)

    parser.add_argument("--dfs_value_threshold", type=float, default=1.0)
    parser.add_argument("--dfs_time_limit_s", type=float, default=600.0)

    args = parser.parse_args()
    args.early_stop = bool(args.early_stop)
    args.write_trace = bool(args.write_trace)

    task = GSM8KTask(file=args.gsm8k_file, steps=int(args.gsm8k_steps))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/gsm8k/{args.mode}_{timestamp}"
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
