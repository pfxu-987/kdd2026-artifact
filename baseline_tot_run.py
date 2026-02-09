import sys
import os

sys.path.insert(0, "src")
os.environ["OPENAI_API_KEY"] = "dummy"

import argparse
import json
import os
import copy
import queue
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any

from tot.tasks import get_task
import tot.models as models
from tot.models import request_extra_headers


call_log = {"propose": [], "value": []}
call_log_lock = threading.Lock()
value_cache_lock = threading.Lock()


def _task_propose_prompt(task, x: str, y: str, step: int):
    try:
        return task.propose_prompt_wrap(x, y, step)
    except TypeError:
        return task.propose_prompt_wrap(x, y)


def _task_parse_proposals(task, parent_y: str, response: str, step: int):
    if hasattr(task, "parse_proposals"):
        try:
            return task.parse_proposals(parent_y, response, step)
        except TypeError:
            return task.parse_proposals(parent_y, response)
    return parse_proposals(parent_y, response)


def _task_parse_proposals_from_responses(task, x: str, parent_y: str, responses: list[str], step: int) -> list[str]:
    if hasattr(task, "propose_outputs_unwrap"):
        try:
            return task.propose_outputs_unwrap(x, parent_y, responses)
        except TypeError:
            return task.propose_outputs_unwrap(x, parent_y, responses, getattr(task, "n_max_propose", -1))
    proposals: list[str] = []
    for resp in (responses or []):
        proposals.extend(_task_parse_proposals(task, parent_y, resp, step))
    return proposals


def _task_is_dead_end(task, y: str) -> bool:
    if hasattr(task, "is_dead_end"):
        return bool(task.is_dead_end(y))
    return is_dead_end(y)


def _task_is_solved(task, idx: int, y: str) -> bool:
    if hasattr(task, "is_solved"):
        return bool(task.is_solved(idx, y))
    # Default to old Game24 heuristic.
    return ("left: 24" in y.lower()) or y.strip().endswith("24")


def reset_call_log():
    global call_log
    with call_log_lock:
        if not isinstance(call_log, dict):
            call_log = {"propose": [], "value": []}
            return
        call_log.setdefault("propose", [])
        call_log.setdefault("value", [])
        call_log["propose"].clear()
        call_log["value"].clear()


def log_call(call_type, prompt, response, step, parent_idx=None):
    global call_log
    entry = {
        "step": step,
        "parent_idx": parent_idx,
        "prompt_length": len(prompt),
        "response_length": len(response[0]) if response else 0,
    }
    with call_log_lock:
        call_log[call_type].append(entry)


def parse_proposals(y, response):
    import re

    current_numbers = (
        y.strip().split("\n")[-1].split("left: ")[-1].split(")")[0]
        if y and "left: " in y
        else ""
    )

    if current_numbers == "24":
        answer_lines = [
            line
            for line in response.split("\n")
            if "answer" in line.lower() and "=" in line
        ]
        if answer_lines:
            return [y + answer_lines[0] + "\n"]
        return []

    lines = response.split("\n")
    cleaned_proposals = []

    i = 0
    while i < len(lines):
        line = lines[i]
        line_clean = re.sub(r"\*\*", "", line)
        line_clean = re.sub(r"#+", "", line_clean)
        line_clean = re.sub(r"→", ":", line_clean)
        line_clean = re.sub(r"^\d+\.\s*", "", line_clean)
        line_clean = re.sub(r"^-\s+", "", line_clean)
        line_clean = line_clean.strip()

        expr_match = re.search(
            r"(\-?\d+\.?\d*\s*[\+\-\*\/]\s*\-?\d+\.?\d*\s*=\s*\-?[\d\.]+)",
            line_clean,
        )

        if expr_match:
            expr = expr_match.group(1).strip()
            left_nums = None

            left_match = re.search(
                r"(?:left|remaining|Left|Remaining):\s*(.+?)(?:\)|$)",
                line_clean,
                re.IGNORECASE,
            )
            if left_match:
                left_nums = left_match.group(1).strip()
            elif "(" in line_clean and ")" in line_clean:
                paren_match = re.search(
                    r"\((?:left|remaining):\s*([^\)]+)\)",
                    line_clean,
                    re.IGNORECASE,
                )
                if paren_match:
                    left_nums = paren_match.group(1).strip()
            elif i + 1 < len(lines):
                next_line = lines[i + 1]
                next_clean = re.sub(r"\*\*", "", next_line).strip()
                left_match = re.search(
                    r"(?:left|remaining|Left|Remaining):\s*(.+?)(?:\)|$)",
                    next_clean,
                    re.IGNORECASE,
                )
                if left_match:
                    left_nums = left_match.group(1).strip()
                    i += 1

            if left_nums:
                left_nums = re.sub(r",\s*", " ", left_nums)
                left_nums = left_nums.replace("`", "")
                left_nums = re.sub(r"\s+", " ", left_nums)
                cleaned_proposals.append(f"{expr} (left: {left_nums})")

        i += 1

    return [y + _ + "\n" for _ in cleaned_proposals]


def extract_remaining_numbers(y):
    if not y or not y.strip():
        return []

    last_line = y.strip().split("\n")[-1]
    if "left: " in last_line:
        nums_str = last_line.split("left: ")[-1].split(")")[0]
        nums = nums_str.strip().split()
        return nums
    return []


def is_dead_end(y):
    nums = extract_remaining_numbers(y)
    if len(nums) == 1:
        try:
            value = float(nums[0])
            return abs(value - 24) > 0.01
        except Exception:
            return True
    return False


def get_value(
    task,
    x,
    y,
    n_evaluate_sample,
    step,
    model_name="qwen3-32b",
    log_it=False,
    task_id=None,
    parent_idx=None,
    branch_idx=None,
):
    value_prompt = task.value_prompt_wrap(x, y)
    with value_cache_lock:
        if value_prompt in task.value_cache:
            return task.value_cache[value_prompt]

    extra_headers = {
        "X-TOT-Task-Id": str(task_id) if task_id is not None else "",
        "X-TOT-Depth": str(step),
        "X-TOT-Parent-Idx": str(parent_idx) if parent_idx is not None else "",
        "X-TOT-Branch-Idx": str(branch_idx) if branch_idx is not None else "",
        "X-TOT-Call-Type": "value",
    }

    with request_extra_headers(extra_headers):
        value_outputs = models.gpt(
            value_prompt,
            n=n_evaluate_sample,
            stop=getattr(task, "value_stop", None),
            max_tokens=getattr(task, "value_max_tokens", 1000),
            temperature=getattr(task, "value_temperature", 0.0),
            model=model_name,
        )

    if log_it:
        log_call("value", value_prompt, value_outputs, step, parent_idx)

    value = task.value_outputs_unwrap(x, y, value_outputs)
    with value_cache_lock:
        task.value_cache[value_prompt] = value
    return value


def run_step0(task, x, args, model_name="qwen3-32b", task_id=None):
    start_time = time.time()
    print(f"[STEP0] Starting for task_id={task_id}")

    propose_start = time.time()
    propose_prompt = _task_propose_prompt(task, x, "", 0)
    extra_headers = {
        "X-TOT-Task-Id": str(task_id) if task_id is not None else "",
        "X-TOT-Depth": "0",
        "X-TOT-Parent-Idx": "root",
        "X-TOT-Branch-Idx": "root",
        "X-TOT-Call-Type": "propose",
    }
    with request_extra_headers(extra_headers):
        response = models.gpt(
            propose_prompt,
            n=max(1, getattr(args, "n_propose_sample", 1)),
            stop=getattr(task, "propose_stop", None),
            max_tokens=getattr(task, "propose_max_tokens", 1000),
            temperature=getattr(task, "propose_temperature", 0.7),
            model=model_name,
        )
    propose_time = time.time() - propose_start

    trace_step0 = None
    if getattr(args, "trace_search", 0):
        trace_step0 = {
            "propose_prompt": propose_prompt,
            "propose_responses": response,
            "proposals": [],
        }

    proposals = _task_parse_proposals_from_responses(task, x, "", response, 0)

    if hasattr(task, "dedup_proposals"):
        proposals = task.dedup_proposals(proposals)

    step0_max_proposals = getattr(args, "step0_max_proposals", None)
    if step0_max_proposals is None:
        max_children = getattr(args, "max_children_per_parent", None)
        if max_children:
            step0_max_proposals = max(int(getattr(args, "n_select_sample", 1)) * int(max_children), int(max_children))
    if step0_max_proposals and len(proposals) > int(step0_max_proposals):
        proposals = proposals[: int(step0_max_proposals)]
    print(f"[STEP0] Propose done: {len(proposals)} proposals in {propose_time:.2f}s")

    values = [0.0] * len(proposals)
    value_start = time.time()
    if args.value_concurrency and args.value_concurrency > 1 and len(proposals) > 1:
        print(f"[STEP0] Evaluating {len(proposals)} proposals with concurrency={args.value_concurrency}")
        with ThreadPoolExecutor(max_workers=args.value_concurrency) as ex:
            futures = []
            for proposal_idx, proposal in enumerate(proposals):
                futures.append(
                    (
                        proposal_idx,
                        ex.submit(
                            get_value,
                            task,
                            x,
                            proposal,
                            args.n_evaluate_sample,
                            0,
                            model_name,
                            False,
                            task_id,
                            "root",
                            proposal_idx,
                        ),
                    )
                )
            for proposal_idx, fut in futures:
                values[proposal_idx] = fut.result()
    else:
        print(f"[STEP0] Evaluating {len(proposals)} proposals sequentially (SLOW!)")
        for proposal_idx, proposal in enumerate(proposals):
            value = get_value(
                task,
                x,
                proposal,
                args.n_evaluate_sample,
                0,
                model_name,
                task_id=task_id,
                parent_idx="root",
                branch_idx=proposal_idx,
            )
            values[proposal_idx] = value
    value_time = time.time() - value_start

    ids = list(range(len(proposals)))
    select_ids = sorted(ids, key=lambda i: values[i], reverse=True)[: args.n_select_sample]
    selected = [proposals[i] for i in select_ids]
    selected_values = [values[i] for i in select_ids]

    if trace_step0 is not None:
        selected_set = set(select_ids)
        for i, p in enumerate(proposals):
            trace_step0["proposals"].append(
                {
                    "proposal_idx": i,
                    "proposal": p,
                    "value": values[i],
                    "selected": i in selected_set,
                }
            )

    elapsed_time = time.time() - start_time
    print(f"[STEP0] Done: selected {len(selected)}/{len(proposals)} candidates, "
          f"values={[f'{v:.2f}' for v in selected_values]}, "
          f"total_time={elapsed_time:.2f}s (propose={propose_time:.2f}s, value={value_time:.2f}s)")
    return selected, elapsed_time, trace_step0


def run_baseline(task, x, step0_candidates, args, model_name="qwen3-32b", task_id=None):
    reset_call_log()
    ys = step0_candidates

    total_filtered = 0
    step_times = {}

    trace = [] if getattr(args, "trace_search", 0) else None

    for step in range(1, task.steps):
        step_start_time = time.time()

        new_ys = []
        values = []

        def _do_propose(parent_idx: int, parent_y: str):
            propose_prompt = _task_propose_prompt(task, x, parent_y, step)
            extra_headers = {
                "X-TOT-Task-Id": str(task_id) if task_id is not None else "",
                "X-TOT-Depth": str(step),
                "X-TOT-Parent-Idx": str(parent_idx),
                "X-TOT-Branch-Idx": str(parent_idx),
                "X-TOT-Call-Type": "propose",
            }
            with request_extra_headers(extra_headers):
                response = models.gpt(
                    propose_prompt,
                    n=max(1, getattr(args, "n_propose_sample", 1)),
                    stop=getattr(task, "propose_stop", None),
                    max_tokens=getattr(task, "propose_max_tokens", 1000),
                    temperature=getattr(task, "propose_temperature", 0.7),
                    model=model_name,
                )
            proposals = _task_parse_proposals_from_responses(task, x, parent_y, response, step)

            if hasattr(task, "dedup_proposals"):
                proposals = task.dedup_proposals(proposals)
            log_call("propose", propose_prompt, response, step, parent_idx)
            return parent_idx, parent_y, propose_prompt, response, proposals

        if args.schedule_mode == "layered":
            parent_proposals: list[tuple[int, str, list[str]]] = []
            if args.propose_concurrency and args.propose_concurrency > 1 and len(ys) > 1:
                with ThreadPoolExecutor(max_workers=args.propose_concurrency) as ex:
                    futs = [
                        ex.submit(_do_propose, parent_idx, parent_y)
                        for parent_idx, parent_y in enumerate(ys)
                    ]
                    for fut in futs:
                        parent_proposals.append(fut.result())
            else:
                for parent_idx, parent_y in enumerate(ys):
                    parent_proposals.append(_do_propose(parent_idx, parent_y))

            all_candidates: list[tuple[int, int, str]] = []
            propose_meta = {}
            proposals_by_parent = {}
            for parent_idx, parent_y, propose_prompt, propose_response, proposals in parent_proposals:
                propose_meta[parent_idx] = {
                    "parent_idx": parent_idx,
                    "parent_y": parent_y,
                    "propose_prompt": propose_prompt,
                    "propose_responses": propose_response,
                }
                proposals_by_parent[parent_idx] = proposals
                for proposal_idx, proposal in enumerate(proposals):
                    all_candidates.append((parent_idx, proposal_idx, proposal))

            if trace is not None:
                parents_trace = []
                for parent_idx in sorted(propose_meta.keys()):
                    parents_trace.append(
                        {
                            **propose_meta[parent_idx],
                            "proposals": [
                                {
                                    "proposal_idx": i,
                                    "proposal": p,
                                    "value": None,
                                    "selected": False,
                                }
                                for i, p in enumerate(proposals_by_parent.get(parent_idx, []))
                            ],
                        }
                    )
                trace.append({"step": step, "parents": parents_trace})

            value_by_parent = {}

            if args.value_concurrency and args.value_concurrency > 1 and len(all_candidates) > 1:
                with ThreadPoolExecutor(max_workers=args.value_concurrency) as ex:
                    futs = []
                    for parent_idx, proposal_idx, proposal in all_candidates:
                        futs.append(
                            (
                                parent_idx,
                                proposal_idx,
                                proposal,
                                ex.submit(
                                    get_value,
                                    task,
                                    x,
                                    proposal,
                                    args.n_evaluate_sample,
                                    step,
                                    model_name,
                                    True,
                                    task_id,
                                    parent_idx,
                                    proposal_idx,
                                ),
                            )
                        )
                    for parent_idx, proposal_idx, proposal, fut in futs:
                        new_ys.append(proposal)
                        v = fut.result()
                        values.append(v)
                        value_by_parent[(parent_idx, proposal_idx)] = v
            else:
                for parent_idx, proposal_idx, proposal in all_candidates:
                    value = get_value(
                        task,
                        x,
                        proposal,
                        args.n_evaluate_sample,
                        step,
                        model_name,
                        log_it=True,
                        task_id=task_id,
                        parent_idx=parent_idx,
                        branch_idx=proposal_idx,
                    )
                    new_ys.append(proposal)
                    values.append(value)
                    value_by_parent[(parent_idx, proposal_idx)] = value
        else:
            for parent_idx, parent_y in enumerate(ys):
                parent_idx, parent_y, propose_prompt, propose_response, proposals = _do_propose(parent_idx, parent_y)

                if trace is not None:
                    trace.append(
                        {
                            "step": step,
                            "parents": [
                                {
                                    "parent_idx": parent_idx,
                                    "parent_y": parent_y,
                                    "propose_prompt": propose_prompt,
                                    "propose_responses": propose_response,
                                    "proposals": [
                                        {
                                            "proposal_idx": i,
                                            "proposal": p,
                                            "value": None,
                                            "selected": False,
                                        }
                                        for i, p in enumerate(proposals)
                                    ],
                                }
                            ],
                        }
                    )

                if args.value_concurrency and args.value_concurrency > 1 and len(proposals) > 1:
                    with ThreadPoolExecutor(max_workers=args.value_concurrency) as ex:
                        futures = []
                        for proposal_idx, proposal in enumerate(proposals):
                            futures.append(
                                (
                                    proposal_idx,
                                    proposal,
                                    ex.submit(
                                        get_value,
                                        task,
                                        x,
                                        proposal,
                                        args.n_evaluate_sample,
                                        step,
                                        model_name,
                                        True,
                                        task_id,
                                        parent_idx,
                                        proposal_idx,
                                    ),
                                )
                            )
                        for proposal_idx, proposal, fut in futures:
                            new_ys.append(proposal)
                            v = fut.result()
                            values.append(v)
                            if trace is not None:
                                trace[-1]["parents"][0]["proposals"][proposal_idx]["value"] = v
                else:
                    for proposal_idx, proposal in enumerate(proposals):
                        value = get_value(
                            task,
                            x,
                            proposal,
                            args.n_evaluate_sample,
                            step,
                            model_name,
                            log_it=True,
                            task_id=task_id,
                            parent_idx=parent_idx,
                            branch_idx=proposal_idx,
                        )
                        new_ys.append(proposal)
                        values.append(value)
                        if trace is not None:
                            trace[-1]["parents"][0]["proposals"][proposal_idx]["value"] = value

        if len(new_ys) == 0:
            step_times[f"step{step}"] = time.time() - step_start_time
            break

        ids = list(range(len(new_ys)))
        select_ids = sorted(ids, key=lambda i: values[i], reverse=True)[: args.n_select_sample]
        select_ys = [new_ys[i] for i in select_ids]

        if trace is not None:
            selected_set = set(select_ids)
            # Mark selected at the last trace step record.
            last = trace[-1]
            idx_cursor = 0
            for parent_entry in last.get("parents", []):
                for p in parent_entry.get("proposals", []):
                    if idx_cursor in selected_set:
                        p["selected"] = True
                    idx_cursor += 1
            # Fill values for layered mode if present.
            if "value_by_parent" in locals():
                for parent_entry in last.get("parents", []):
                    parent_idx = parent_entry.get("parent_idx")
                    for p in parent_entry.get("proposals", []):
                        key = (parent_idx, p.get("proposal_idx"))
                        if key in value_by_parent:
                            p["value"] = value_by_parent[key]

        before_filter = len(select_ys)
        select_ys = [y for y in select_ys if not _task_is_dead_end(task, y)]
        filtered = before_filter - len(select_ys)
        total_filtered += filtered

        step_times[f"step{step}"] = time.time() - step_start_time

        ys = select_ys
        if len(ys) == 0:
            break

    log_copy = call_log.copy()
    log_copy["total_filtered"] = total_filtered
    log_copy["step_times"] = step_times
    if trace is not None:
        log_copy["trace"] = trace
    return ys, log_copy


def _assess_gsm8k_difficulty(model_name: str, question: str) -> dict[str, Any]:
    prompt = (
        "You are given a grade-school math word problem. Classify its difficulty for multi-step reasoning as an integer 1-5.\n"
        "Rubric (choose the BEST fit; avoid defaulting to 3):\n"
        "1=single operation, direct from one number to answer.\n"
        "2=two operations or a simple ratio/percentage, small reasoning chain.\n"
        "3=multi-step (3-4 operations), requires bookkeeping or combining conditions.\n"
        "4=multi-step with careful case/remaining/sequence logic, higher error risk.\n"
        "5=very long chain, tricky constraints, easy to make mistakes without careful planning.\n\n"
        "Examples (format only; do NOT copy text):\n"
        "Problem: 'Alice has 3 apples and buys 2 more. How many apples?' -> {\"difficulty\":1,\"rationale\":\"single addition\"}\n"
        "Problem: 'Ratio 1:3 with 45 wires; find poles' -> {\"difficulty\":2,\"rationale\":\"simple ratio conversion\"}\n"
        "Problem: 'Two-stage discount then tax' -> {\"difficulty\":3,\"rationale\":\"several arithmetic steps\"}\n"
        "Problem: 'People leave/return across multiple time windows; final count' -> {\"difficulty\":4,\"rationale\":\"state bookkeeping across phases\"}\n"
        "Problem: 'Multiple nested fractions with remainders and conditionals' -> {\"difficulty\":5,\"rationale\":\"long chain with tricky constraints\"}\n\n"
        "Output ONLY a JSON object with keys: difficulty (int 1-5), rationale (short).\n"
        "Do NOT output markdown fences or extra text.\n\n"
        f"Problem: {question}\n\n"
        "JSON:"
    )
    raw = ""
    try:
        resp = models.gpt(
            prompt,
            n=1,
            stop=None,
            max_tokens=128,
            temperature=0.0,
            model=model_name,
        )
        raw = (resp[0] if resp else "").strip()
        text = raw
        # Best-effort JSON parse: extract the outermost {...}.
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]
        data = json.loads(text)
        d = int(data.get("difficulty", 3))
        d = max(1, min(5, d))
        return {
            "difficulty": d,
            "rationale": str(data.get("rationale", "")),
            "raw": raw,
        }
    except Exception:
        return {"difficulty": 3, "rationale": "", "raw": raw}


def _adaptive_beam_settings(args, question: str) -> dict[str, Any]:
    info = _assess_gsm8k_difficulty(args.backend, question)
    d = int(info.get("difficulty", 3))

    # Default mapping: easy -> narrow (1-3), hard -> wider (5-10)
    # Users can override via CLI: e.g. "1:2,2:3,3:5,4:8,5:10"
    mapping = {1: 2, 2: 3, 3: 5, 4: 8, 5: 10}
    raw = getattr(args, "adaptive_beam_map", "")
    if raw:
        try:
            for part in raw.split(","):
                if not part.strip():
                    continue
                k, v = part.split(":")
                mapping[int(k.strip())] = int(v.strip())
        except Exception:
            pass

    k = int(mapping.get(d, args.n_select_sample))
    k = max(1, k)

    # Optional per-difficulty max_expansions mapping.
    # Empty string => do not override.
    exp_map = {}
    raw_exp = getattr(args, "adaptive_max_expansions_map", "")
    if raw_exp:
        try:
            for part in raw_exp.split(","):
                if not part.strip():
                    continue
                dk, dv = part.split(":")
                exp_map[int(dk.strip())] = int(dv.strip())
        except Exception:
            exp_map = {}

    max_expansions = None
    if exp_map:
        max_expansions = int(exp_map.get(d, args.max_expansions))
        max_expansions = max(1, max_expansions)

    return {
        "difficulty": d,
        "difficulty_rationale": info.get("rationale", ""),
        "difficulty_raw": info.get("raw", ""),
        "n_select_sample": k,
        "max_expansions": max_expansions,
    }


def run_dfs(task, x, step0_candidates, args, model_name="qwen3-32b", task_id=None):
    reset_call_log()
    trace = [] if getattr(args, "trace_search", 0) else None

    expansions_used = 0
    solutions_found = 0
    best_solution = {"y": "", "depth": None, "value": None}
    solved_y = ""
    solved_value = None

    dfs_start_t = time.time()
    time_limit_s = float(getattr(args, "dfs_time_limit_s", 0.0) or 0.0)
    value_threshold = float(getattr(args, "dfs_value_threshold", 0.0) or 0.0)
    dfs_timeout = False

    def _timed_out() -> bool:
        if time_limit_s <= 0:
            return False
        return (time.time() - dfs_start_t) >= time_limit_s

    # Stack items: (depth, y)
    stack: list[tuple[int, str]] = []
    for y0 in reversed(step0_candidates):
        stack.append((0, y0))

    while stack and expansions_used < getattr(args, "max_expansions", 500):
        if _timed_out():
            dfs_timeout = True
            break
        depth, parent_y = stack.pop()
        if not parent_y:
            continue

        if _task_is_solved(task, task_id - 1 if task_id is not None else 0, parent_y):
            solutions_found += 1
            step_for_value = max(0, int(depth))
            if _timed_out():
                dfs_timeout = True
                break
            v = get_value(
                task,
                x,
                parent_y,
                args.n_evaluate_sample,
                step_for_value,
                model_name,
                True,
                task_id,
                0,
                0,
            )
            if best_solution.get("value") is None or float(v) > float(best_solution.get("value") or 0.0):
                best_solution = {"y": parent_y, "depth": int(depth), "value": float(v)}

            if value_threshold <= 0 or float(v) >= float(value_threshold):
                if not solved_y:
                    solved_y = parent_y
                    solved_value = float(v)
                if args.early_stop:
                    break
            continue

        # Do not expand beyond max depth.
        if depth >= task.steps - 1:
            continue

        step = depth + 1
        expansions_used += 1

        propose_prompt = _task_propose_prompt(task, x, parent_y, step)
        extra_headers = {
            "X-TOT-Task-Id": str(task_id) if task_id is not None else "",
            "X-TOT-Depth": str(step),
            "X-TOT-Parent-Idx": "0",
            "X-TOT-Branch-Idx": "0",
            "X-TOT-Call-Type": "propose",
        }
        with request_extra_headers(extra_headers):
            response = models.gpt(
                propose_prompt,
                n=max(1, getattr(args, "n_propose_sample", 1)),
                stop=getattr(task, "propose_stop", None),
                max_tokens=getattr(task, "propose_max_tokens", 1000),
                temperature=getattr(task, "propose_temperature", 0.7),
                model=model_name,
            )
        proposals = _task_parse_proposals_from_responses(task, x, parent_y, response, step)
        if hasattr(task, "dedup_proposals"):
            proposals = task.dedup_proposals(proposals)
        if args.max_children_per_parent and len(proposals) > args.max_children_per_parent:
            proposals = proposals[: args.max_children_per_parent]
        log_call("propose", propose_prompt, response, step, 0)

        values: list[float] = []
        for proposal_idx, child_y in enumerate(proposals):
            if _timed_out():
                dfs_timeout = True
                break
            v = get_value(
                task,
                x,
                child_y,
                args.n_evaluate_sample,
                step,
                model_name,
                True,
                task_id,
                0,
                proposal_idx,
            )
            values.append(float(v))

            if best_solution.get("value") is None or float(v) > float(best_solution.get("value") or 0.0):
                best_solution = {"y": child_y, "depth": step, "value": float(v)}

        if dfs_timeout:
            break

        k = int(getattr(args, "n_select_sample", 1) or 1)
        k = max(1, k)
        order = sorted(range(len(proposals)), key=lambda i: values[i], reverse=True)
        top_order = order[: min(k, len(order))]
        value_by_idx: dict[int, float] = {i: float(values[i]) for i in range(len(values))}

        if trace is not None:
            trace.append(
                {
                    "step": step,
                    "parents": [
                        {
                            "parent_idx": 0,
                            "parent_y": parent_y,
                            "propose_prompt": propose_prompt,
                            "propose_responses": response,
                            "proposals": [
                                {
                                    "proposal_idx": i,
                                    "proposal": p,
                                    "value": value_by_idx.get(i),
                                    "selected": bool(i in set(top_order)),
                                }
                                for i, p in enumerate(proposals)
                            ],
                        }
                    ],
                }
            )

        next_children = [proposals[i] for i in top_order]
        next_children = [y for y in next_children if not _task_is_dead_end(task, y)]

        for child_y in reversed(next_children):
            stack.append((depth + 1, child_y))

    log_copy = call_log.copy()
    log_copy["total_filtered"] = 0
    log_copy["step_times"] = {}
    log_copy["expansions_used"] = expansions_used
    log_copy["solutions_found"] = solutions_found
    log_copy["dfs_stats"] = {
        "time_limit_s": time_limit_s,
        "value_threshold": value_threshold,
        "timeout": dfs_timeout,
        "qualified_solved": bool(solved_y),
        "qualified_solved_value": solved_value,
        "best_solution": best_solution,
    }
    if trace is not None:
        log_copy["trace"] = trace

    if solved_y:
        return [solved_y], log_copy
    if best_solution["y"]:
        return [best_solution["y"]], log_copy
    return ([stack[-1][1]] if stack else []), log_copy


def run_pipelined_beam(task, x, step0_candidates, args, model_name="qwen3-32b", task_id=None):
    reset_call_log()

    node_lock = threading.Lock()
    next_node_id = 0

    best_solution = {"y": "", "depth": None, "node_id": None}

    def _alloc_node_id() -> int:
        nonlocal next_node_id
        with node_lock:
            nid = next_node_id
            next_node_id += 1
            return nid

    def _is_solved(y: str) -> bool:
        if not y:
            return False
        # Kept for backward compatibility; actual solved logic is checked via task in callers.
        return ("left: 24" in y.lower()) or y.strip().endswith("24")

    def _record_solution(y: str, depth: int, node_id: int) -> None:
        if best_solution["depth"] is None:
            best_solution["y"] = y
            best_solution["depth"] = depth
            best_solution["node_id"] = node_id

    def _make_log_copy() -> dict[str, Any]:
        log_copy = call_log.copy()
        log_copy["total_filtered"] = 0
        log_copy["step_times"] = {}
        return log_copy

    def _do_propose(parent_node: dict, step: int, proposals_override: list[str] | None = None) -> list[str]:
        if proposals_override is not None:
            return proposals_override
        parent_y = parent_node["y"]
        parent_id_hdr = parent_node["parent_id_hdr"]
        propose_prompt = _task_propose_prompt(task, x, parent_y, step)
        extra_headers = {
            "X-TOT-Task-Id": str(task_id) if task_id is not None else "",
            "X-TOT-Depth": str(step),
            "X-TOT-Parent-Idx": parent_id_hdr,
            "X-TOT-Branch-Idx": parent_id_hdr,
            "X-TOT-Call-Type": "propose",
        }
        with request_extra_headers(extra_headers):
            response = models.gpt(
                propose_prompt,
                n=max(1, getattr(args, "n_propose_sample", 1)),
                stop=None,
                model=model_name,
            )
        log_call("propose", propose_prompt, response, step, parent_id_hdr)
        proposals = _task_parse_proposals_from_responses(task, x, parent_y, response, step)
        if args.max_children_per_parent and len(proposals) > args.max_children_per_parent:
            proposals = proposals[: args.max_children_per_parent]
        return proposals

    beam_width = args.n_select_sample
    prefetch_k = getattr(args, "prefetch_k", 1)

    frontier: list[dict] = []
    for y in step0_candidates:
        nid = _alloc_node_id()
        frontier.append(
            {
                "node_id": nid,
                "depth": 0,
                "y": y,
                "parent_id_hdr": str(nid),
                "prefetched_proposals": None,
                "score": None,
            }
        )

    propose_pool = ThreadPoolExecutor(max_workers=max(1, args.propose_concurrency))
    value_pool = ThreadPoolExecutor(max_workers=max(1, args.value_concurrency))

    try:
        for step in range(1, task.steps):
            if not frontier:
                break

            # B方案：propose完成一个立即eval，不等所有propose完成
            propose_futs = []
            for parent_node in frontier:
                prefetched = parent_node.get("prefetched_proposals")
                parent_node["prefetched_proposals"] = None
                propose_futs.append(
                    (parent_node, propose_pool.submit(_do_propose, parent_node, step, prefetched))
                )

            all_children: list[dict] = []
            child_values: dict[int, float] = {}
            value_futs = []
            
            # 使用as_completed逐个处理propose结果，不等所有完成
            fut_to_parent = {pfut: parent_node for parent_node, pfut in propose_futs}
            for pfut in as_completed(fut_to_parent):
                parent_node = fut_to_parent[pfut]
                proposals = pfut.result()
                parent_id_hdr = parent_node["parent_id_hdr"]
                
                # 立即为这个父节点的子节点创建eval任务
                for proposal_idx, child_y in enumerate(proposals):
                    child_node_id = _alloc_node_id()
                    child_node = {
                        "node_id": child_node_id,
                        "depth": step,
                        "y": child_y,
                        "parent_id_hdr": str(child_node_id),
                        "prefetched_proposals": None,
                        "score": None,
                    }
                    all_children.append(child_node)

                    if _task_is_solved(task, task_id - 1 if task_id is not None else 0, child_y):
                        _record_solution(child_y, step, child_node_id)
                        if args.early_stop:
                            return [best_solution["y"]], _make_log_copy()
                        continue

                    # 立即提交value评估，不等其他propose完成
                    value_futs.append(
                        (
                            child_node,
                            value_pool.submit(
                                get_value,
                                task,
                                x,
                                child_y,
                                args.n_evaluate_sample,
                                step,
                                model_name,
                                True,
                                task_id,
                                parent_id_hdr,
                                proposal_idx,
                            ),
                        )
                    )

            pipelined_next_futs: dict[int, Any] = {}
            pipelined_started = False
            pipelined_step = step + 1

            if step < task.steps - 1:
                fut_to_child = {vfut: child_node for child_node, vfut in value_futs}
                for vfut in as_completed(fut_to_child):
                    child_node = fut_to_child[vfut]
                    v = vfut.result()
                    child_node["score"] = v
                    child_values[child_node["node_id"]] = v

                    if args.early_stop and _task_is_solved(task, task_id - 1 if task_id is not None else 0, child_node["y"]):
                        _record_solution(child_node["y"], step, child_node["node_id"])
                        return [best_solution["y"]], _make_log_copy()

                    if (
                        (not pipelined_started)
                        and prefetch_k
                        and len(child_values) >= beam_width
                    ):
                        pipelined_started = True
                        top_children = sorted(
                            (c for c in all_children if c.get("score") is not None),
                            key=lambda n: n["score"],
                            reverse=True,
                        )[: min(beam_width, prefetch_k)]
                        for cand in top_children:
                            pipelined_next_futs[cand["node_id"]] = propose_pool.submit(
                                _do_propose,
                                cand,
                                pipelined_step,
                                None,
                            )
            else:
                fut_to_child = {vfut: child_node for child_node, vfut in value_futs}
                for vfut in as_completed(fut_to_child):
                    child_node = fut_to_child[vfut]
                    v = vfut.result()
                    child_node["score"] = v
                    child_values[child_node["node_id"]] = v

            scored_children = [c for c in all_children if c.get("score") is not None]
            if not scored_children:
                break

            scored_children.sort(key=lambda n: n["score"], reverse=True)
            frontier = scored_children[:beam_width]

            if step < task.steps - 1 and pipelined_next_futs:
                selected_ids = {n["node_id"] for n in frontier}
                for node in frontier:
                    fut = pipelined_next_futs.get(node["node_id"])
                    if fut is not None and node["node_id"] in selected_ids:
                        try:
                            node["prefetched_proposals"] = fut.result()
                        except Exception:
                            node["prefetched_proposals"] = None

        ys = [best_solution["y"]] if best_solution["y"] else ([frontier[0]["y"]] if frontier else [])
        return ys, _make_log_copy()
    finally:
        propose_pool.shutdown(wait=False)
        value_pool.shutdown(wait=False)


def run_bestfirst(task, x, step0_candidates, args, model_name="qwen3-32b", task_id=None):
    reset_call_log()

    node_lock = threading.Lock()
    stats_lock = threading.Lock()
    stop_event = threading.Event()

    inflight_lock = threading.Lock()
    inflight_count = 0

    inflight_sem = threading.Semaphore(max(1, args.max_inflight_http))

    next_node_id = 0
    expansions = 0

    best_solution = {"y": "", "depth": None, "node_id": None}
    solutions_found = 0
    
    # Detailed event log
    event_log = []
    event_log_lock = threading.Lock()
    
    def _log_event(event_type: str, **kwargs):
        with event_log_lock:
            event_log.append({
                "timestamp": time.time(),
                "type": event_type,
                **kwargs
            })

    # Priority queues: higher score first => store negative score.
    expand_pq: queue.PriorityQueue = queue.PriorityQueue()
    eval_pq: queue.PriorityQueue = queue.PriorityQueue(maxsize=max(1, args.max_pending_eval))

    def _is_solved(y: str) -> bool:
        if not y:
            return False
        return ("left: 24" in y.lower()) or y.strip().endswith("24")

    def _alloc_node_id() -> int:
        nonlocal next_node_id
        with node_lock:
            nid = next_node_id
            next_node_id += 1
            return nid

    def _should_stop() -> bool:
        if stop_event.is_set():
            return True
        with stats_lock:
            if expansions >= args.max_expansions:
                return True
        return False

    def _record_solution(y: str, depth: int, node_id: int) -> None:
        # Keep the first solution (early stop semantics).
        nonlocal solutions_found
        solutions_found += 1
        if best_solution["depth"] is None:
            best_solution["y"] = y
            best_solution["depth"] = depth
            best_solution["node_id"] = node_id
            _log_event("solution_found", node_id=node_id, depth=depth, solution_num=solutions_found, y=y[:100])
            print(f"[SOLUTION] Found at depth={depth}, node_id={node_id}, total_solutions={solutions_found}")

    def _inflight_acquire() -> None:
        nonlocal inflight_count
        inflight_sem.acquire()
        with inflight_lock:
            inflight_count += 1

    def _inflight_release() -> None:
        nonlocal inflight_count
        with inflight_lock:
            inflight_count -= 1
        inflight_sem.release()

    def _propose(node_id: int, depth: int, y: str, score: float | None):
        # score is unused here but kept for potential future priority heuristics.
        # The depth parameter here is the parent's depth; children will be at depth+1
        propose_prompt = task.propose_prompt_wrap(x, y)
        child_depth = depth + 1
        extra_headers = {
            "X-TOT-Task-Id": str(task_id) if task_id is not None else "",
            "X-TOT-Depth": str(child_depth),
            "X-TOT-Parent-Idx": str(node_id),
            "X-TOT-Branch-Idx": "",
            "X-TOT-Call-Type": "propose",
        }

        _inflight_acquire()
        try:
            with request_extra_headers(extra_headers):
                response = models.gpt(
                    propose_prompt,
                    n=1,
                    stop=None,
                    model=model_name,
                )
        finally:
            _inflight_release()

        log_call("propose", propose_prompt, response, child_depth, node_id)
        proposals = parse_proposals(y, response[0])
        return proposals

    def _expand_worker():
        nonlocal expansions
        while not _should_stop():
            try:
                neg_score, arrival_t, node = expand_pq.get(timeout=0.5)
            except Exception:
                continue

            if _should_stop():
                expand_pq.task_done()
                break

            node_id = node["node_id"]
            depth = node["depth"]
            y = node["y"]
            score = node["score"]

            # Do not expand beyond max depth.
            # Children will be at depth+1, so stop if depth+1 >= task.steps
            if depth + 1 >= task.steps:
                expand_pq.task_done()
                continue

            with stats_lock:
                if expansions >= args.max_expansions:
                    stop_event.set()
                    expand_pq.task_done()
                    break
                expansions += 1
                current_expansions = expansions
            
            _log_event("expand_start", node_id=node_id, depth=depth, expansion_num=current_expansions, score=score)
            if current_expansions % 10 == 0:
                print(f"[EXPAND] #{current_expansions}: node_id={node_id}, depth={depth}, score={score:.3f}")

            try:
                proposals = _propose(node_id=node_id, depth=depth, y=y, score=score)
                _log_event("expand_done", node_id=node_id, depth=depth, num_proposals=len(proposals))
            except Exception as e:
                proposals = []
                _log_event("expand_error", node_id=node_id, depth=depth, error=str(e))

            # Submit eval tasks for children.
            child_depth = depth + 1
            parent_id_hdr = str(node_id)
            for proposal_idx, child_y in enumerate(proposals):
                if _should_stop():
                    break

                child_node_id = _alloc_node_id()
                if _is_solved(child_y):
                    _record_solution(child_y, child_depth, child_node_id)
                    _log_event("child_solved_before_eval", parent_id=node_id, child_id=child_node_id, depth=child_depth)
                    if args.early_stop:
                        stop_event.set()
                        break
                    # No early stop: do not enqueue solved nodes for further eval/expansion.
                    continue
                
                _log_event("child_created", parent_id=node_id, child_id=child_node_id, depth=child_depth, proposal_idx=proposal_idx)

                # Backpressure: bounded eval_pq.
                try:
                    eval_pq.put(
                        (
                            neg_score,
                            time.monotonic(),
                            {
                                "child_node_id": child_node_id,
                                "parent_node_id": node_id,
                                "parent_id_hdr": parent_id_hdr,
                                "depth": child_depth,
                                "proposal_idx": proposal_idx,
                                "y": child_y,
                            },
                        ),
                        timeout=1.0,
                    )
                except Exception:
                    # If queue is full and we can't enqueue, we stop expanding aggressively.
                    # This is a conservative fallback to avoid runaway.
                    break

            expand_pq.task_done()

    def _eval_worker():
        while not _should_stop():
            try:
                neg_parent_score, arrival_t, item = eval_pq.get(timeout=0.5)
            except Exception:
                continue

            if _should_stop():
                eval_pq.task_done()
                break

            child_node_id = item["child_node_id"]
            depth = item["depth"]
            child_y = item["y"]
            proposal_idx = item["proposal_idx"]
            parent_node_id = item["parent_node_id"]
            parent_id_hdr = item["parent_id_hdr"]

            _log_event("eval_start", child_id=child_node_id, depth=depth, parent_id=parent_node_id)
            
            _inflight_acquire()
            try:
                eval_start_time = time.time()
                value = get_value(
                    task,
                    x,
                    child_y,
                    args.n_evaluate_sample,
                    depth,
                    model_name,
                    log_it=True,
                    task_id=task_id,
                    parent_idx=parent_id_hdr,
                    branch_idx=proposal_idx,
                )
                eval_time = time.time() - eval_start_time
                _log_event("eval_done", child_id=child_node_id, depth=depth, value=value, eval_time=eval_time)
            finally:
                _inflight_release()

            # If solved, record and early stop.
            if _is_solved(child_y):
                _record_solution(child_y, depth, child_node_id)
                _log_event("child_solved_after_eval", child_id=child_node_id, depth=depth, value=value)
                if args.early_stop:
                    stop_event.set()

            if (not args.early_stop) and _is_solved(child_y):
                # No early stop: keep searching but do not expand solved nodes.
                eval_pq.task_done()
                continue

            # Push child node into expand_pq (use existing child_node_id).
            child_node = {
                "node_id": child_node_id,
                "depth": depth,
                "y": child_y,
                "score": value,
            }
            expand_pq.put((-float(value), time.monotonic(), child_node))
            _log_event("child_queued_for_expand", child_id=child_node_id, depth=depth, value=value)

            eval_pq.task_done()

    # Seed: Initialize with step0_candidates at depth=0.
    # These are already evaluated, so we put them directly into expand_pq.
    print(f"[INIT] Seeding with {len(step0_candidates)} step0 candidates")
    for idx, y0 in enumerate(step0_candidates):
        node_id = _alloc_node_id()
        # Use a high initial score to prioritize step0 candidates equally
        expand_pq.put((0.0, time.monotonic(), {"node_id": node_id, "depth": 0, "y": y0, "score": 1.0}))
        _log_event("seed_node", node_id=node_id, depth=0, seed_idx=idx, y=y0[:100])
    
    search_start_time = time.time()

    # Start workers.
    expand_threads = []
    for _ in range(max(1, args.expand_concurrency)):
        t = threading.Thread(target=_expand_worker, daemon=True)
        t.start()
        expand_threads.append(t)

    eval_threads = []
    for _ in range(max(1, args.eval_concurrency)):
        t = threading.Thread(target=_eval_worker, daemon=True)
        t.start()
        eval_threads.append(t)

    # Wait until stop.
    while not _should_stop():
        if stop_event.is_set():
            break

        # Natural termination: queues drained and no in-flight requests.
        with inflight_lock:
            inflight_now = inflight_count
        if expand_pq.empty() and eval_pq.empty() and inflight_now == 0:
            break
        time.sleep(0.2)

    stop_event.set()

    # Best-effort join.
    for t in expand_threads:
        t.join(timeout=1.0)
    for t in eval_threads:
        t.join(timeout=1.0)

    search_end_time = time.time()
    total_search_time = search_end_time - search_start_time
    
    print(f"[DONE] Search completed: expansions={expansions}, solutions_found={solutions_found}, time={total_search_time:.2f}s")
    
    ys = [best_solution["y"]] if best_solution["y"] else []
    log_copy = call_log.copy()
    log_copy["total_filtered"] = 0
    log_copy["step_times"] = {}
    with stats_lock:
        log_copy["expansions_used"] = expansions
    log_copy["solutions_found"] = solutions_found
    log_copy["total_search_time"] = total_search_time
    log_copy["event_log"] = event_log
    log_copy["best_solution"] = best_solution
    return ys, log_copy


def total_length(log_list, key):
    return sum(entry.get(key, 0) for entry in log_list)


def test_single_task(task, idx, args, log_dir):
    task_id = idx + 1
    x = task.get_input(idx)

    run_args = args
    adaptive_info = None
    if getattr(args, "adaptive_beam", 0) and args.task == "gsm8k":
        adaptive_info = _adaptive_beam_settings(args, x)
        run_args = copy.copy(args)
        run_args.n_select_sample = int(adaptive_info.get("n_select_sample", args.n_select_sample))
        if adaptive_info.get("max_expansions") is not None:
            run_args.max_expansions = int(adaptive_info["max_expansions"])

    # All modes need step0_candidates
    step0_candidates, step0_time, step0_trace = run_step0(task, x, run_args, run_args.backend, task_id=task_id)

    models.completion_tokens = 0
    models.prompt_tokens = 0

    baseline_start_time = time.time()
    if run_args.schedule_mode == "bestfirst":
        ys_baseline, log_baseline = run_bestfirst(
            task,
            x,
            step0_candidates,
            run_args,
            run_args.backend,
            task_id=task_id,
        )
    elif run_args.schedule_mode == "pipelined_beam":
        ys_baseline, log_baseline = run_pipelined_beam(
            task,
            x,
            step0_candidates,
            run_args,
            run_args.backend,
            task_id=task_id,
        )
    elif run_args.schedule_mode == "dfs":
        ys_baseline, log_baseline = run_dfs(
            task,
            x,
            step0_candidates,
            run_args,
            run_args.backend,
            task_id=task_id,
        )
    else:
        ys_baseline, log_baseline = run_baseline(
            task,
            x,
            step0_candidates,
            run_args,
            run_args.backend,
            task_id=task_id,
        )
    search_time = time.time() - baseline_start_time
    total_time = step0_time + search_time

    baseline_tokens = models.completion_tokens + models.prompt_tokens
    if run_args.schedule_mode == "dfs" and float(getattr(run_args, "dfs_value_threshold", 0.0) or 0.0) > 0:
        baseline_success = bool((log_baseline.get("dfs_stats", {}) or {}).get("qualified_solved"))
    elif ys_baseline and run_args.schedule_mode in {"interleaved", "layered"} and float(getattr(run_args, "stop_threshold", 0.0) or 0.0) > 0:
        thr = float(getattr(run_args, "stop_threshold", 0.0) or 0.0)
        step_for_value = max(0, int(getattr(task, "steps", 1)) - 1)
        baseline_success = False
        for y in ys_baseline:
            if not _task_is_solved(task, idx, y):
                continue
            v = get_value(task, x, y, run_args.n_evaluate_sample, step_for_value, run_args.backend, False, task_id, 0, 0)
            if float(v) >= thr:
                baseline_success = True
                break
    elif ys_baseline:
        baseline_success = any(_task_is_solved(task, idx, y) for y in ys_baseline)
    else:
        baseline_success = False

    result = {
        "task_id": task_id,
        "idx": idx,
        "input": x,
        "adaptive": adaptive_info,
        "step0_time": step0_time,
        "step0_trace": step0_trace,
        "baseline": {
            "output": ys_baseline[0] if ys_baseline else "",
            "success": baseline_success,
            "tokens": baseline_tokens,
            "total_time": total_time,
            "search_time": search_time,
            "total_filtered": log_baseline.get("total_filtered", 0),
            "propose_calls": len(log_baseline["propose"]),
            "value_calls": len(log_baseline["value"]),
            "propose_prompt_chars": total_length(log_baseline["propose"], "prompt_length"),
            "propose_response_chars": total_length(log_baseline["propose"], "response_length"),
            "value_prompt_chars": total_length(log_baseline["value"], "prompt_length"),
            "value_response_chars": total_length(log_baseline["value"], "response_length"),
            "step_times": log_baseline.get("step_times", {}),
            "expansions_used": log_baseline.get("expansions_used"),
            "solutions_found": log_baseline.get("solutions_found"),
            "total_search_time": log_baseline.get("total_search_time"),
            "event_log": log_baseline.get("event_log"),
            "best_solution": log_baseline.get("best_solution"),
            "trace": log_baseline.get("trace"),
        },
    }

    task_log_file = os.path.join(log_dir, f"task_{task_id}.json")
    with open(task_log_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="game24",
        choices=["game24", "crosswords", "text", "gsm8k"],
    )
    parser.add_argument("--start", type=int, default=940)
    parser.add_argument("--end", type=int, default=940)
    parser.add_argument("--backend", type=str, default="qwen3-32b-vllm")
    parser.add_argument("--n_propose_sample", type=int, default=1)
    parser.add_argument("--trace_search", type=int, default=0)
    parser.add_argument("--gsm8k_file", type=str, default="test.jsonl")
    parser.add_argument("--gsm8k_steps", type=int, default=6)
    parser.add_argument("--n_evaluate_sample", type=int, default=1)
    parser.add_argument("--n_select_sample", type=int, default=2)
    parser.add_argument("--value_concurrency", type=int, default=1)
    parser.add_argument(
        "--schedule_mode",
        type=str,
        default="interleaved",
        choices=["interleaved", "layered", "bestfirst", "pipelined_beam", "dfs"],
    )
    parser.add_argument("--propose_concurrency", type=int, default=1)
    parser.add_argument("--max_children_per_parent", type=int, default=24)

    parser.add_argument("--dfs_time_limit_s", type=float, default=0.0)
    parser.add_argument("--dfs_value_threshold", type=float, default=0.0)

    parser.add_argument("--stop_threshold", type=float, default=0.0)

    # Best-first (Scheme B) params.
    parser.add_argument("--expand_concurrency", type=int, default=4)
    parser.add_argument("--eval_concurrency", type=int, default=32)
    parser.add_argument("--max_pending_eval", type=int, default=256)
    parser.add_argument("--max_inflight_http", type=int, default=64)
    parser.add_argument("--early_stop", type=int, default=1)
    parser.add_argument("--max_expansions", type=int, default=500)
    parser.add_argument("--prefetch_k", type=int, default=1)

    # GSM8K adaptive width
    parser.add_argument("--adaptive_beam", type=int, default=0)
    parser.add_argument("--adaptive_beam_map", type=str, default="")
    parser.add_argument("--adaptive_max_expansions_map", type=str, default="")
    args = parser.parse_args()

    args.early_stop = bool(args.early_stop)

    if args.task == "gsm8k":
        from tot.tasks.gsm8k import GSM8KTask

        task = GSM8KTask(file=args.gsm8k_file, steps=args.gsm8k_steps)
    else:
        task = get_task(args.task)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = "dfs" if args.schedule_mode == "dfs" else "baseline_tot"
    log_dir = f"logs/{args.task}/{run_tag}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)

    print(f"logs_dir: {log_dir}")

    all_results = []
    for task_id in range(args.start, args.end + 1):
        idx = task_id - 1
        input_str = task.get_input(idx)
        print(f"\nTask {task_id} (idx={idx}): {input_str}")

        try:
            result = test_single_task(task, idx, args, log_dir)
            all_results.append(result)
            b = result["baseline"]
            step0_time = result.get("step0_time", 0.0)
            search_time = b.get("search_time", 0.0)
            total_time = b.get("total_time", 0.0)
            print(
                f"Result: total={total_time:.3f}s (step0={step0_time:.3f}s, search={search_time:.3f}s), "
                f"tokens={b['tokens']}, filtered={b['total_filtered']}, "
                f"expansions={b.get('expansions_used', 'N/A')}, success={'✓' if b['success'] else '✗'}"
            )
        except Exception as e:
            print(f"Task {task_id} failed: {e}")
            import traceback
            traceback.print_exc()

    summary_file = os.path.join(log_dir, "summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(
            {"timestamp": datetime.now().isoformat(), "results": all_results},
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"summary: {summary_file}")


if __name__ == "__main__":
    main()