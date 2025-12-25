import json
import requests
import importlib
import threading
from collections import Counter

API = "http://localhost:8000/v1/solve"
checkers = importlib.import_module("checkers")


def normalize_body(code_body: str) -> str:
    """
    Preserve relative indentation structure.
    - remove fences
    - trim blank lines
    - remove common leading indent (min indent among non-empty lines)
    - then indent whole block by 4 spaces (function body)
    """
    s = (code_body or "").replace("```python", "").replace("```", "")
    lines = s.splitlines()

    while lines and lines[0].strip() == "":
        lines.pop(0)
    while lines and lines[-1].strip() == "":
        lines.pop()

    if not lines:
        return "    pass"

    indents = []
    for ln in lines:
        if ln.strip():
            indents.append(len(ln) - len(ln.lstrip(" \t")))
    min_indent = min(indents) if indents else 0

    norm = []
    for ln in lines:
        if ln.strip() == "":
            norm.append("")
        else:
            norm.append(ln[min_indent:].rstrip())

    body = "\n".join(("    " + ln) if ln != "" else "" for ln in norm)
    return body if body.strip() else "    pass"


def build_function_source(fname: str, sig: str, body_indented: str) -> str:
    return f"def {fname}{sig}:\n{body_indented}\n"


def run_with_timeout(fn, checker_fn, timeout_s: float = 1.0):
    """
    Windows-friendly checker timeout.
    Returns (ok: bool, err: str)
    """
    result = {"ok": False, "err": ""}

    def target():
        try:
            result["ok"] = bool(checker_fn(fn))
            result["err"] = ""
        except Exception as e:
            result["ok"] = False
            result["err"] = f"{type(e).__name__}: {e}"

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout_s)

    if t.is_alive():
        return False, "Timeout"
    return result["ok"], result["err"]


def main():
    results = []
    err_counter = Counter()

    with open("bench.jsonl", "r", encoding="utf-8") as f:
        tasks = [json.loads(line) for line in f if line.strip()]

    total = len(tasks)

    for i, task in enumerate(tasks, start=1):
        tid = task.get("id", f"task_{i}")
        print(f"[{i:02d}/{total:02d}] {tid} -> requesting...", flush=True)

        payload = {
            "function_name": task["function_name"],
            "signature": task["signature"],
            "docstring": task["docstring"],
            "mode": "quality",

            # override quality (nếu server cho phép)
            "n_candidates": 24,
            "tries": 2,
            "max_new_tokens": 220,
            "temperature": 0.75,
            "top_p": 0.92,
            "max_rounds": 3,
            "accept_score": 90.0,
        }

        # --- call API ---
        try:
            resp = requests.post(API, json=payload, timeout=600)
            resp.raise_for_status()
            r = resp.json()
        except Exception as e:
            err = f"APIError: {type(e).__name__}: {e}"
            err_counter["APIError"] += 1
            results.append({
                "id": tid,
                "pass": False,
                "score": 0.0,
                "latency_ms": 0.0,
                "error": err,
            })
            print(f"[{i:02d}/{total:02d}] {tid} -> {err}", flush=True)
            continue

        print(f"[{i:02d}/{total:02d}] {tid} -> got response", flush=True)

        best = r.get("best") or {}
        timing = r.get("timing") or {}
        metrics = (best.get("metrics") or {})

        fname = task["function_name"]
        sig = task["signature"]

        raw_body = best.get("code_body") or ""
        body_indented = normalize_body(raw_body)

        src = build_function_source(fname, sig, body_indented)
        fn_env = {}

        # --- exec ---
        try:
            exec(src, fn_env, fn_env)
            fn = fn_env[fname]
        except Exception as e:
            err = f"ExecError: {type(e).__name__}: {e}"
            err_counter["ExecError"] += 1
            results.append({
                "id": tid,
                "pass": False,
                "score": float(best.get("score", 0.0)),
                "latency_ms": float(timing.get("t_total_ms", 0.0)),
                "error": err,
                "reason": best.get("reason", ""),
                "determinism": float(metrics.get("determinism", 0.0)),
                "passed": int(metrics.get("passed", 0)),
                "total": int(metrics.get("total", 0)),
                "code_full": best.get("code_full", ""),
            })
            print(f"[{i:02d}/{total:02d}] {tid} -> {err}", flush=True)
            continue

        # --- checker with timeout ---
        checker_name = task["checker"]
        checker_fn = getattr(checkers, checker_name)

        ok, err = run_with_timeout(fn, checker_fn, timeout_s=1.0)
        if err:
            err_counter[err.split(":")[0]] += 1

        item = {
            "id": tid,
            "pass": bool(ok),
            "score": float(best.get("score", 0.0)),
            "latency_ms": float(timing.get("t_total_ms", 0.0)),
            "error": err,
            "reason": best.get("reason", ""),
            "determinism": float(metrics.get("determinism", 0.0)),
            "passed": int(metrics.get("passed", 0)),
            "total": int(metrics.get("total", 0)),
        }

        # nếu fail thì lưu code_full để debug (optional)
        if not ok:
            item["code_full"] = best.get("code_full", "")

        results.append(item)

        extra = f" ERR={err}" if err else ""
        print(f"[{i:02d}/{total:02d}] {tid} -> pass={bool(ok)} score={item['score']:.1f}{extra}", flush=True)

    with open("results.jsonl", "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    acc = sum(1 for x in results if x["pass"]) / max(1, len(results))
    print("Accuracy:", acc)

    if err_counter:
        print("Error breakdown:")
        for k, v in err_counter.most_common():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
