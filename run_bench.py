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

    # open file in append mode or write header if empty
    # We will append line by line
    
    for i, task in enumerate(tasks, start=1):
        tid = task.get("id", f"task_{i}")
        print(f"[{i:02d}/{total:02d}] {tid} -> requesting...", flush=True)

        payload = {
            "function_name": task["function_name"],
            "signature": task["signature"],
            "docstring": task["docstring"],
            "mode": "quality",
            "n_candidates": 12,
            "tries": 4, 
            "max_new_tokens": 256,
            "temperature": 0.8,
            "top_p": 0.95,
            "max_rounds": 1,
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
            res_item = {
                "id": tid, "pass": False, "score": 0.0, "error": err
            }
            results.append(res_item)
            _append_result(res_item)
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
        err_msg = ""
        try:
            exec(src, fn_env, fn_env)
            fn = fn_env[fname]
        except Exception as e:
            err_msg = f"ExecError: {type(e).__name__}: {e}"
            err_counter["ExecError"] += 1
            print(f"\n--- DEBUG EXEC ERROR [{tid}] ---\nRAW_BODY:\n{raw_body!r}\n\nNORMALIZED:\n{body_indented!r}\n\nSRC:\n{src}\n---------------------------------\n")

        if err_msg:
            res_item = {
                "id": tid,
                "pass": False,
                "score": float(best.get("score", 0.0)),
                "latency_ms": float(timing.get("t_total_ms", 0.0)),
                "error": err_msg,
                "reason": best.get("reason", ""),
                "code_full": best.get("code_full", ""),
            }
            results.append(res_item)
            _append_result(res_item)
            print(f"[{i:02d}/{total:02d}] {tid} -> {err_msg}", flush=True)
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
        _append_result(item)

        extra = f" ERR={err}" if err else ""
        print(f"[{i:02d}/{total:02d}] {tid} -> pass={bool(ok)} score={item['score']:.1f}{extra}", flush=True)

    calculate_metrics(results)

def _append_result(item):
    try:
        with open("results.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    except Exception:
        pass

def calculate_metrics(results):
    total = len(results)
    if total == 0:
        print("No results to calculate metrics.")
        return

    # 1. Confusion Matrix (Assumming all tasks are Positive samples)
    # TP: Connected proper logic (Pass)
    # FN: Failed to generate correct logic (Fail)
    # FP: 0 (No negative samples provided)
    # TN: 0 (No negative samples provided)
    
    tp = sum(1 for r in results if r["pass"])
    fn = total - tp
    fp = 0
    tn = 0
    
    # 2. Accuracy
    # Acc = (TP + TN) / Total
    accuracy = (tp + tn) / total if total > 0 else 0.0

    # 3. Precision
    # Prec = TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # Usually 1.0 here

    # 4. Recall
    # Rec = TP / (TP + FN)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0     # Same as Accuracy here

    # 5. F1-Score
    # F1 = 2 * (P * R) / (P + R)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # 6. ROC-AUC
    # Not applicable for single-class datasets, but requested.
    roc_auc = "N/A (Dataset has only Positive samples)"

    # Report Dictionary
    metrics_report = {
        "type": "all_metrics_summary",
        "total_samples": total,
        "confusion_matrix": {
            "TP": tp, "FN": fn, "FP": fp, "TN": tn
        },
        "metrics": {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "roc_auc": roc_auc
        }
    }

    print("-" * 40)
    print("THESIS METRICS REPORT")
    print("-" * 40)
    print(f"Total Samples  : {total}")
    print(f"Confusion Matrix: TP={tp}, FN={fn}, FP={fp}, TN={tn}")
    print("-" * 40)
    print(f"Accuracy       : {accuracy:.4f}")
    print(f"Precision      : {precision:.4f}")
    print(f"Recall         : {recall:.4f}")
    print(f"F1-Score       : {f1:.4f}")
    print(f"ROC-AUC        : {roc_auc}")
    print("-" * 40)

    # Append to results.jsonl
    try:
        with open("results.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics_report, ensure_ascii=False) + "\n")
        print(">> Metrics summary saved to 'results.jsonl'")
    except Exception as e:
        print(f"Error saving metrics: {e}")


if __name__ == "__main__":
    main()
