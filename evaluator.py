import __future__

import dataclasses
import inspect
import multiprocessing as mp
import time
import traceback
import types
from collections.abc import Iterator
from typing import Any, Dict, List, Optional, Tuple, get_args, get_origin, Union

_INDENT = "    "

@dataclasses.dataclass(frozen=True)
class EvalConfig:
    timeout_s: float = 3.0
    repeats: int = 2
    probe_cases: int = 6
    max_code_chars: int = 1800
    mp_start_method: str = "spawn"
    pool_workers: int = 1

# -------- heuristics (generic) --------
_MAX_CONTAINER_LEN = 10_000
_HUGE_NUMBER_ABS = 10**18  # penalty threshold (generic, not correctness)
_HUGE_NUMBER_PENALTY = 20.0

# -----------------------
# normalize / indent
# -----------------------

def _normalize_block(code: str) -> str:
    if not code:
        return ""
    code = code.replace("```python", "").replace("```", "")
    lines = code.splitlines()
    while lines and lines[0].strip() == "":
        lines.pop(0)
    while lines and lines[-1].strip() == "":
        lines.pop()
    return "\n".join(lines)

def _indent_for_function(block: str) -> str:
    block = _normalize_block(block)
    if not block.strip():
        return _INDENT + "pass"

    raw = block.splitlines()

    indents = []
    for ln in raw:
        if ln.strip():
            indents.append(len(ln) - len(ln.lstrip(" \t")))
    min_indent = min(indents) if indents else 0

    lines = []
    for ln in raw:
        if ln.strip() == "":
            lines.append("")
        else:
            ln2 = ln[min_indent:] if len(ln) >= min_indent else ln
            lines.append(_INDENT + ln2.rstrip())

    fixed: List[str] = []
    i = 0
    while i < len(lines):
        cur = lines[i]
        fixed.append(cur)
        if cur.rstrip().endswith(":"):
            j = i + 1
            while j < len(lines) and lines[j].strip() == "":
                fixed.append(lines[j]); j += 1
            if j < len(lines):
                nxt = lines[j]
                cur_indent = len(cur) - len(cur.lstrip(" "))
                nxt_indent = len(nxt) - len(nxt.lstrip(" "))
                if nxt_indent <= cur_indent:
                    fixed.append(_INDENT + nxt)
                    i = j + 1
                    continue
        i += 1

    out = "\n".join(fixed).rstrip()
    return out if out.strip() else (_INDENT + "pass")

def build_function_code(func_name: str, signature: str, docstring: str, body: str) -> str:
    header = f"def {func_name}{signature}:\n"
    ds = f'{_INDENT}"""{docstring}"""\n' if docstring else ""
    return header + ds + _indent_for_function(body) + "\n"


# -----------------------
# probe cases (generic)
# -----------------------

def _is_optional_annotation(origin: Any, args: tuple[Any, ...]) -> bool:
    """
    Optional[T] == Union[T, NoneType]
    Supports both typing.Union and types.UnionType (T | None).
    """
    if not args:
        return False

    none_in_args = any(a is type(None) for a in args)  # noqa: E721
    if not none_in_args:
        return False

    # typing.Union or built-in union (|)
    if origin is Union:
        return True
    if origin is getattr(types, "UnionType", None):
        return True

    return False

def _simple_value_for_annotation(ann: Any) -> Any:
    if ann is inspect._empty:
        return 0

    origin = get_origin(ann)
    args = get_args(ann)

    if _is_optional_annotation(origin, args):
        return None

    if origin is list:
        inner = args[0] if args else inspect._empty
        return [_simple_value_for_annotation(inner)]
    if origin is dict:
        k = args[0] if len(args) > 0 else inspect._empty
        v = args[1] if len(args) > 1 else inspect._empty
        return {_simple_value_for_annotation(k): _simple_value_for_annotation(v)}
    if origin is tuple:
        if not args:
            return (0,)
        return tuple(_simple_value_for_annotation(a) for a in args)

    if ann is int: return 1
    if ann is float: return 1.0
    if ann is str: return "x"
    if ann is bool: return True
    return 0

def _variants_for_value(v: Any) -> List[Any]:
    if isinstance(v, int):   return [-3, 0, 1, 2, 5, 50]
    if isinstance(v, float): return [-1.0, 0.0, 1.0, 2.0, 10.0, -10.0][:6]
    if isinstance(v, str):   return ["", "x", "abc", "0", " ", "xyz"][:6]
    if isinstance(v, bool):  return [True, False, True, False, True, False][:6]
    if v is None:            return [None] * 6
    return [v] * 6

def _parse_signature(signature: str):
    stub = f"def __f__{signature}:\n    pass\n"
    g: Dict[str, Any] = {}
    l: Dict[str, Any] = {}
    exec(stub, g, l)
    return inspect.signature(l["__f__"])

def build_probe_cases(signature: str, k: int) -> List[Tuple[List[Any], Dict[str, Any]]]:
    sig = _parse_signature(signature)
    base_args: List[Any] = []
    base_kwargs: Dict[str, Any] = {}

    for _, p in sig.parameters.items():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        val = p.default if p.default is not inspect._empty else _simple_value_for_annotation(p.annotation)
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
            base_args.append(val)
        elif p.kind is p.KEYWORD_ONLY:
            base_kwargs[p.name] = val

    if not base_args and not base_kwargs:
        return [([], {}) for _ in range(k)]

    cases: List[Tuple[List[Any], Dict[str, Any]]] = []
    if base_args:
        for vv in _variants_for_value(base_args[0])[:k]:
            a2 = list(base_args); a2[0] = vv
            cases.append((a2, dict(base_kwargs)))
    else:
        cases = [(list(base_args), dict(base_kwargs)) for _ in range(k)]

    return cases[:k]

# -----------------------
# sanity checks (generic)
# -----------------------

def _sanity_check_output(out: Any) -> float:
    """
    Returns penalty (>=0). Raises to mark probe as failed.
    """
    if out is None:
        raise TypeError("Returned None")

    if isinstance(out, (types.GeneratorType, Iterator)):
        raise TypeError("Returned generator/iterator (yield not allowed).")

    if isinstance(out, (list, tuple, dict, set)) and len(out) > _MAX_CONTAINER_LEN:
        raise TypeError("Returned huge container")

    penalty = 0.0
    if isinstance(out, (int, float)) and abs(out) > _HUGE_NUMBER_ABS:
        penalty += _HUGE_NUMBER_PENALTY  # not reject, just penalize

    return penalty

# -----------------------
# worker (runs INSIDE pool)
# -----------------------

def _worker_eval(func_code: str, func_name: str, probes: List[Tuple[List[Any], Dict[str, Any]]], repeats: int):
    t0 = time.perf_counter()
    penalty = 0.0
    try:
        compiled = compile(
            func_code, "<candidate>", "exec",
            flags=__future__.annotations.compiler_flag,
            dont_inherit=True
        )

        safe_builtins = {
            "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict, "enumerate": enumerate,
            "float": float, "int": int, "len": len, "list": list, "max": max, "min": min, "range": range,
            "reversed": reversed, "set": set, "sorted": sorted, "str": str, "sum": sum, "tuple": tuple, "zip": zip,
            "map": map, "filter": filter,  # ✅ add (common in generated code)
        }
        env: Dict[str, Any] = {"__builtins__": safe_builtins}
        exec(compiled, env, env)

        fn = env.get(func_name)
        if not callable(fn):
            dt = (time.perf_counter() - t0) * 1000.0
            return {
                "compile_ok": True, "has_func": False,
                "passed": 0, "total": len(probes),
                "outs": [], "probe_outs": [],
                "elapsed_ms": dt, "penalty": 0.0
            }

        if inspect.isgeneratorfunction(fn):
            dt = (time.perf_counter() - t0) * 1000.0
            return {
                "compile_ok": True, "has_func": True,
                "passed": 0, "total": len(probes),
                "outs": ["GENERATOR_FUNCTION"], "probe_outs": [],
                "elapsed_ms": dt, "penalty": 0.0
            }

        passed = 0
        probe_outs: List[str] = []
        for args, kwargs in probes:
            try:
                out = fn(*args, **kwargs)
                penalty += _sanity_check_output(out)
                probe_outs.append(repr(out))
                passed += 1
            except Exception:
                probe_outs.append("EXC")

        outs: List[str] = []
        if probes:
            a0, k0 = probes[0]
            for _ in range(repeats):
                try:
                    out0 = fn(*a0, **k0)
                    penalty += _sanity_check_output(out0)
                    outs.append(repr(out0))
                except Exception as e:
                    outs.append(f"EXC:{type(e).__name__}")

        dt = (time.perf_counter() - t0) * 1000.0
        return {
            "compile_ok": True, "has_func": True,
            "passed": passed, "total": len(probes),
            "outs": outs, "probe_outs": probe_outs,
            "elapsed_ms": dt, "penalty": float(penalty)
        }

    except Exception:
        dt = (time.perf_counter() - t0) * 1000.0
        return {
            "compile_ok": False,
            "has_func": False,
            "passed": 0,
            "total": len(probes),
            "outs": [],
            "probe_outs": [],
            "elapsed_ms": dt,
            "penalty": 0.0,
            "tb": traceback.format_exc(limit=6),
        }

# -----------------------
# Evaluator (POOL)
# -----------------------

class Evaluator:
    def __init__(self, cfg: EvalConfig):
        self.cfg = cfg
        ctx = mp.get_context(cfg.mp_start_method)
        self.pool = ctx.Pool(processes=cfg.pool_workers, maxtasksperchild=200)

    def close(self):
        try:
            self.pool.close()
            self.pool.join()
        except Exception:
            pass

    def evaluate_candidate(self, body: str, func_name: str, signature: str, docstring: str) -> Dict[str, Any]:
        code = build_function_code(func_name, signature, docstring, body)
        length_chars = len(code)

        try:
            compile(
                code, "<candidate>", "exec",
                flags=__future__.annotations.compiler_flag,
                dont_inherit=True
            )
        except Exception:
            return {"compile_ok": False, "passed": 0, "total": 0, "score": 0.0, "reason": "compile_fail",
                    "determinism": 0.0, "avg_ms": 0.0, "length_chars": length_chars, "code": code}

        probes = build_probe_cases(signature, self.cfg.probe_cases)
        ar = self.pool.apply_async(_worker_eval, (code, func_name, probes, self.cfg.repeats))

        try:
            r = ar.get(timeout=self.cfg.timeout_s)
        except mp.TimeoutError:
            return {"compile_ok": True, "passed": 0, "total": len(probes), "score": 0.0, "reason": "timeout",
                    "determinism": 0.0, "avg_ms": self.cfg.timeout_s * 1000.0, "length_chars": length_chars, "code": code}

        if not r.get("compile_ok", False):
            return {"compile_ok": False, "passed": 0, "total": len(probes), "score": 0.0, "reason": "compile_fail",
                    "determinism": 0.0, "avg_ms": float(r.get("elapsed_ms", 0.0)), "length_chars": length_chars, "code": code}

        if not r.get("has_func", True):
            return {"compile_ok": True, "passed": 0, "total": len(probes), "score": 5.0, "reason": "missing_func",
                    "determinism": 0.0, "avg_ms": float(r.get("elapsed_ms", 0.0)), "length_chars": length_chars, "code": code}

        passed = int(r.get("passed", 0))
        total = int(r.get("total", len(probes)))
        outs = r.get("outs", [])
        penalty = float(r.get("penalty", 0.0))
        probe_outs = r.get("probe_outs", [])

        determinism = 0.0
        if len(outs) >= 2:
            determinism = 1.0 if all(o == outs[0] for o in outs[1:]) else 0.0
        elif len(outs) == 1:
            determinism = 0.5

        score = 20.0 + 70.0 * (passed / total if total else 0.0) + 10.0 * determinism

        if length_chars > self.cfg.max_code_chars:
            score -= min(20.0, (length_chars - self.cfg.max_code_chars) / 200.0)

        score -= penalty

        # constant/diversity penalty (generic)
        ok_probe_outs = [x for x in probe_outs if x != "EXC"]
        constant_penalty = 0.0
        diversity = 0.0
        if ok_probe_outs:
            diversity = len(set(ok_probe_outs)) / len(ok_probe_outs)
            if len(ok_probe_outs) >= max(2, total // 2) and len(set(ok_probe_outs)) <= 1:
                constant_penalty = 35.0
            elif diversity < 0.34:
                constant_penalty = 15.0

        score -= constant_penalty
        score = max(0.0, min(100.0, score))

        reason = "ok" if passed == total and determinism >= 1.0 else ("partial_runtime" if passed < total else "nondeterministic")
        if penalty > 0 and reason == "ok":
            reason = "ok_but_suspicious"
        if constant_penalty > 0 and reason == "ok":
            reason = "ok_but_constant"

        return {
            "compile_ok": True,
            "passed": passed,
            "total": total,
            "score": float(score),
            "reason": reason,
            "determinism": float(determinism),
            "avg_ms": float(r.get("elapsed_ms", 0.0)),
            "length_chars": length_chars,
            "code": code,
            "diversity": float(diversity),
            "penalty": float(penalty),
        }

    def select_best_generic(self, candidates: List[str], func_name: str, signature: str, docstring: str):
        rows = []
        best = None
        for i, body in enumerate(candidates):
            rr = self.evaluate_candidate(body, func_name, signature, docstring)
            row = {"idx": i, "body": body, **rr}
            rows.append(row)
            if best is None or row["score"] > best["score"]:
                best = row
        rows.sort(key=lambda x: x["score"], reverse=True)
        return best, rows
