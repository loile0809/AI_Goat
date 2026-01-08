from __future__ import annotations
print(f"DEBUG: SERVER LOADING name={__name__}")

import os
import time
import uuid
from fastapi import FastAPI
import uvicorn

from api_schema import (
    SolveRequest, SolveResponse, WarningItem,
    CandidateMetrics, LeaderboardItem, BestItem
)
from model_loader import load_codet5
from generator import generate_function_bodies
from evaluator import EvalConfig, Evaluator

app = FastAPI(title="CodeT5 Function Body API", version="1.0")

CHECKPOINT = os.getenv("CODET5_CHECKPOINT", "Salesforce/codet5p-770m-py")

# Globals (lazy init to avoid multiprocessing recursive load)
tokenizer = None
model = None
device = None
quality_evaluator = None
fast_evaluator = None

# Persistent evaluators (production)
# Use spawn-safe configuration
QUALITY_EVAL_CFG = EvalConfig(timeout_s=3.0, repeats=2, probe_cases=8, pool_workers=1)
FAST_EVAL_CFG    = EvalConfig(timeout_s=1.5, repeats=1, probe_cases=6, pool_workers=1)

@app.on_event("startup")
def _startup():
    global tokenizer, model, device, quality_evaluator, fast_evaluator
    
    t0 = time.perf_counter()
    print("[BOOT] loading model...")
    tokenizer, model, device = load_codet5(CHECKPOINT)
    print(f"[BOOT] model loaded in {(time.perf_counter() - t0):.1f}s on {device}")

    quality_evaluator = Evaluator(QUALITY_EVAL_CFG)
    fast_evaluator = Evaluator(FAST_EVAL_CFG)

@app.on_event("shutdown")
def _shutdown():
    if fast_evaluator: fast_evaluator.close()
    if quality_evaluator: quality_evaluator.close()

# ---- Presets for VS Code extension ----
PRESET_FAST = dict(
    n_candidates=6, tries=1, max_new_tokens=120,
    temperature=0.8, top_p=0.92,
    timeout_s=1.5, repeats=1, probe_cases=6,
    accept_score=0.0, max_rounds=1,
)

PRESET_QUALITY = dict(
    n_candidates=12, tries=1, max_new_tokens=200,
    temperature=0.75, top_p=0.92,
    timeout_s=3.0, repeats=2, probe_cases=8,
    accept_score=90.0, max_rounds=2,
)

def _prompt(req: SolveRequest, extra: str = "") -> str:
    return f"""# Python 3
# TASK: Write ONLY the function body (no def, no imports, no tests, no I/O).
# RULES:
# - Pure function: deterministic, no side effects.
# - No print/input/open/import/os/subprocess/time/random.
# - No yield. No generator.
# - Must return a value on all paths (never None).
# - Output should depend on inputs (avoid constant return).
# - Keep it short and avoid huge intermediate containers.
{extra}
def {req.function_name}{req.signature}:
    \"\"\"{req.docstring}\"\"\"
"""

def _feedback(best: dict) -> str:
    score = float(best.get("score", 0.0))
    reason = best.get("reason", "unknown")
    passed = int(best.get("passed", 0))
    total = int(best.get("total", 0))
    return (
        f"# Previous attempt did not meet target: reason={reason} score={score:.1f} tests={passed}/{total}\n"
        f"# Fix runtime errors/timeouts. Ensure returns a value for all paths.\n"
        f"# Keep deterministic and avoid constant return.\n"
    )

def _pool_warnings(best: dict | None, top: list[dict], timeout_rate: float) -> list[WarningItem]:
    ws: list[WarningItem] = []
    if not best:
        ws.append(WarningItem(code="GENERATION_EMPTY", message="No valid candidates produced."))
        return ws

    best_score = float(best.get("score", 0.0))
    best_reason = best.get("reason", "compile_fail")

    if best_score < 60.0:
        ws.append(WarningItem(code="BEST_SCORE_LOW", message="Best score is low; result may be incorrect."))
    if best_reason == "nondeterministic":
        ws.append(WarningItem(code="BEST_NONDETERMINISTIC", message="Best candidate is not deterministic."))
    if best_reason in ("ok_but_constant",):
        ws.append(WarningItem(code="BEST_MAY_BE_CONSTANT", message="Best candidate looks constant-like; verify behavior."))

    if top:
        diversity_pool = sum(float(x.get("diversity", 0.0)) for x in top) / max(1, len(top))
        if diversity_pool < 0.25:
            ws.append(WarningItem(code="LOW_DIVERSITY_POOL", message="Candidate outputs lack diversity; consider more tries or higher temperature."))

    if timeout_rate > 0.30:
        ws.append(WarningItem(code="EVAL_TIMEOUT_RATE_HIGH", message="Many candidates timed out; consider lowering max_new_tokens."))

    return ws

@app.get("/health")
def health():
    return {
        "ok": True,
        "device": device,
        "checkpoint": CHECKPOINT,
        "evaluators": {
            "fast": {
                "timeout_s": FAST_EVAL_CFG.timeout_s if fast_evaluator else 0,
            },
            "quality": {
                "timeout_s": QUALITY_EVAL_CFG.timeout_s if quality_evaluator else 0,
            },
        },
    }

def _pack_metrics(r: dict) -> CandidateMetrics:
    return CandidateMetrics(
        compile_ok=bool(r.get("compile_ok", False)),
        passed=int(r.get("passed", 0)),
        total=int(r.get("total", 0)),
        determinism=float(r.get("determinism", 0.0)),
        avg_ms=float(r.get("avg_ms", 0.0)),
        length_chars=int(r.get("length_chars", 0)),
        diversity=float(r.get("diversity", 0.0)),
        penalty=float(r.get("penalty", 0.0)),
    )

def _pack_lb(rows: list[dict], limit: int = 20) -> list[LeaderboardItem]:
    items: list[LeaderboardItem] = []
    for rank, r in enumerate(rows[: min(limit, len(rows))], start=1):
        items.append(LeaderboardItem(
            rank=rank,
            candidate_id=f"c_{int(r['idx']):04d}",
            score=float(r.get("score", 0.0)),
            reason=r.get("reason", "compile_fail"),
            warnings=[],
            metrics=_pack_metrics(r),
        ))
    return items

def _apply_mode_defaults(req: SolveRequest) -> SolveRequest:
    mode = getattr(req, "mode", None)
    if mode not in ("fast", "quality"):
        return req

    preset = PRESET_FAST if mode == "fast" else PRESET_QUALITY
    r = req.model_copy()

    # pydantic v2: fields user actually provided
    fields_set = getattr(req, "model_fields_set", set())

    # Only apply preset if client did NOT explicitly set that field
    for k, v in preset.items():
        if hasattr(r, k) and (k not in fields_set):
            setattr(r, k, v)

    return r

def _run_round(
    req: SolveRequest,
    extra: str,
    temperature: float,
    top_p: float,
    tries: int,
    n_candidates: int,
    max_new_tokens: int,
    ev: Evaluator,
) -> tuple[dict | None, list[dict], float]:
    """Return (best, leaderboard, gen_ms)."""
    prefix = _prompt(req, extra=extra)

    tg0 = time.perf_counter()
    bodies = generate_function_bodies(
        tokenizer, model, device,
        code_prefix=prefix,
        n=n_candidates,
        tries=tries,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    gen_ms = (time.perf_counter() - tg0) * 1000.0

    best, leaderboard = ev.select_best_generic(
        bodies, req.function_name, req.signature, req.docstring
    )
    return best, leaderboard, gen_ms

@app.post("/v1/solve", response_model=SolveResponse)
def solve(req: SolveRequest):
    request_id = str(uuid.uuid4())
    t0 = time.perf_counter()

    # Apply mode presets (but do NOT override client explicit fields)
    req2 = _apply_mode_defaults(req)

    # Hard safety: fast must be truly fast
    if getattr(req2, "mode", None) == "fast":
        req2.max_rounds = 1
        req2.accept_score = 0.0
    
    # Select evaluator
    ev = quality_evaluator
    if getattr(req2, "mode", None) == "fast":
        if fast_evaluator: 
            ev = fast_evaluator

    if ev is None:
        # Fallback if evaluators not ready or failed init
        # For now, just raise or return empty
        pass

    best: dict | None = None
    rows: list[dict] = []

    t_gen_ms = 0.0
    t_eval_ms = 0.0

    extra = ""

    for i in range(int(req2.max_rounds)):
        # Round 0: use requested temperature/top_p (preset already set)
        # Later rounds: diversify + feedback (mostly for quality)
        if i == 0:
            temperature, top_p = float(req2.temperature), float(req2.top_p)
            extra = ""
        else:
            temperature, top_p = 0.90, 0.95
            extra = _feedback(best) if best else ""

        te0 = time.perf_counter()
        best_i, rows_i, gen_ms_i = _run_round(
            req2,
            extra=extra,
            temperature=temperature,
            top_p=top_p,
            tries=int(req2.tries),
            n_candidates=int(req2.n_candidates),
            max_new_tokens=int(req2.max_new_tokens),
            ev=ev,
        )
        t_eval_ms += (time.perf_counter() - te0) * 1000.0
        t_gen_ms += gen_ms_i

        if best is None or (best_i and float(best_i.get("score", 0.0)) > float(best.get("score", 0.0))):
            best, rows = best_i, rows_i

        if best and float(best.get("score", 0.0)) >= float(req2.accept_score):
            break

    lb_items = _pack_lb(rows, limit=20)

    best_item: BestItem | None = None
    if best:
        best_item = BestItem(
            rank=1,
            candidate_id=f"c_{int(best['idx']):04d}",
            score=float(best.get("score", 0.0)),
            reason=best.get("reason", "compile_fail"),
            warnings=[],
            metrics=_pack_metrics(best),
            code_body=best.get("body", ""),
            code_full=best.get("code"),
        )

    timeout_rate = 0.0
    if rows:
        timeout_rate = sum(1 for r in rows if r.get("reason") == "timeout") / max(1, len(rows))

    ws = _pool_warnings(best, [r for r in rows[:20]], timeout_rate)

    t_total_ms = (time.perf_counter() - t0) * 1000.0

    return SolveResponse(
        request_id=request_id,
        model={"name": CHECKPOINT, "device": device, "dtype": "float16" if device == "cuda" else "float32"},
        input={"function_name": req.function_name, "signature": req.signature, "docstring": req.docstring},
        config=req2.model_dump(),
        best=best_item,
        leaderboard=lb_items,
        warnings=ws,
        timing={"t_total_ms": t_total_ms, "t_generate_ms": t_gen_ms, "t_eval_ms": t_eval_ms},
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
