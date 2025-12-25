from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal, Optional


Reason = Literal[
    "ok",
    "ok_but_suspicious",
    "ok_but_constant",
    "partial_runtime",
    "nondeterministic",
    "timeout",
    "compile_fail",
    "missing_func",
]


class WarningItem(BaseModel):
    code: str
    message: str


class CandidateMetrics(BaseModel):
    compile_ok: bool
    passed: int
    total: int
    determinism: float
    avg_ms: float
    length_chars: int
    diversity: float = 0.0
    penalty: float = 0.0


class LeaderboardItem(BaseModel):
    rank: int
    candidate_id: str
    score: float
    reason: Reason
    warnings: List[WarningItem] = Field(default_factory=list)
    metrics: CandidateMetrics


class BestItem(LeaderboardItem):
    code_body: str
    code_full: Optional[str] = None


class SolveRequest(BaseModel):
    function_name: str
    signature: str
    docstring: str = ""
    mode: str = "quality"  # or "fast"


    # generation
    n_candidates: int = Field(default=12, ge=1, le=64)
    tries: int = Field(default=2, ge=1, le=10)
    max_new_tokens: int = Field(default=160, ge=16, le=512)
    temperature: float = Field(default=0.85, ge=0.0, le=2.0)
    top_p: float = Field(default=0.92, ge=0.0, le=1.0)

    # evaluation (logical knobs; pool stays persistent)
    timeout_s: float = Field(default=3.0, ge=0.2, le=30.0)
    repeats: int = Field(default=2, ge=1, le=5)
    probe_cases: int = Field(default=6, ge=1, le=20)

    # policy
    accept_score: float = Field(default=80.0, ge=0.0, le=100.0)
    max_rounds: int = Field(default=2, ge=1, le=5)


class SolveResponse(BaseModel):
    request_id: str
    model: Dict[str, Any]
    input: Dict[str, str]
    config: Dict[str, Any]

    best: Optional[BestItem]
    leaderboard: List[LeaderboardItem]

    warnings: List[WarningItem]
    timing: Dict[str, float]
