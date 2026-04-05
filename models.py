from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class Observation(BaseModel):
    columns: List[str]
    shape: List[int]  # [rows, cols]
    missing_counts: Dict[str, int]
    duplicate_count: int
    sample_rows: List[Dict[str, Any]]  # first 5 rows
    issues: List[str]  # human-readable list of detected issues
    task_id: str
    step: int
    max_steps: int


class Action(BaseModel):
    operation: Literal[
        "fill_missing",
        "remove_duplicates",
        "standardize_format",
        "replace_value",
        "drop_outliers",
        "finish",
    ]
    column: Optional[str] = None
    strategy: Optional[str] = None   # mean, median, mode, or a literal value
    value: Optional[Any] = None      # literal fill value
    old_value: Optional[str] = None  # for replace_value
    new_value: Optional[str] = None  # for replace_value
    threshold: Optional[float] = None  # z-score threshold for outliers
    format_type: Optional[str] = None  # date, phone, category


class Reward(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    breakdown: Dict[str, float]  # per-criterion scores
    message: str
