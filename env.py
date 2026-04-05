"""
DataCleaningEnv — OpenEnv-compliant environment for CSV data cleaning.
"""
import re
import copy
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Tuple

from models import Action, Observation, Reward
from datasets import easy_dataset, medium_dataset, hard_dataset
from graders import GRADERS

TASK_CONFIGS = {
    "easy":   {"dataset_fn": easy_dataset,   "max_steps": 10},
    "medium": {"dataset_fn": medium_dataset, "max_steps": 20},
    "hard":   {"dataset_fn": hard_dataset,   "max_steps": 30},
}


class DataCleaningEnv:
    def __init__(self, task_id: str = "easy"):
        if task_id not in TASK_CONFIGS:
            raise ValueError(f"task_id must be one of {list(TASK_CONFIGS)}")
        self.task_id = task_id
        self._config = TASK_CONFIGS[task_id]
        self._df: Optional[pd.DataFrame] = None
        self._step_count = 0
        self._done = False

    # ------------------------------------------------------------------ #
    #  OpenEnv API                                                         #
    # ------------------------------------------------------------------ #

    def reset(self) -> Observation:
        """Reset environment to initial dirty dataset."""
        self._df = self._config["dataset_fn"]()
        self._step_count = 0
        self._done = False
        return self._observe()

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """
        Apply one cleaning operation.
        Returns (observation, reward, done, info).
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new one.")

        info: Dict[str, Any] = {"operation": action.operation, "error": None}

        if action.operation == "finish":
            self._done = True
        else:
            error = self._apply_action(action)
            if error:
                info["error"] = error

        self._step_count += 1
        if self._step_count >= self._config["max_steps"]:
            self._done = True

        reward_obj: Reward = GRADERS[self.task_id](self._df)
        obs = self._observe()

        # Penalty for doing nothing useful (loop detection)
        if info["error"]:
            adjusted_score = max(0.0, reward_obj.score - 0.02)
        else:
            adjusted_score = reward_obj.score

        info["reward_breakdown"] = reward_obj.breakdown
        info["reward_message"] = reward_obj.message

        return obs, adjusted_score, self._done, info

    def state(self) -> Dict[str, Any]:
        """Return full internal state (for inspection/debugging)."""
        if self._df is None:
            return {"status": "not_started"}
        # Convert to JSON-safe types
        records = self._df.where(pd.notnull(self._df), None).to_dict(orient="records")
        safe_records = [
            {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
             for k, v in row.items()}
            for row in records
        ]
        return {
            "task_id": self.task_id,
            "step": self._step_count,
            "max_steps": self._config["max_steps"],
            "done": self._done,
            "dataframe": safe_records,
            "shape": list(self._df.shape),
        }

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _observe(self) -> Observation:
        df = self._df
        missing = {col: int(df[col].isna().sum()) for col in df.columns}
        issues = self._detect_issues(df)
        raw_rows = df.head(5).where(pd.notnull(df.head(5)), None).to_dict(orient="records")
        safe_rows = [
            {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
             for k, v in row.items()}
            for row in raw_rows
        ]
        return Observation(
            columns=list(df.columns),
            shape=list(df.shape),
            missing_counts=missing,
            duplicate_count=int(df.duplicated().sum()),
            sample_rows=safe_rows,
            issues=issues,
            task_id=self.task_id,
            step=self._step_count,
            max_steps=self._config["max_steps"],
        )

    def _detect_issues(self, df: pd.DataFrame):
        issues = []

        # 1. Duplicates
        dups = df.duplicated().sum()
        if dups > 0:
            issues.append(f"{dups} duplicate rows detected")

        # 2. Missing values
        for col in df.columns:
            missing = df[col].isna().sum()
            if missing > 0:
                issues.append(f"Column '{col}' has {missing} missing values")

        # 3. Outliers in numeric columns
        for col in df.select_dtypes(include=[np.number]).columns:
            if col in ("id", "order_id", "customer_id"):
                continue
            col_data = df[col].dropna()
            if len(col_data) > 3:
                z_scores = np.abs((col_data - col_data.mean()) / (col_data.std() + 1e-9))
                outliers = (z_scores > 3).sum()
                if outliers > 0:
                    issues.append(f"Column '{col}' has {outliers} potential outlier(s)")

        # 4. Inconsistent casing/values in string columns
        for col in df.columns:
            if df[col].dtype == object:
                unique_lower = df[col].dropna().str.lower().nunique()
                unique_orig = df[col].dropna().nunique()
                if unique_orig > unique_lower:
                    issues.append(f"Column '{col}' has inconsistent casing/values")

        # 5. Date format inconsistency — detect mixed formats
        date_cols = [c for c in df.columns if any(kw in c.lower() for kw in ("date", "dob", "birth", "created"))]
        iso_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
        for col in date_cols:
            non_null = df[col].dropna().astype(str)
            non_iso = non_null[~non_null.apply(lambda x: bool(iso_pattern.match(x)))]
            if len(non_iso) > 0:
                issues.append(f"Column '{col}' has {len(non_iso)} non-standard date format(s) — use standardize_format with format_type=date")

        # 6. Phone format inconsistency
        phone_cols = [c for c in df.columns if "phone" in c.lower()]
        phone_pattern = re.compile(r"^\(\d{3}\)\d{3}-\d{4}$")
        for col in phone_cols:
            non_null = df[col].dropna().astype(str)
            non_standard = non_null[~non_null.apply(lambda x: bool(phone_pattern.match(x)))]
            if len(non_standard) > 0:
                issues.append(f"Column '{col}' has {len(non_standard)} non-standard phone format(s) — use standardize_format with format_type=phone")

        return issues

    def _apply_action(self, action: Action) -> Optional[str]:
        """Apply action to self._df. Returns error string or None."""
        df = self._df
        op = action.operation

        try:
            if op == "remove_duplicates":
                self._df = df.drop_duplicates().reset_index(drop=True)

            elif op == "fill_missing":
                col = action.column
                if col not in df.columns:
                    return f"Column '{col}' not found"
                strategy = action.strategy or "mean"
                if strategy in ("mean",):
                    fill_val = df[col].mean() if df[col].dtype in [np.float64, np.int64, float, int] else df[col].mode()[0]
                elif strategy == "median":
                    fill_val = df[col].median() if df[col].dtype in [np.float64, np.int64, float, int] else df[col].mode()[0]
                elif strategy in ("mode", "most_frequent"):  # support both names
                    fill_val = df[col].mode()[0] if len(df[col].mode()) > 0 else "Unknown"
                elif strategy == "value":
                    fill_val = action.value
                else:
                    # fallback: mode for strings, mean for numbers
                    if df[col].dtype == object:
                        fill_val = df[col].mode()[0] if len(df[col].mode()) > 0 else "Unknown"
                    else:
                        fill_val = df[col].mean()
                self._df[col] = df[col].fillna(fill_val)

            elif op == "replace_value":
                col = action.column
                if col not in df.columns:
                    return f"Column '{col}' not found"
                self._df[col] = df[col].replace(action.old_value, action.new_value)

            elif op == "standardize_format":
                col = action.column
                fmt = action.format_type
                if col not in df.columns:
                    return f"Column '{col}' not found"
                if fmt == "date":
                    self._df[col] = pd.to_datetime(df[col], format="mixed", dayfirst=False).dt.strftime("%Y-%m-%d")
                elif fmt == "phone":
                    def norm_phone(p):
                        if pd.isna(p):
                            return p
                        digits = re.sub(r'\D', '', str(p))[-10:]
                        return f"({digits[:3]}){digits[3:6]}-{digits[6:]}"
                    self._df[col] = df[col].apply(norm_phone)
                elif fmt == "upper":
                    self._df[col] = df[col].str.upper()
                elif fmt == "lower":
                    self._df[col] = df[col].str.lower()
                elif fmt in ("title", "category", "normalize"):  # support "category" as alias for title
                    self._df[col] = df[col].dropna().str.title()
                    self._df[col] = self._df[col].str.title()
                elif fmt == "category_gender":
                    def norm_gender(g):
                        if pd.isna(g):
                            return g
                        g = str(g).strip().lower()
                        return "M" if g in ("m", "male") else "F"
                    self._df[col] = df[col].apply(norm_gender)
                else:
                    # fallback: try title case for unknown formats on string columns
                    if df[col].dtype == object:
                        self._df[col] = df[col].str.title()
                    else:
                        return f"Unknown format_type '{fmt}'"

            elif op == "drop_outliers":
                col = action.column
                threshold = action.threshold or 3.0
                if col not in df.columns:
                    return f"Column '{col}' not found"
                col_data = df[col].dropna()
                z_scores = np.abs((col_data - col_data.mean()) / (col_data.std() + 1e-9))
                keep_idx = z_scores[z_scores <= threshold].index
                # Keep rows that either have no value in col or are within threshold
                mask = df[col].isna() | df.index.isin(keep_idx)
                self._df = df[mask].reset_index(drop=True)

        except Exception as e:
            return str(e)

        return None
