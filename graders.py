"""
Programmatic graders for each task.
Each grader returns a score 0.0–1.0 with a breakdown dict.
"""
import re
import pandas as pd
import numpy as np
from models import Reward


def grade_easy(df: pd.DataFrame) -> Reward:
    """Grade the easy task: duplicates removed + missing salary filled."""
    breakdown = {}

    # 1. Duplicate removal (0.4)
    dup_count = df.duplicated().sum()
    breakdown["duplicates_removed"] = 0.4 if dup_count == 0 else max(0.0, 0.4 - dup_count * 0.1)

    # 2. Missing salary filled (0.4)
    missing_salary = df["salary"].isna().sum() if "salary" in df.columns else 999
    breakdown["missing_filled"] = 0.4 if missing_salary == 0 else max(0.0, 0.4 - missing_salary * 0.1)

    # 3. No data loss — original unique rows should be 8 (0.2)
    expected_rows = 8
    actual_rows = len(df)
    breakdown["row_integrity"] = 0.2 if actual_rows == expected_rows else max(0.0, 0.2 - abs(actual_rows - expected_rows) * 0.05)

    score = round(sum(breakdown.values()), 3)
    messages = []
    if dup_count == 0:
        messages.append("✓ All duplicates removed")
    else:
        messages.append(f"✗ {dup_count} duplicates remain")
    if missing_salary == 0:
        messages.append("✓ All missing salaries filled")
    else:
        messages.append(f"✗ {missing_salary} missing salary values remain")

    return Reward(score=min(score, 1.0), breakdown=breakdown, message=" | ".join(messages))


def grade_medium(df: pd.DataFrame) -> Reward:
    """Grade medium task: standardized dates, phones, gender."""
    breakdown = {}

    # 1. Date format consistency (0.35) — all should be YYYY-MM-DD
    if "dob" in df.columns:
        date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
        valid_dates = df["dob"].dropna().apply(lambda x: bool(date_pattern.match(str(x)))).sum()
        total_dates = df["dob"].dropna().shape[0]
        breakdown["date_standardized"] = round(0.35 * valid_dates / max(total_dates, 1), 3)
    else:
        breakdown["date_standardized"] = 0.0

    # 2. Phone format consistency (0.35) — all should be (XXX)XXX-XXXX
    if "phone" in df.columns:
        phone_pattern = re.compile(r"^\(\d{3}\)\d{3}-\d{4}$")
        valid_phones = df["phone"].dropna().apply(lambda x: bool(phone_pattern.match(str(x)))).sum()
        total_phones = df["phone"].dropna().shape[0]
        breakdown["phone_standardized"] = round(0.35 * valid_phones / max(total_phones, 1), 3)
    else:
        breakdown["phone_standardized"] = 0.0

    # 3. Gender normalization (0.30) — accept M/F or Male/Female (case-insensitive)
    if "gender" in df.columns:
        def is_valid_gender(g):
            g = str(g).strip().lower()
            return g in ("m", "f", "male", "female")
        valid_genders = df["gender"].dropna().apply(is_valid_gender).sum()
        total_genders = df["gender"].dropna().shape[0]
        breakdown["gender_normalized"] = round(0.30 * valid_genders / max(total_genders, 1), 3)
    else:
        breakdown["gender_normalized"] = 0.0

    score = round(sum(breakdown.values()), 3)
    date_pct  = breakdown["date_standardized"]  / 0.35 * 100
    phone_pct = breakdown["phone_standardized"] / 0.35 * 100
    gender_pct= breakdown["gender_normalized"]  / 0.30 * 100
    return Reward(
        score=min(score, 1.0),
        breakdown=breakdown,
        message=f"Date:{date_pct:.0f}% | Phone:{phone_pct:.0f}% | Gender:{gender_pct:.0f}%"
    )


def grade_hard(df: pd.DataFrame) -> Reward:
    """Grade hard task: duplicates, missing, outliers, formats, product names."""
    breakdown = {}

    # 1. Duplicates removed (0.2)
    dup_count = df.duplicated(subset=["order_id"] if "order_id" in df.columns else None).sum()
    breakdown["duplicates_removed"] = 0.2 if dup_count == 0 else max(0.0, 0.2 - dup_count * 0.05)

    # 2. Missing values filled (0.2)
    total_missing = df.isnull().sum().sum()
    breakdown["missing_filled"] = 0.2 if total_missing == 0 else max(0.0, 0.2 - total_missing * 0.02)

    # 3. Outliers removed — revenue > 10000 considered outlier (0.2)
    if "revenue" in df.columns:
        outliers = (df["revenue"] > 10000).sum()
        breakdown["outliers_removed"] = 0.2 if outliers == 0 else max(0.0, 0.2 - outliers * 0.05)
    else:
        breakdown["outliers_removed"] = 0.0

    # 4. Product name standardization — title case (0.2)
    if "product" in df.columns:
        non_null = df["product"].dropna()
        title_case = non_null.apply(lambda x: x == x.title()).sum()
        breakdown["product_standardized"] = round(0.2 * title_case / max(len(non_null), 1), 3)
    else:
        breakdown["product_standardized"] = 0.0

    # 5. Date format consistency — YYYY-MM-DD (0.2)
    if "date" in df.columns:
        date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
        valid = df["date"].dropna().apply(lambda x: bool(date_pattern.match(str(x)))).sum()
        total = df["date"].dropna().shape[0]
        breakdown["date_standardized"] = round(0.2 * valid / max(total, 1), 3)
    else:
        breakdown["date_standardized"] = 0.0

    score = round(sum(breakdown.values()), 3)
    return Reward(score=min(score, 1.0), breakdown=breakdown, message=str(breakdown))


GRADERS = {
    "easy": grade_easy,
    "medium": grade_medium,
    "hard": grade_hard,
}
