# DataCleaning OpenEnv

An OpenEnv-compliant environment where an AI agent cleans dirty CSV datasets through a structured action API.

## Environment Description

The agent is presented with a messy DataFrame and must apply cleaning operations step-by-step to fix real data quality issues: duplicates, missing values, inconsistent formats, outliers, and value normalization.

# Prepend HF metadata to README.md
$metadata = @"
---
title: OpenEnv Data Cleaning
emoji: 🧹
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
---

"@
$existing = Get-Content README.md -Raw
Set-Content README.md ($metadata + $existing)


## Action Space

| Operation | Parameters | Description |
|---|---|---|
| `remove_duplicates` | — | Drop exact duplicate rows |
| `fill_missing` | `column`, `strategy` (mean/median/mode/value), `value` | Fill NaN values |
| `standardize_format` | `column`, `format_type` (date/phone/title/upper/lower/category_gender) | Normalize formats |
| `replace_value` | `column`, `old_value`, `new_value` | Replace specific values |
| `drop_outliers` | `column`, `threshold` (z-score, default 3.0) | Remove statistical outliers |
| `finish` | — | Signal task complete |

## Observation Space

```json
{
  "columns": ["id", "name", "salary"],
  "shape": [10, 5],
  "missing_counts": {"salary": 3},
  "duplicate_count": 2,
  "sample_rows": [...],
  "issues": ["2 duplicate rows detected", "Column 'salary' has 3 missing values"],
  "task_id": "easy",
  "step": 1,
  "max_steps": 10
}
```

## Tasks

| ID | Name | Difficulty | Max Steps | Issues to Fix |
|---|---|---|---|---|
| `easy` | Remove Duplicates & Fill Nulls | Easy | 10 | Duplicates + missing salary |
| `medium` | Format Standardization | Medium | 20 | Date/phone/gender inconsistency |
| `hard` | Full Multi-Issue Cleaning | Hard | 30 | All of the above + outliers |

## Reward Function

- **Partial credit** — score updates every step based on current data quality
- **Per-criterion breakdown** — separate scores for each issue type
- **Row integrity penalty** — penalizes over-dropping of valid data
- **Error penalty** — -0.02 per invalid action

Score range: `0.0 – 1.0`

## Setup

```bash
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 7860
```

## API Endpoints

```
GET  /           — health check
POST /reset      — reset environment (?task_id=easy|medium|hard)
POST /step       — apply action (JSON body)
GET  /state      — inspect internal state
GET  /tasks      — list all tasks
```

## Docker

```bash
docker build -t data-cleaning-env .
docker run -p 7860:7860 data-cleaning-env
```

## Running Inference

The inference script uses the OpenAI client and reads all credentials from environment variables.
It is compatible with any OpenAI-compatible API endpoint (OpenAI, Groq, HuggingFace, etc.).

**Required environment variables:**

| Variable | Description | Example |
|---|---|---|
| `API_BASE_URL` | LLM API endpoint | `https://api.groq.com/openai/v1` |
| `MODEL_NAME` | Model identifier | `llama-3.1-8b-instant` |
| `HF_TOKEN` | Your API key (used as Bearer token) | `gsk_...` or `hf_...` |
| `ENV_BASE_URL` | URL where this environment is running | `http://localhost:7860` |

**Linux/Mac:**
```bash
export API_BASE_URL=https://api.groq.com/openai/v1
export MODEL_NAME=llama-3.1-8b-instant
export HF_TOKEN=your_api_key_here
export ENV_BASE_URL=http://localhost:7860

python inference.py
```

**Windows (PowerShell):**
```powershell
$env:API_BASE_URL = "https://api.groq.com/openai/v1"
$env:MODEL_NAME   = "llama-3.1-8b-instant"
$env:HF_TOKEN     = "your_api_key_here"
$env:ENV_BASE_URL = "http://localhost:7860"

python inference.py
```

**Compatible providers (any OpenAI-compatible endpoint works):**
- [Groq](https://console.groq.com) — `https://api.groq.com/openai/v1`
- [OpenAI](https://platform.openai.com) — `https://api.openai.com/v1`
- [HuggingFace](https://huggingface.co/settings/tokens) — `https://api-inference.huggingface.co/v1`

## Baseline Scores

| Task | Score |
|---|---|
| easy | ~0.95 |
| medium | ~0.85 |
| hard | ~0.70 |

## Infra

- Runtime: < 20 min
- Requirements: 2 vCPU, 8 GB RAM