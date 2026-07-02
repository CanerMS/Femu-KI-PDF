from __future__ import annotations

from typing import Any
from pathlib import Path
from datetime import datetime, timezone

from project_config import *

import json


def build_pipeline_version(
    model_version: str | None,
    tfidf_version: str | None,
    scaler_version: str | None,
) -> str:
    parts = [
        f"model={model_version or 'unknown'}",
        f"tfidf={tfidf_version or 'unknown'}",
        f"scaler={scaler_version or 'unknown'}",
    ]
    return " | ".join(parts)



def submit_feedback(
    filename: str,
    predicted_label: str,
    predicted_score: float,
    human_label: str,
    annotator: str | None = None,
    notes: str | None = None,
    model_version: str | None = None,
    tfidf_version: str | None = None,
    scaler_version: str | None = None
) -> dict[str, Any]:
    # Save human feedback as JSONL   
    # Important, because wrong data can destruct the structure of JSON
    if not filename or not filename.strip(): # Check if the file name is empty
        raise ValueError("Filename cannot be empty")

    if predicted_label not in VALID_LABELS: # Check if the provided label is valid or exists
        raise ValueError(f"Predicted label must be one of {sorted(VALID_LABELS)}")

    if human_label not in VALID_LABELS: # Check if the human gave the valid input   
        raise ValueError(f"Human label must be one of {sorted(VALID_LABELS)}") 

    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)

    record = {
        "record_id": f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}_{filename}",
        "filename": filename,
        "predicted_label": predicted_label,
        "predicted_score": float(predicted_score),
        "human_label": human_label,
        "annotator": annotator, # reviewer name or id number
        "notes": notes,
        "model_version": model_version,
        "tfidf_version": tfidf_version,
        "scaler_version": scaler_version,
        "pipeline_version": build_pipeline_version(model_version, tfidf_version, scaler_version), # summary of three models
        "created_at": datetime.now(timezone.utc).isoformat()
    }

    with open(FEEDBACK_FILE, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return record


