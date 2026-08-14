from __future__ import annotations  # Must be first import

from typing import Any # helps to handle with types 
from datetime import datetime, timezone 
from pathlib import Path # helps to handle with files and directories

import json 
import shutil # helps to move files
import logging 

from project_config import (
    VALID_LABELS, FEEDBACK_FILE, FEEDBACK_DIR,
    RAW_TXTS_DIR, USEFUL_TXTS_DIR,
    RAW_PDFS_DIR, USEFUL_PDFS_DIR, 
    MANUAL_CHECK_DIR,
    DATA_DIR,
)
from label_files import create_labels

logger = logging.getLogger(__name__)


def build_pipeline_version( # for feedback.jsonl to track the version of the pipeline
    model_version: str | None,
    tfidf_version: str | None,
    scaler_version: str | None,
) -> str:
    parts = [
        f"model={model_version or 'unknown'}",
        f"tfidf={tfidf_version or 'unknown'}",
        f"scaler={scaler_version or 'unknown'}",
    ]
    return " | ".join(parts) # converts into one single string with separator " |"


# Private helpers
def _get_dirs(file_type: str) -> dict[str, Path]:
    """Return {label -> directory} mapping for a given file type."""
    if file_type == "txt":
        return {"useful": USEFUL_TXTS_DIR, "not_useful": RAW_TXTS_DIR}
    if file_type == "pdf":
        return {"useful": USEFUL_PDFS_DIR, "not_useful": RAW_PDFS_DIR}
    raise ValueError(f"Invalid file_type: {file_type!r}. Use 'txt' or 'pdf'.")


def _find_file(filename: str, file_type: str) -> tuple[Path, str | None]:
    """Locate a file across training and manual_check directories.

    Returns:
        (file_path, detected_label)
        detected_label is "useful" / "not_useful" / "manual_check.

    Raises:
        FileNotFoundError: if the file is not found anywhere.
    """
    dirs = _get_dirs(file_type)

    # 1. Search training directories (label-aware)
    for label, directory in dirs.items(): # iterate over the training directories: useful_texts and raw_texts or useful_pdfs and raw_pdfs
        candidate = directory / filename 
        if candidate.exists(): # check if file exists in training directory
            return candidate, label # returns the file path and the label

    # 2. Search manual_check (unlabelled)
    candidate = MANUAL_CHECK_DIR / filename 
    if candidate.exists(): # check if file exists in manual_check directory
        return candidate, None # returns the file path and None as the label

    # Not found
    searched = ", ".join(d.name for d in [*dirs.values(), MANUAL_CHECK_DIR]) # create a string of all directories that were searched
    raise FileNotFoundError(f"File '{filename}' not found. Searched: {searched}") # raise FileNotFoundError if file is not found


def _move_file(
    file_path: Path, # path to the file
    target_dir: Path, # target directory
    filename: str, # file name
) -> tuple[Path, bool]: # returns the target directory and a boolean indicating if the file was already in the target directory
    """Move a file to the target directory.

    Returns:
        (target_directory, was_already_there)
    """
    was_already_there = file_path.parent.resolve() == target_dir.resolve() # check if file is already in the target directory

    if was_already_there:
        logger.info(f"[CORRECT] {filename}: already in {target_dir.name}/")
        return target_dir, True

    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(file_path), target_dir / filename)
    logger.info(f"[CORRECT] {filename}: {file_path.parent.name}/ → {target_dir.name}/")
    return target_dir, False


def _log_to_jsonl(record: dict) -> None:
    """Append a single feedback record to feedback.jsonl."""
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    with open(FEEDBACK_FILE, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n") # json.dumps() converts a Python dictionary into a JSON formatted string. ensure_ascii=False ensures that non-ASCII characters are written as-is, and + "\n" appends a newline character to ensure each record is on its own line in the JSONL file.


def _refresh_labels(file_type: str, update: bool) -> None: # for feedback.jsonl to track the version of the pipeline
    """Re-generate labels.csv if requested."""
    if not update:
        return
    try:
        create_labels(file_type=file_type)
        logger.info("  labels.csv updated.")
    except Exception as exc:
        logger.warning(f"  labels.csv update failed: {exc}")


# Public API
def correct(
    filename: str,
    decision: str,
    *,
    predicted_score: float | None = None,
    file_type: str = "txt",
    reviewer: str | None = None,
    update_labels: bool = True,
) -> dict[str, str]:
    """Unified correction: change the label of a wrong/uncertain prediction.

    decision="useful" | "not_useful" (explicit label)
      Moves the file to the directory matching the reviewer's
      decision.  Use for wrong predictions + manual_check directory.

    Args:
        filename:        Name of the file (e.g. "paper_42.txt").
        decision:        Reviewer's label.
        predicted_score: Model's confidence score (optional, for logging).
        file_type:       "txt" or "pdf" (default: "txt").
        reviewer:        Optional reviewer name for logging.
        update_labels:   If True (default), refresh labels.csv after move.

    Returns:
        A dict with correction details::

            {"filename", "old_label", "new_label", "moved_to",
             "was_already_there"}

    Raises:
        FileNotFoundError: If the file is not found.
        ValueError:        If decision is not a valid label or file_type
                           is invalid.

    Examples:

        # Explicit label (review_uncertain): reviewer labels uncertain file
        correct("paper_42.txt", "useful")
    """
    if decision not in VALID_LABELS:
        raise ValueError( 
            f"decision must be one of {sorted(VALID_LABELS)}, "
            f"got {decision!r}"
        )

    dirs = _get_dirs(file_type)
    file_path, current_label = _find_file(filename, file_type)


    new_label = decision
    method = "Wrong marked or Uncertain labeled"

    target_dir, was_already_there = _move_file(
        file_path, dirs[new_label], filename
    )

    now = datetime.now(timezone.utc)
    _log_to_jsonl({
        "record_id":       f"{now.strftime('%Y%m%d%H%M%S%f')}_{filename}",
        "filename":        filename,
        "old_label":       current_label,
        "new_label":       new_label,
        "predicted_score": predicted_score,
        "reviewer":        reviewer,
        "method":          method,
        "apply_status":    "applied",
        "applied_at":      now.isoformat(),
        "created_at":      now.isoformat(),
    })

    _refresh_labels(file_type, update_labels)

    return {
        "filename":          filename,
        "old_label":         current_label,
        "new_label":         new_label,
        "moved_to":          target_dir.name,
        "was_already_there": was_already_there,
    }

# Detailed feedback for the full feedback_apply.py workflow (for the future improvements)

def submit_feedback(
    filename: str,
    predicted_label: str,
    predicted_score: float,
    human_label: str,
    annotator: str | None = None,
    notes: str | None = None,
    model_version: str | None = None,
    tfidf_version: str | None = None,
    scaler_version: str | None = None,
    source_dir: str | None = None,  # Directory the file came from at prediction time ("useful_txts" / "raw_txts")
) -> dict[str, Any]:
    """Save human feedback as a JSONL record.

    This is the detailed version that stores prediction metadata.
    Records are applied later by running feedback_apply.py.
    For a simpler one-call correction, use correct() instead.

    Args:
        filename:        Name of the file that was predicted (e.g. 'paper_42.txt').
        predicted_label: Label the model assigned.
        predicted_score: Confidence score produced by the model.
        human_label:     Correct label provided by the reviewer.
        annotator:       Reviewer name or ID (optional).
        notes:           Free-text notes from the reviewer (optional).
        model_version:   Model artifact version used for prediction (optional).
        tfidf_version:   TF-IDF artifact version used for prediction (optional).
        scaler_version:  Scaler artifact version used for prediction (optional).
        source_dir:      Directory where the file was located at prediction time.
                         When provided, feedback_apply.py uses this as the first
                         search path, avoiding an exhaustive directory scan.
                         Typical values: "useful_txts", "raw_txts",
                         "useful_pdfs", "raw_pdfs".

    Returns:
        The feedback record dict that was written to FEEDBACK_FILE.

    Raises:
        ValueError: If filename is empty or either label is not in VALID_LABELS.
    """
    # Important: wrong data can destruct the structure of JSON
    if not filename or not filename.strip():  # Check if the file name is empty
        raise ValueError("Filename cannot be empty")

    if predicted_label not in VALID_LABELS:  # Check if the provided label is valid
        raise ValueError(f"Predicted label must be one of {sorted(VALID_LABELS)}")

    if human_label not in VALID_LABELS:  # Check if the human gave a valid label
        raise ValueError(f"Human label must be one of {sorted(VALID_LABELS)}")

    now = datetime.now(timezone.utc)

    record = {
        "record_id":        f"{now.strftime('%Y%m%d%H%M%S%f')}_{filename}",  # Unique ID
        "filename":         filename,
        "predicted_label":  predicted_label,
        "predicted_score":  float(predicted_score),
        "human_label":      human_label,
        "annotator":        annotator,       # Reviewer name or ID
        "notes":            notes,
        "model_version":    model_version,
        "tfidf_version":    tfidf_version,
        "scaler_version":   scaler_version,
        "pipeline_version": build_pipeline_version(model_version, tfidf_version, scaler_version),
        "source_dir":       source_dir,      # Hint for feedback_apply.py (where the file lived)
        "apply_status":     "pending",       # "pending" | "applied" | "skipped" | "file_not_found"
        "applied_at":       None,            # Filled in by feedback_apply.py after processing
        "created_at":       now.isoformat()  # ISO 8601 UTC timestamp
    }

    _log_to_jsonl(record)

    return record
