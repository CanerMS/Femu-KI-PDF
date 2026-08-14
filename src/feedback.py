from __future__ import annotations  # Must be first import

from typing import Any
from datetime import datetime, timezone

import json
import shutil
import logging

from project_config import (
    VALID_LABELS, FEEDBACK_FILE, FEEDBACK_DIR,
    RAW_TXTS_DIR, USEFUL_TXTS_DIR,
    RAW_PDFS_DIR, USEFUL_PDFS_DIR,
)
from label_files import create_labels

logger = logging.getLogger(__name__)


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


# Simple correction — "Falsch" button


def mark_wrong(
    filename: str,
    file_type: str = "txt",
    reviewer: str | None = None,
    update_labels: bool = True,
) -> dict[str, str]:
    """One-call correction: reviewer says "Falsch" (wrong).

    Finds the file, flips its label (useful ↔ not_useful),
    moves it to the opposite directory, logs the action to
    feedback.jsonl for traceability, and refreshes labels.csv.

    Args:
        filename:      Name of the file (e.g. "paper_42.txt").
        file_type:     "txt" or "pdf" (default: "txt").
        reviewer:      Optional reviewer name for logging.
        update_labels: If True (default), call label_files.py
                       to regenerate labels.csv after the move.

    Returns:
        A dict with the correction details:
            {"filename", "old_label", "new_label", "moved_to"}

    Raises:
        FileNotFoundError: If the file is not found in any directory.
        ValueError:        If file_type is invalid.

    Example:
        >>> from feedback import mark_wrong
        >>> mark_wrong("paper_femur_biomech.txt")
        {'filename': 'paper_femur_biomech.txt',
         'old_label': 'not_useful',
         'new_label': 'useful',
         'moved_to': 'useful_txts'}
    """
    # 1. Determine directory mapping based on file type
    if file_type == "txt":
        dirs = {"useful": USEFUL_TXTS_DIR, "not_useful": RAW_TXTS_DIR}
    elif file_type == "pdf":
        dirs = {"useful": USEFUL_PDFS_DIR, "not_useful": RAW_PDFS_DIR}
    else:
        raise ValueError(f"Invalid file_type: {file_type!r}. Use 'txt' or 'pdf'.")

    # 2. Find the file and determine its current label
    current_label = None
    file_path = None
    
    from project_config import DATA_DIR
    sorted_dir = DATA_DIR / "sorted_pdfs"
    
    search_paths = {
        "useful": [dirs["useful"], sorted_dir / "Useful"],
        "not_useful": [dirs["not_useful"], sorted_dir / "Not_Useful"]
    }

    for label, paths in search_paths.items():
        for directory in paths:
            candidate = directory / filename
            if candidate.exists():
                current_label = label
                file_path = candidate
                break
        if file_path:
            break

    if file_path is None:
        searched = ", ".join(str(d.name) for paths in search_paths.values() for d in paths)
        raise FileNotFoundError(
            f"File '{filename}' not found. Searched: {searched}"
        )

    # 3. Flip the label
    new_label = "useful" if current_label == "not_useful" else "not_useful"
    train_target_dir = dirs[new_label]
    
    # 4. Move and/or Copy the file
    is_in_sorted = sorted_dir.resolve() in file_path.resolve().parents
    
    train_target_dir.mkdir(parents=True, exist_ok=True)
    
    if is_in_sorted:
        # 1. Copy to train directory
        shutil.copy(str(file_path), train_target_dir / filename)
        # 2. Move to the corrected sorted_pdfs folder
        sorted_target_dir = sorted_dir / ("Useful" if new_label == "useful" else "Not_Useful")
        sorted_target_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(file_path), sorted_target_dir / filename)
        target_dir = sorted_target_dir
        logger.info(f"[MARK WRONG] {filename}: {current_label} → {new_label} "
                    f"(Moved to {sorted_target_dir.name}/ AND Copied to {train_target_dir.name}/)")
    else:
        # Just move within train directories
        shutil.move(str(file_path), train_target_dir / filename)
        target_dir = train_target_dir
        logger.info(f"[MARK WRONG] {filename}: {current_label} → {new_label} "
                    f"({file_path.parent.name}/ → {train_target_dir.name}/)")

    # 5. Log to JSONL for traceability (lightweight — also connects to the
    #    full feedback_apply.py workflow if needed later)
    now = datetime.now(timezone.utc)
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)

    log_record = {
        "record_id":   f"{now.strftime('%Y%m%d%H%M%S%f')}_{filename}",
        "filename":    filename,
        "old_label":   current_label,
        "new_label":   new_label,
        "reviewer":    reviewer,
        "method":      "mark_wrong",       # distinguishes from submit_feedback records
        "apply_status": "applied",         # already applied (immediate move)
        "applied_at":  now.isoformat(),
        "created_at":  now.isoformat(),
    }

    with open(FEEDBACK_FILE, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(log_record, ensure_ascii=False) + "\n")

    # 6. Refresh labels.csv so it reflects the new directory state
    if update_labels:
        try:
            create_labels(file_type=file_type)
            logger.info(f"  labels.csv updated.")
        except Exception as exc:
            logger.warning(f"  labels.csv update failed: {exc}")

    return {
        "filename":  filename,
        "old_label": current_label,
        "new_label": new_label,
        "moved_to":  target_dir.name,
    }


# Reviewer decision for low-confidence (below threshold) predictions
def review_uncertain(
    filename: str,
    decision: str,
    predicted_score: float | None = None,
    file_type: str = "txt",
    reviewer: str | None = None,
    update_labels: bool = True,
) -> dict[str, str]:
    """Reviewer labels an article that the model was uncertain about.

    When the model's confidence score is below the threshold,
    the article is sent to a reviewer. The reviewer reads it and
    decides: "useful" or "not_useful". The file is then placed
    in the correct directory so the model can learn from it
    during the next retrain cycle.

    Args:
        filename:        Name of the file (e.g. "paper_42.txt").
        decision:        Reviewer's label — "useful" or "not_useful".
        predicted_score: Model's confidence score (optional, logged for tracking).
        file_type:       "txt" or "pdf" (default: "txt").
        reviewer:        Optional reviewer name for logging.
        update_labels:   If True (default), refresh labels.csv after the move.

    Returns:
        A dict with the review details:
            {"filename", "decision", "placed_in", "was_already_there"}

    Raises:
        FileNotFoundError: If the file is not found in any known directory.
        ValueError:        If decision is not "useful" or "not_useful".

    Example:
        >>> from feedback import review_uncertain
        >>> review_uncertain("low_score_paper.txt", decision="useful")
        {'filename': 'low_score_paper.txt',
         'decision': 'useful',
         'placed_in': 'useful_txts',
         'was_already_there': False}
    """
    # Validate decision
    if decision not in VALID_LABELS:
        raise ValueError(f"Decision must be one of {sorted(VALID_LABELS)}, got {decision!r}")

    # Directory mapping
    if file_type == "txt":
        dirs = {"useful": USEFUL_TXTS_DIR, "not_useful": RAW_TXTS_DIR}
    elif file_type == "pdf":
        dirs = {"useful": USEFUL_PDFS_DIR, "not_useful": RAW_PDFS_DIR}
    else:
        raise ValueError(f"Invalid file_type: {file_type!r}. Use 'txt' or 'pdf'.")

    # Find the file — could be in core dirs, to_test_files, or sorted_pdfs
    from project_config import DATA_DIR
    sorted_dir = DATA_DIR / "sorted_pdfs"
    
    search_dirs = list(dirs.values()) + [
        sorted_dir / "Manual_Check",
        sorted_dir / "Useful",
        sorted_dir / "Not_Useful",
        DATA_DIR / "to_test_files"
    ]

    file_path = None
    for directory in search_dirs:
        candidate = directory / filename
        if candidate.exists():
            file_path = candidate
            break

    if file_path is None:
        searched = ", ".join(d.name for d in search_dirs)
        raise FileNotFoundError(
            f"File '{filename}' not found. Searched: {searched}"
        )

    # Move/Copy to the correct directory based on reviewer's decision
    train_target_dir = dirs[decision]
    was_already_there = file_path.parent.resolve() == train_target_dir.resolve()
    is_in_sorted = sorted_dir.resolve() in file_path.resolve().parents

    if not was_already_there:
        train_target_dir.mkdir(parents=True, exist_ok=True)
        
        if is_in_sorted:
            # 1. Copy to train directory
            shutil.copy(str(file_path), train_target_dir / filename)
            # 2. Move to the corrected sorted_pdfs folder
            sorted_target_dir = sorted_dir / ("Useful" if decision == "useful" else "Not_Useful")
            sorted_target_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(file_path), sorted_target_dir / filename)
            logger.info(f"[REVIEW UNCERTAIN] {filename}: reviewer decided '{decision}' "
                        f"(Moved from {file_path.parent.name}/ to {sorted_target_dir.name}/ AND Copied to {train_target_dir.name}/)")
        else:
            # Just move within train directories (or to_test_files)
            shutil.move(str(file_path), train_target_dir / filename)
            logger.info(f"[REVIEW UNCERTAIN] {filename}: reviewer decided '{decision}' "
                        f"({file_path.parent.name}/ → {train_target_dir.name}/)")
    else:
        logger.info(f"[REVIEW UNCERTAIN] {filename}: reviewer decided '{decision}' "
                    f"(already in {train_target_dir.name}/)")

    # Log to JSONL
    now = datetime.now(timezone.utc)
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)

    log_record = {
        "record_id":      f"{now.strftime('%Y%m%d%H%M%S%f')}_{filename}",
        "filename":       filename,
        "decision":       decision,
        "predicted_score": predicted_score,
        "reviewer":       reviewer,
        "method":         "review_uncertain",  # distinguishes from mark_wrong and submit_feedback
        "apply_status":   "applied",
        "applied_at":     now.isoformat(),
        "created_at":     now.isoformat(),
    }

    with open(FEEDBACK_FILE, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(log_record, ensure_ascii=False) + "\n")

    # Refresh labels.csv
    if update_labels:
        try:
            create_labels(file_type=file_type)
            logger.info(f"  labels.csv updated.")
        except Exception as exc:
            logger.warning(f"  labels.csv update failed: {exc}")

    return {
        "filename":          filename,
        "decision":          decision,
        "placed_in":         train_target_dir.name,
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
    For a simpler one-call correction, use mark_wrong() instead.

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

    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)

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

    with open(FEEDBACK_FILE, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return record
