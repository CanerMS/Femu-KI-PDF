import json
import sys
from pathlib import Path

import pytest

base_dir = Path(__file__).parent.parent
sys.path.insert(0, str(base_dir))
sys.path.insert(0, str(base_dir / "src"))

import feedback

def test_submit_feedback_writes_record(tmp_path, monkeypatch):
    feedback_dir = tmp_path / "feedback"
    feedback_file = feedback_dir / "feedback.jsonl"


    monkeypatch.setattr(feedback, "FEEDBACK_DIR", feedback_dir)
    monkeypatch.setattr(feedback, "FEEDBACK_FILE", feedback_file)
    monkeypatch.setattr(feedback, "VALID_LABELS", {"useful", "not_useful"})

    record = feedback.submit_feedback(
        filename = "0001.txt",
        predicted_label = "not_useful",
        predicted_score = 0.11,
        human_label = "useful",
        annotator = "Anna",
        notes = "",
        model_version = "m1",
        tfidf_version = "t1",
        scaler_version = "s1"
    )

    assert feedback_file.exists()
    saved = json.loads(feedback_file.read_text(encoding="utf-8").strip())
    assert saved["filename"] == "0001.txt"
    assert saved["annotator"] == "Anna"
    assert saved["pipeline_version"] == "model=m1 | tfidf=t1 | scaler=s1" 
    assert record["filename"] == "0001.txt" # check if the record was successfull

def test_submit_feedback_rejects_empty_filename(tmp_path, monkeypatch):
    monkeypatch.setattr(feedback, "FEEDBACK_DIR", tmp_path / "feedback")
    monkeypatch.setattr(feedback, "FEEDBACK_FILE", tmp_path / "feedback" / "feedback.jsonl")
    monkeypatch.setattr(feedback, "VALID_LABELS", {"useful", "not_useful"})
                         
    with pytest.raises(ValueError, match= "Filename cannot be empty"):
        feedback.submit_feedback(
            filename="",
            predicted_label="useful",
            predicted_score="0.5",
            human_label="useful"
        )
