import sys
import os
import shutil  # For carrying the files
from pathlib import Path
import logging
import joblib
import numpy as np
import scipy.sparse as sp
import warnings

# Silent Mode Settings
SILENT_MODE = True # If this is true, only the score will show up

if SILENT_MODE:
    # 1. Turn off every warnings
    warnings.filterwarnings("ignore")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # set HF Transformers logging level as ERROR
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    # 2. For the main file, hide all of the warnings except critical errors
    logging.basicConfig(level=logging.CRITICAL)
    logger = logging.getLogger(__name__)

    # 3. Hide the errors coming from other models
    for log_name in ["transformers", "httpx", "extractor", "features", "model", "preprocess", "huggingface_hub.utils._http"]:
        logging.getLogger(log_name).setLevel(logging.CRITICAL)

else:
    original_stdout = sys.stdout  # always available for LISA output
    # If silent mode is deactivated
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))

import lisa
from project_config import FEATURE_MODE, LOGS_DIR, RAW_TXTS_DIR, USEFUL_TXTS_DIR, RAW_PDFS_DIR, USEFUL_PDFS_DIR, MANUAL_CHECK_DIR
from extractor import PDFExtractor, TXTExtractor
from preprocess import TextPreprocessor
from semantic import SciBERTSemanticFeatureExtractor
from model import LogisticRegressionClassifier


def find_latest_model_set(results_dir: Path, file_type: str, model_type: str):
    """
    Find the latest model set by sorting filenames lexicographically.
    """
    models = sorted(results_dir.glob(f"{file_type}_{model_type}_*_classifier.joblib"))  # find the matching classifier.joblib files and sort them

    if not models: return None, None, None

    latest_model = models[-1]  # asc to desc, so last one is the latest

    suffix = latest_model.stem[len(f"{file_type}_{model_type}_"):].removesuffix("_classifier")

    return (
        latest_model,
        results_dir / f"{file_type}_tfidf_vocabulary_{suffix}.joblib",
        results_dir / f"{file_type}_scaler_{suffix}.joblib"
    )


def main():
    LISA_MODE = os.environ.get("LISA", "") != ""

    if LISA_MODE:
        original_stdout = sys.stdout
        sys.stdout = (LOGS_DIR / "predict_out.log").open("w", encoding="utf-8")
        sys.stderr = (LOGS_DIR / "predict_err.log").open("w", encoding="utf-8")

    # Which model would you like to use?
    MODEL_PATH = "results/txt_logistic_regression_93_1_classifier.joblib"
    FILE_TYPE  = 'txt'
    RESULTS_DIR = Path(__file__).parent.parent / "results"  # absolute path from project root

    MODEL_PATH, TFIDF_DICT_PATH, SCALER_PATH = find_latest_model_set(
        RESULTS_DIR, FILE_TYPE, "logistic_regression"
    )

    if MODEL_PATH is None:
        logger.error("Error: No trained model found!")
        sys.exit(1)

    logger.info(f"Auto selected model: {MODEL_PATH.name}")

    CONFIDENCE_THRESHOLD = 75.0  # Human in the loop threshold
    FLAG_EX = False  # for more explanations

    # 2. Arrange the files
    TARGET_DIR = Path("data/to_test_files")  # Which pdf/txt would you like to test?
    SORTED_DIR = Path("data/sorted_pdfs")

    # Create aim directory
    DIR_USEFUL = SORTED_DIR / "Useful"
    DIR_NOT_USEFUL = SORTED_DIR / "Not_Useful"
    DIR_MANUAL_CHECK = SORTED_DIR / "Manual_Check"

    for d in [DIR_USEFUL, DIR_NOT_USEFUL, DIR_MANUAL_CHECK]:
        d.mkdir(parents=True, exist_ok=True)

    # Clean up and recreate TARGET_DIR so LISA gets fresh files
    if LISA_MODE and os.path.exists(TARGET_DIR):
        shutil.rmtree(TARGET_DIR)
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    if LISA_MODE:
        lisa_items = lisa.write_items_to_directory(lisa.read_input(sys.stdin), TARGET_DIR)

    logger.info(f"Uploaded Model: {MODEL_PATH}")

    # 3. Load Model and tools
    model = LogisticRegressionClassifier()
    try:
        model.load_model(MODEL_PATH)
    except FileNotFoundError:
        logger.error("Error: Trained model could not be found!")
        sys.exit(1)

    # Upload TF-IDF Dict
    tfidf_extractor = None
    if FEATURE_MODE in ['tfidf', 'combined']:
        try:
            tfidf_extractor = joblib.load(TFIDF_DICT_PATH)
            logger.info(f"TF-IDF Dict uploaded successfully: {TFIDF_DICT_PATH}")
        except FileNotFoundError:
            logger.error(f"Critical Error: Feature_Mode='{FEATURE_MODE} but TF-IDF doesn't exist")
            sys.exit(1)

    scaler = None
    if FEATURE_MODE == 'combined':
        try:
            scaler = joblib.load(SCALER_PATH)
            logger.info(f"Scaler uploaded successfully: {SCALER_PATH}")
        except FileNotFoundError:
            logger.error(f"Critical Error: Feature_Mode='{FEATURE_MODE}' but Scaler doesn't exist")
            sys.exit(1)

    preprocessor = TextPreprocessor()
    semantic_extractor = SciBERTSemanticFeatureExtractor()

    if FILE_TYPE == 'pdf':
        extractor = PDFExtractor(output_dir=Path("data/temp_extract"))
    else:
        extractor = TXTExtractor()

    files = list(TARGET_DIR.glob(f"*.{FILE_TYPE}"))
    if not files:
        logger.warning(f"No {FILE_TYPE} in: {TARGET_DIR}")

    logger.info(f"In total {len(files)} will be predicted\n")

    # 4. Predict and divide to directories
    for file_path in files:
        logger.info(f"{file_path.name} checking...")

        # Extract the text
        if FILE_TYPE == 'pdf':
            text = extractor.extract_text_from_pdf(file_path)
        else:
            text = extractor.extract_text(file_path)

        clean_text = preprocessor.clean_text(text)

        # If the clean text is empty, we cannot make a prediction, skip the file
        if not clean_text.strip():
            logger.warning(f"Skipping {file_path.name} because clean text is empty")

            continue

        final_vector = None
        if FEATURE_MODE == 'semantic':
            final_vector = semantic_extractor.extract_embeddings([clean_text], [file_path.stem])
        elif FEATURE_MODE == 'tfidf':
            final_vector = tfidf_extractor.transform([clean_text])
        elif FEATURE_MODE == 'combined':
            sem_vec = semantic_extractor.extract_embeddings([clean_text], [file_path.stem])
            tf_vec = tfidf_extractor.transform([clean_text])

            if sp.issparse(tf_vec):
                final_vector = sp.hstack((tf_vec, sp.csr_matrix(sem_vec)))
            else:
                final_vector = np.hstack((tf_vec, sem_vec))

            # Scale before predicting
            if scaler is not None:
                final_vector = scaler.transform(final_vector)

        # Prediction and confident rate
        try:
            prediction = model.predict(final_vector)[0]
            score = model.predict_scores(final_vector)[0]
        except ValueError as e:
            logger.error(f"Convergence error: {e}")

            continue

        result = "USEFUL" if prediction == 1 else "NOT USEFUL"
        uncertain = score_percentage < CONFIDENCE_THRESHOLD

        if prediction == 1:
            score_percentage = score * 100

            if LISA_MODE:
                lisa_items[file_path.name] = lisa.OutputItem(
                    relevant=True,
                    score=score_percentage,
                    uncertain=uncertain,
                )
        else:
            score_percentage = (1.0 - score) * 100

            if LISA_MODE:
                lisa_items[file_path.name] = lisa.OutputItem(
                    relevant=False,
                    score=score_percentage,
                    uncertain=uncertain,
                )
        # if not enough confident
        if uncertain:
            aim_directory = DIR_MANUAL_CHECK
            explanation = "Human control needed"
        elif result == "USEFUL":
            aim_directory = DIR_USEFUL
            explanation = "Overconfident!"
        else:
            aim_directory = DIR_NOT_USEFUL
            explanation = "Overconfident, archive!"

        # Carry the file
        try:
            shutil.move(str(file_path), str(aim_directory / file_path.name))

            if not SILENT_MODE:
                logger.info(f" -> Decision: {result} (%{score_percentage:.1f} Trust) | {explanation}\n")

        except Exception as e:
            if not SILENT_MODE:
                logger.error(f" -> Error while {file_path.name} was carried: {e}\n")

    if LISA_MODE:
        lisa.write_output(list(lisa_items.values()), original_stdout)

    logger.info("You can see the results in 'data/sorted_pdfs' .")


if __name__ == "__main__":
    main()
