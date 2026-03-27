"""
Main Pipeline
Orchestrates the PDF/TXT classification workflow
"""

import sys
import numpy as np
import pandas as pd
import logging
import joblib
from pathlib import Path


# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src')) # Ensure src is in path

from sklearn.metrics import accuracy_score # For evaluating model accuracy
from loader import PDFLoader, TXTLoader  # Loaders for PDF and TXT files
from extractor import PDFExtractor, TXTExtractor # Extractors for PDF and TXT files
from preprocess import TextPreprocessor # Text preprocessing utilities
from features import FeatureExtractor # Feature extraction and selection
from utils import save_predictions, load_labels # Utility functions
from project_config import * # Import all necessary configurations
from imblearn.over_sampling import SMOTE # For handling class imbalance, helpful if needed
from semantic import SemanticFeatureExtractor # TODO: Integrate semantic understanding in feature extraction
from model import create_classifier


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__) # Main logger


def main():
    
    # 0. Choose File Type

    """
    File Type Selection: choose txt, if you are working with text files instead of PDFs
    Model Type is now configured in project_config.py (MODEL_TYPE variable)
    It is expandable to more models in the future, currently supports 'random_forest', 'logistic_regression', and 'svm'
    Workflow steps:
    0. Remove previous extracted and preprocessed data to avoid confusion
    1. Load labels from labels.csv
    2. Initialize appropriate loader and extractor based on file type
    3. Extract text from files 
    4. Preprocess text (cleaning, normalization)
    5. Extract features (TF-IDF) and handle class imbalance if needed
    6. Train classifier (supervised) and evaluate on train/test sets
    7. Save results and trained model
    """

    FILE_TYPE = 'txt'  # 'pdf' or 'txt', type with lowercase letters only 

    logger.info("="*60) 
    logger.info(f"Starting {FILE_TYPE.upper()} Classification Pipeline")
    logger.info(f"Model Type (from config): {MODEL_TYPE.upper()}")
    logger.info("="*60)

    logger.info(f"File type selected: {FILE_TYPE.upper()}") # Log selected file type in uppercase
    
    # 1. Load Labels
    logger.info("\n[Step 1] Loading Labels...")
    labels_df = load_labels(LABELS_PATH) # labels.csv path

    if labels_df is None:
        logger.error("Labels file not found. Please create labels.csv in the data directory before running the pipeline.")
        return
    
    # 2. Initialize Loader / Extractor based on file type
    logger.info(f"\n[Step 2] Loading {FILE_TYPE.upper()}...") 

    if FILE_TYPE == 'pdf':
        data_loader = PDFLoader(pdf_dir=RAW_PDFS_DIR, useful_dir=USEFUL_PDFS_DIR)
        extractor_raw = PDFExtractor(output_dir=EXTRACTED_RAW_PDFS_DIR)
        extractor_useful = PDFExtractor(output_dir=EXTRACTED_USEFUL_PDFS_DIR) # Separate extractor for useful PDFs if needed (can be same as raw)   
    elif FILE_TYPE == 'txt':
        data_loader = TXTLoader(txt_dir=RAW_TXTS_DIR, useful_dir=USEFUL_TXTS_DIR)
        extractor_raw = TXTExtractor()
        extractor_useful = TXTExtractor() # Separate extractor for useful TXTs if needed (can be same as raw)
    else:
        logger.error(f"Invalid FILE_TYPE specified. Use {FILE_TYPE}.")
        return

    logger.info(f"{FILE_TYPE.upper()} loader initialized")
    logger.info(f"{FILE_TYPE.upper()} extractor initialized")

    # Get train/test splits (75/25)
    train_files = data_loader.get_files_by_split('train', labels_df)
    test_files = data_loader.get_files_by_split('test', labels_df)

    train_labels = data_loader.get_labels_for_files(train_files, labels_df)
    test_labels = data_loader.get_labels_for_files(test_files, labels_df)

    logger.info(f"\nDataset Split (from labels.csv):")
    logger.info(f"  Training: {len(train_files)} {FILE_TYPE.upper()}s ({sum(train_labels)} useful)")
    logger.info(f"  Testing: {len(test_files)} {FILE_TYPE.upper()}s ({sum(test_labels)} useful)")

    # Analyze class distribution in training set
    useful_count = sum(train_labels)
    not_useful_count = len(train_labels) - useful_count

    # Safety checks
    if len(train_labels) == 0:
        logger.error("No training data found!")
        return

    if useful_count == 0:
        logger.error(f"No useful {FILE_TYPE.upper()} in training set!")
        logger.error("Please run: py src\\label_files.py")
        return

    if not_useful_count == 0:
        logger.error(f"No 'not useful' {FILE_TYPE.upper()} in training set!")
        return

    # Calculate imbalance ratio to decide on handling strategy
    imbalance_ratio = not_useful_count / useful_count  # Now safe

    logger.info(f"\nClass Balance:")
    logger.info(f"   Useful: {useful_count} ({useful_count/len(train_labels)*100:.1f}%)")
    logger.info(f"   Not Useful: {not_useful_count} ({not_useful_count/len(train_labels)*100:.1f}%)")
    logger.info(f"   Imbalance Ratio: 1:{imbalance_ratio:.1f}")

    # Warn if high imbalance detected
    if imbalance_ratio > 5:
        logger.warning("HIGH CLASS IMBALANCE detected!")
        logger.warning(f" Current ratio is 1:{imbalance_ratio:.1f} (Not Useful : Useful)")
        logger.info("Consider collecting more useful samples or applying balancing techniques.")
    
    # 3 Extract texts
    logger.info(f"\n[Step 3] Extracting text from {FILE_TYPE.upper()}s...")

    # Train files separation based on labels
    train_useful_files = [f for f, l in zip(train_files, train_labels) if l == 1]
    train_raw_files = [f for f, l in zip(train_files, train_labels) if l == 0]
    
    # Test files separation based on labels
    test_useful_files = [f for f, l in zip(test_files, test_labels) if l == 1]
    test_raw_files = [f for f, l in zip(test_files, test_labels) if l == 0]

    # Useful files separate extraction (and cache in separate folders if needed for inspection)
    logger.info("Extracting USEFUL files...")
    train_useful_dict = extractor_useful.extract_batch(train_useful_files)
    test_useful_dict = extractor_useful.extract_batch(test_useful_files)
    
    # Raw files separate extraction (and cache in separate folders if needed for inspection)
    logger.info("Extracting RAW files...")
    train_raw_dict = extractor_raw.extract_batch(train_raw_files)
    test_raw_dict = extractor_raw.extract_batch(test_raw_files)

    # Combine useful and raw extracted texts into single dictionaries for train and test
    train_texts_dict = {**train_useful_dict, **train_raw_dict}
    test_texts_dict = {**test_useful_dict, **test_raw_dict}

    logger.info(f"Extracted text from {len(train_texts_dict)} training {FILE_TYPE.upper()}s")
    logger.info(f"Extracted text from {len(test_texts_dict)} testing {FILE_TYPE.upper()}s") 
    
    # Optional: Save extracted texts to cache directories for inspection before preprocessing (especially useful for TXT files where extraction is just reading text, but can be helpful for PDFs to verify extraction quality)
    if FILE_TYPE == 'txt':
        logger.info("\n[Step 3.1] Saving extracted TXTs to cache directories...")
        EXTRACTED_RAW_TXTS_DIR.mkdir(parents=True, exist_ok=True)
        EXTRACTED_USEFUL_TXTS_DIR.mkdir(parents=True, exist_ok=True)
        
        # save extracted useful files
        for file_stem, text in train_useful_dict.items():
            (EXTRACTED_USEFUL_TXTS_DIR / f"{file_stem}.txt").write_text(text, encoding='utf-8')
        for file_stem, text in test_useful_dict.items():
            (EXTRACTED_USEFUL_TXTS_DIR / f"{file_stem}.txt").write_text(text, encoding='utf-8')
            
        # save extracted raw files
        for file_stem, text in train_raw_dict.items():
            (EXTRACTED_RAW_TXTS_DIR / f"{file_stem}.txt").write_text(text, encoding='utf-8')
        for file_stem, text in test_raw_dict.items():
            (EXTRACTED_RAW_TXTS_DIR / f"{file_stem}.txt").write_text(text, encoding='utf-8')
            
        logger.info("Saved original TXT contents to extracted_raw_texts and extracted_useful_texts directories.")

    # Filter out files with empty extracted text
    valid_train_files = []
    valid_train_labels = []
    for file_path, label in zip(train_files, train_labels):
        if file_path.stem in train_texts_dict and train_texts_dict[file_path.stem].strip():
            valid_train_files.append(file_path)
            valid_train_labels.append(label)
        else:
            logger.warning(f"Skipping {file_path.name} - empty or failed extraction")
    
    valid_test_files = []
    valid_test_labels = []
    for file_path, label in zip(test_files, test_labels):
        if file_path.stem in test_texts_dict and test_texts_dict[file_path.stem].strip():
            valid_test_files.append(file_path)
            valid_test_labels.append(label)
        else:
            logger.warning(f"Skipping {file_path.name} - empty or failed extraction")
    
    # Update variables
    train_files = valid_train_files
    train_labels = valid_train_labels
    test_files = valid_test_files
    test_labels = valid_test_labels
    
    logger.info(f"Valid training samples: {len(train_files)} (skipped {len(train_texts_dict) - len(train_files)})")
    logger.info(f"Valid testing samples: {len(test_files)} (skipped {len(test_texts_dict) - len(test_files)})")
    
    # 4 Preprocess texts
    logger.info(f"\n[Step 4] Preprocessing text from {FILE_TYPE.upper()}s...")

    preprocessor = TextPreprocessor()

    train_texts_clean = preprocessor.preprocess_batch( # preprocess train texts
        [train_texts_dict[f.stem] for f in train_files],  # Use only valid files
        filenames=[f.stem for f in train_files]
        )
    
    test_texts_clean = preprocessor.preprocess_batch( # preprocess test texts
        [test_texts_dict[f.stem] for f in test_files],  # Use only valid files
        filenames=[f.stem for f in test_files]
        )
    
    logger.info(f"Preprocessed {len(train_texts_clean)} training texts")
    logger.info(f"Preprocessed {len(test_texts_clean)} testing texts")

    # Save preprocessed texts for inspection
    logger.info("\n[Step 4.1] Saving sample preprocessed texts...")
    clean_useful_dir = PREPROCESSED_USEFUL_TEXTS_DIR
    clean_raw_dir = PREPROCESSED_RAW_TEXTS_DIR

    clean_useful_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists
    clean_raw_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists

    # Save preprocessed texts in separate directories based on their labels for easier inspection (useful vs not useful)
    for split_name, files, texts, labels in [
        ('train', train_files, train_texts_clean, train_labels),
        ('test', test_files, test_texts_clean, test_labels)
    ]:   
         # Save preprocessed texts in separate directories based on their labels for easier inspection (useful vs not useful)
        for file_id, clean_text, label in zip(files, texts, labels):
            # if label is 1 (useful), save in useful directory, else save in raw directory
            target_dir = clean_useful_dir if label == 1 else clean_raw_dir
            
            clean_path = target_dir / f"{split_name}_{file_id.stem}_clean.txt"
            
            with open(clean_path, 'w', encoding='utf-8') as f:
                f.write(clean_text)

    logger.info(f"Saved {len(train_texts_clean)} preprocessed training texts (split into separate directories)")
    logger.info(f"Saved {len(test_texts_clean)} preprocessed testing texts (split into separate directories)")

    # Create a comparison report
    logger.info("\n[Step 4.2] Generating preprocessing comparison report...")
    report_path = RESULTS_DIR / 'preprocessing_report.txt' 
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, 'w', encoding='utf-8') as report_file: # Open report file
        report_file.write("="*60 + "\n") # Header 
        report_file.write("Preprocessing Comparison Report\n") # Title
        report_file.write("="*60 + "\n\n")

        for i, (file_id, raw_text, clean_text) in enumerate(zip( 
            train_files[:5], # Only first 5 samples
            list(train_texts_dict.values())[:5], # Only first 5 samples
            train_texts_clean[:5] # Only first 5 samples
        ), 1): # 1 is for starting index
            report_file.write(f"\n{'='*60}\n") # Section separator
            report_file.write(f"File #{i}: {file_id.stem}{file_id.suffix}\n") # Write filename with extension
            report_file.write(f"{'='*60}\n\n")
            report_file.write(f"Original length: {len(raw_text)} characters\n") # Original text length
            report_file.write(f"Cleaned (preprocessed) length: {len(clean_text)} characters\n\n") # Cleaned text length
            report_file.write(f"Reduction in size: {(1 - len(clean_text)/len(raw_text))*100:.1f}%\n\n") # Reduction percentage
            report_file.write("----- Original Text (First 500 chars) -----\n")
            report_file.write(raw_text[:500] + "\n\n")
            report_file.write("----- Preprocessed Text (First 500 chars) -----\n")
            report_file.write(clean_text[:500] + "\n\n")
            report_file.write("\n" + ("="*60) + "\n\n")
            # Removed keywords
            removed_words = []
            if 'education' in raw_text.lower() and 'education' not in clean_text.lower():
                removed_words.append('education')
            if 'university' in raw_text.lower() and 'university' not in clean_text.lower():
                removed_words.append('university')
            if "studying" in raw_text.lower() and "studying" not in clean_text.lower():
                removed_words.append('studying')
            if 'author' in raw_text.lower() and 'author' not in clean_text.lower():
                removed_words.append('author')

            if removed_words:
                report_file.write(f"Removed sensitive (noise) keywords: {', '.join(removed_words)}\n")
            else:
                report_file.write("No sensitive (noise) keywords removed.\n")

    logger.info(f"Preprocessing comparison report saved to {report_path}")

    # 5. Extract features and handle class imbalance
    logger.info("\n[Step 5.1] Extracting TF-IDF features...")
    feature_extractor = FeatureExtractor(max_features=MAX_FEATURES) 
    X_train = feature_extractor.fit_transform(train_texts_clean)
    X_test = feature_extractor.transform(test_texts_clean)

    logger.info(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

    # TODO: Integrate semantic understanding in this part

    # fix workflow
    y_train = np.array(train_labels)  # convert to numpy array
    SMOTE_applied = False  # Flag 

 # Apply SMOTE if severe class imbalance detected 
    if imbalance_ratio > SMOTE_THRESHOLD: 
        logger.info("\n[Step 5.2] Applying SMOTE to balance classes...")
        try:
            k_neighbors = min(5, useful_count - 1)
        
            if k_neighbors < 1:
                logger.warning("Not enough minority samples for SMOTE (need at least 2). Skipping.")
            else:
                smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
                X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            
                logger.info(f"   Before SMOTE: Useful={useful_count}, Not Useful={not_useful_count}")
                logger.info(f"   After SMOTE: Useful={sum(y_train_resampled)}, Not Useful={len(y_train_resampled) - sum(y_train_resampled)}")
                logger.info(f"   Total samples: {len(y_train_resampled)}")
            
                # Update training data
                X_train = X_train_resampled
                y_train = y_train_resampled
                SMOTE_applied = True

        except ImportError:
            logger.warning("imbalanced-learn not installed. Skipping SMOTE.")
        except ValueError as e: 
            logger.warning(f"SMOTE failed: {e}. Using class_weight='balanced' only.")
    else:
        logger.info("\n[Step 5.2] Skipping SMOTE (imbalance ratio below threshold)")

    logger.info("\n[Step 5.3] Feature Selection (Chi-Squared)...")
    original_features = X_train.shape[1] # number of features
    X_train = feature_extractor.select_best_features(X_train, y_train, k=1000)

    if hasattr(feature_extractor, 'selector') and feature_extractor.selector is not None:
        # Apply selector to filter test set
        support_mask = feature_extractor.selector.get_support()
        X_test = X_test[:, support_mask]
        selected_features = X_train.shape[1]
        logger.info(f"Applied feature selection to test set using selector")
    else:
        # If selector not found, manually reduce to same number of features
        logger.warning("Feature selector not found. Manually selecting first k features.")
        X_test = X_test[:, :X_train.shape[1]]
        selected_features = X_train.shape[1]

    logger.info(f"Reduced features: {original_features} -> {selected_features}")
    logger.info(f"Feature reduction: {(1 - selected_features/original_features)*100:.1f}%")
    logger.info(f"Final shapes - X_train: {X_train.shape}, X_test: {X_test.shape}")


    logger.info("\n[Step 5.4] Analyzing feature importance on selected features...")
    feature_extractor.get_top_features(X_train, y_train, top_n=50)
    
    logger.info("\n[Step 5.5] Initializing Classifier...")
    
    # No need to pass model_type, it reads from config automatically
    classifier = create_classifier(mode='supervised', random_state=42)

    logger.info("\n[Step 5.6] Cross-Validation (5-fold)...")
    cv_scores = classifier.cross_validate(X_train, y_train, cv=5) # 5-fold CV means splitting the data into 5 parts and training/testing 5 times

    logger.info(f"\nCV Results:")
    logger.info(f"  Mean F1: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    logger.info(f"  Scores: {cv_scores}")

    # Decision based on CV (cross-validation) results
    if cv_scores.mean() < 0.5: # Threshold for acceptable performance
        logger.warning("POOR PERFORMANCE! Consider:") 
        logger.warning("1. Collect more useful samples")
        logger.warning("2. Check data quality")
        logger.warning("3. Tune hyperparameters")
    else:
        logger.info("CV performance acceptable. Proceeding to training.")
    
    
    # 6. Train classifier (SUPERVISED)
    logger.info("\n[Step 6] Training Supervised Classifier...")
    classifier.train(X_train, y_train)
    
    

    logger.info("\n[Step 6.1] Analyzing Top Features...")

    if hasattr(classifier.model, 'feature_importances_'):  # Random Forest
        feature_names = feature_extractor.get_feature_names()
        importances = classifier.model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
    
        logger.info("\nTop 10 Most Important Features (by importance):")
        for i, idx in enumerate(indices, 1):
            logger.info(f"  {i}. '{feature_names[idx]}': {importances[idx]:.4f}")

    elif hasattr(classifier.model, 'coef_'):  # Logistic Regression & SVM
        feature_names = feature_extractor.get_feature_names()
        coef = classifier.model.coef_
    
    # FIX: Convert sparse matrix to dense array
        if hasattr(coef, "toarray"):
            coef = coef.toarray()
        coef = np.asarray(coef).ravel()  # Flatten to 1D array
    
        indices = np.argsort(np.abs(coef))[::-1][:10]
    
        logger.info("\nTop 10 Most Important Features (by coefficient magnitude):")
        for i, idx in enumerate(indices, 1):
            logger.info(f"  {i}. '{feature_names[idx]}': {coef[idx]:+.4f}")
    else:
        logger.warning("Model does not support feature importance analysis")

    logger.info("\n[Step 6.2] Evaluating on TRAINING set...")
    train_predictions, _ = classifier.evaluate(X_train, y_train)
    train_acc = accuracy_score(y_train, train_predictions)
    logger.info(f"Training Accuracy: {train_acc:.2%}")

    logger.info("\n[Step 6.3] Evaluating on TESTING set...")
    

    if SMOTE_applied:
        logger.warning("Note: Test set is NOT balanced (original distribution preserved)")
        logger.info(f"Test set: {sum(test_labels)} useful, {len(test_labels) - sum(test_labels)} not_useful")

    
    test_predictions, test_scores = classifier.evaluate(X_test, np.array(test_labels))
    test_acc = accuracy_score(test_labels, test_predictions)
    logger.info(f"Testing Accuracy: {test_acc:.2%}")

    # Overfit Check (compare train/test accuracy)
    if train_acc - test_acc > 0.15:
        logger.warning("POSSIBLE OVERFITTING DETECTED!")
        logger.warning(f" Train Acc: {train_acc:.2%} vs Test Acc: {test_acc:.2%}")
        logger.info("Consider collecting more data or applying regularization.")
        logger.info("You may also tune hyperparameters or use simpler models.")
        logger.info("It may work better with reduces max_depth, increased min_samples_split, or using ensemble methods.")
    elif test_acc >= train_acc:
        logger.info("Great! No overfitting detected.")
    else:
        logger.info(f"Train-test gap {(train_acc - test_acc)*100:.1f}% acceptable.")

    # 7. Save results
    logger.info("\n[Step 7] Saving results...")
    test_filenames = [f.name for f in test_files]
    results_df = save_predictions(test_filenames, test_predictions, test_scores)
    results_df_path = RESULTS_DIR / f'{FILE_TYPE}_{MODEL_TYPE}_test_results.csv'
    results_df.to_csv(results_df_path, index=False)
    logger.info(f"Test results saved to {results_df_path}")
    
    # 8. Save trained model
    classifier.save_model(RESULTS_DIR / f'{FILE_TYPE}_{MODEL_TYPE}_classifier.joblib')

    
    return results_df

if __name__ == "__main__":
    main()
