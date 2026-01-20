"""
Main Pipeline
Orchestrates the PDF/TXT classification workflow
"""

import sys
import numpy as np
import pandas as pd
import logging
from pathlib import Path


# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src')) # Ensure src is in path

from sklearn.metrics import accuracy_score # For evaluating model accuracy
from loader import PDFLoader, TXTLoader  # Loaders for PDF and TXT files
from extractor import PDFExtractor, TXTExtractor # Extractors for PDF and TXT files
from preprocess import TextPreprocessor # Text preprocessing utilities
from features import FeatureExtractor # Feature extraction and selection
from model import PDFClassifier # Classifier model
from utils import save_predictions, load_labels # Utility functions
from project_config import * # Import all necessary configurations
from imblearn.over_sampling import SMOTE # For handling class imbalance, helpful if needed

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__) # Main logger

def main():
    logger.info("="*60) 
    logger.info("Starting PDF/TXT Classification Pipeline")
    logger.info("="*60)

    # 0. Choose File Type

    """
    File Type Selection: choose txt, if you are working with text files instead of PDFs
    """

    FILE_TYPE = 'pdf'  # 'pdf' or 'txt', type with lovercase letters only 
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
        extractor = PDFExtractor()
    elif FILE_TYPE == 'txt':
        data_loader = TXTLoader(txt_dir=RAW_TXTS_DIR, useful_dir=USEFUL_TXTS_DIR)
        extractor = TXTExtractor()
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
        logger.error("No useful PDFs in training set!")
        logger.error("Please run: py src\\label_pdfs.py")
        return

    if not_useful_count == 0:
        logger.error("No 'not useful' PDFs in training set!")
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
    
    # 3 Extract 
    logger.info(f"\n[Step 3] Extracting text from {FILE_TYPE.upper()}s...")

    train_texts_dict = extractor.extract_batch(train_files)
    test_texts_dict = extractor.extract_batch(test_files)

    logger.info(f"Extracted text from {len(train_texts_dict)} training {FILE_TYPE.upper()}s")
    logger.info(f"Extracted text from {len(test_texts_dict)} testing {FILE_TYPE.upper()}s")  
    
    
    # 4 Preprocess texts
    logger.info(f"\n[Step 4] Preprocessing text from {FILE_TYPE.upper()}s...")

    preprocessor = TextPreprocessor()

    train_texts_clean = preprocessor.preprocess_batch(
        list(train_texts_dict.values()),
        filenames=[f.stem for f in train_files]
        )
    
    test_texts_clean = preprocessor.preprocess_batch(
        list(test_texts_dict.values()),
        filenames=[f.stem for f in test_files]
        )
    
    logger.info(f"Preprocessed {len(train_texts_clean)} training texts")
    logger.info(f"Preprocessed {len(test_texts_clean)} testing texts")

    # Save preprocessed texts for inspection
    logger.info("\n[Step 4.1] Saving sample preprocessed texts...")
    clean_dir = PREPROCESSED_TEXTS_DIR
    clean_dir.mkdir(parents=True, exist_ok=True)

    for split_name, files, texts in [
        ('train', train_files, train_texts_clean),
        ('test', test_files, test_texts_clean)
    ]:   
         # Save the train preprocessed texts
        for file_id, clean_text in zip(files, texts):
            clean_path = clean_dir / f"{split_name}_{file_id.stem}_clean.txt"
            with open(clean_path, 'w', encoding='utf-8') as f:
                f.write(clean_text)

    logger.info(f"Saved {len(train_texts_clean)} preprocessed training texts to {clean_dir}")
    logger.info(f"Saved {len(test_texts_clean)} preprocessed testing texts to {clean_dir}")

    # Create a comparison report
    logger.info("\n[Step 4.2] Generating preprocessing comparison report...")
    report_path = RESULTS_DIR / 'preprocessing_report.txt'
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, 'w', encoding='utf-8') as report_file:
        report_file.write("="*60 + "\n")
        report_file.write("Preprocessing Comparison Report\n")
        report_file.write("="*60 + "\n\n")

        for i, (file_id, raw_text, clean_text) in enumerate(zip(
            train_files[:5],
            list(train_texts_dict.values())[:5],
            train_texts_clean[:5]
        ), 1): # 1 is for starting index
            report_file.write(f"\n{'='*60}\n")
            report_file.write(f"PDF #{i}: {file_id.stem}.pdf\n")
            report_file.write(f"{'='*60}\n\n")

            report_file.write(f"Original length: {len(raw_text)} characters\n")
            report_file.write(f"Cleaned (preprocessed) length: {len(clean_text)} characters\n\n")
            report_file.write(f"Reduction in size: {(1 - len(clean_text)/len(raw_text))*100:.1f}%\n\n")
            report_file.write("----- Original Text (First 500 chars) -----\n")
            report_file.write(raw_text[:500] + "\n\n")
            report_file.write("----- Preprocessed Text (First 500 chars) -----\n")
            report_file.write(clean_text[:500] + "\n")
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
    classifier = PDFClassifier(mode='supervised', random_state=42)

    logger.info("\n[Step 5.6] Cross-Validation (5-fold)...")
    cv_scores = classifier.cross_validate(X_train, y_train, cv=5)

    logger.info(f"\nCV Results:")
    logger.info(f"  Mean F1: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    logger.info(f"  Scores: {cv_scores}")

    # Decision based on CV
    if cv_scores.mean() < 0.5:
        logger.warning("POOR CV PERFORMANCE! Consider:")
        logger.warning("1. Collect more useful samples")
        logger.warning("2. Check data quality")
        logger.warning("3. Tune hyperparameters")
    else:
        logger.info("CV performance acceptable. Proceeding to training.")
    
    
    # 6. Train classifier (SUPERVISED)
    logger.info("\n[Step 6] Training Supervised Classifier...")
    classifier.train(X_train, y_train)
    
    logger.info("\n[Step 6.1] Analyzing Top Features...")

    if classifier.mode == 'supervised':
        feature_names = feature_extractor.get_feature_names()
        importances = classifier.model.feature_importances_
        # Get top 10 features
        indices = np.argsort(importances)[::-1][:10]
        
        logger.info("\nTop 10 Most Important Features:")
        for i, idx in enumerate(indices, 1):
            logger.info(f"  {i}. '{feature_names[idx]}': {importances[idx]:.4f}")

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
    
    logger.info("="*60)
    logger.info("PIPELINE COMPLETE!")
    logger.info("="*60)
    
    classifier.save_model(RESULTS_DIR / f'{FILE_TYPE}_classifier.joblib') 
    
    return results_df

if __name__ == "__main__":
    main()
