# PDF Classification System - Supervised Learning

A machine learning system for classifying PDF documents as "useful" or "not useful" using supervised learning with Random Forest and intelligent text extraction caching.

---

## 📊 **Project Status**

| Component            | Status      |          Performance            |
|----------------------|-------------|---------------------------------|
| **Pipeline**         | ✅ Complete | Fully operational               |
| **Text Extraction**  | ✅ Complete | 333/333 PDFs processed          |
| **Model Training**   | ✅ Complete | Random Forest trained           |
| **Caching System**   | ✅ Complete | 36x speedup                     |
| **Overall Accuracy** | ⚠️ 88%      | Not useful: 100%, Useful: 9%    |
| **Production Ready** | ⚠️ No       | Data quality improvement needed |

---

## 🎯 **Current Results**

### **Performance Metrics (Test Set: 84 PDFs)**

```
Classification Report:
              precision    recall  f1-score   support

  not_useful       0.88      1.00      0.94        73
      useful       1.00      0.09      0.17        11

    accuracy                           0.88        84
```

**Confusion Matrix:**
```
                Actual
              Not_U  Useful
Predicted     ─────────────
Not_Useful │    73      10  │
Useful     │     0       1  │
```

**Key Findings:**
- Excellent at identifying "not useful" PDFs (100% recall)
- Poor at identifying "useful" PDFs (9% recall - only 1/11 detected)
- **Issue identified:** Data quality and labeling criteria need refinement

---

## **Features**

### **Implemented**
- **Supervised Learning Pipeline**
  - Random Forest classifier (100 estimators)
  - SMOTE for class imbalance handling (32→217 useful samples)
  - Class weights for minority class emphasis
  
- **Intelligent Text Extraction**
  - Automatic caching system (reduces 3min to 5sec on re-runs)
  - Fallback mechanism: pdfplumber → PyPDF2
  - Progress tracking with statistics
  
- **Feature Engineering**
  - TF-IDF vectorization (2000 features)
  - Bigram support (1-2 word phrases)
  - English stop words removal
  
- **Automated Workflows**
  - Stratified train/test split (75%/25%)
  - Automated PDF labeling based on directory structure
  - Model persistence (save/load)
  - Comprehensive logging

- **Performance Analysis**
  - Confusion matrix
  - Classification report
  - Feature importance analysis
  - Per-class metrics

---

## **Project Structure**

```
Femu-KI-PDF/
├── data/
│   ├── raw_pdfs/              # "Not useful" PDFs (290 files)
│   ├── useful_pdfs/           # "Useful" PDFs (43 files)
│   ├── extracted_texts/       # Cached text extractions (333 files)
│   └── labels.csv             # Training labels (333 entries)
├── src/
│   ├── project_config.py      # Centralized configuration
│   ├── loader.py              # PDF loading with label integration
│   ├── extractor.py           # Text extraction with caching
│   ├── preprocess.py          # Text cleaning and normalization
│   ├── features.py            # TF-IDF feature extraction
│   ├── model.py               # Random Forest classifier
│   ├── utils.py               # Helper functions
│   └── label_pdfs.py          # Automated labeling system
├── results/
│   ├── predictions.csv        # Test set predictions
│   └── pdf_classifier.joblib  # Trained model
├── logs/
│   └── label_pdfs.log         # Labeling process logs
├── main.py                    # Main pipeline orchestrator
├── requirements.txt
├── .gitignore
└── README.md
```

---

## **Installation**

### **Prerequisites**
- Python 3.8+
- pip package manager

### **Setup**

```bash
# 1. Clone repository
git clone <repository-url>
cd Femu-KI-PDF

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify installation
python -c "import sklearn, pdfplumber, pandas; print('All dependencies installed')"
```

### **Dependencies**
```
pdfplumber>=0.9.0          # Primary PDF text extraction
PyPDF2>=3.0.0              # Fallback PDF extraction
scikit-learn>=1.3.0        # Machine learning
imbalanced-learn>=0.11.0   # SMOTE for class balancing
pandas>=2.0.0              # Data manipulation
numpy>=1.24.0              # Numerical operations
```

---

## 📖 **Usage**

### **Quick Start**

```bash
# Complete pipeline (labeling + training + evaluation)
python main.py
```

### **Step-by-Step**

#### **Step 1: Prepare Your Data**

Place PDFs in appropriate directories:
```bash
data/
├── raw_pdfs/       # Put "not useful" PDFs here
└── useful_pdfs/    # Put "useful" PDFs here
```

#### **Step 2: Create Labels**

```bash
python src/label_pdfs.py
```

**Output:**
```
Labels saved to data/labels.csv
Dataset Summary:
  Total: 333 PDFs
  Training: 249 (74.8%)
  Testing: 84 (25.2%)

Class Distribution:
  train - not_useful: 217
  train - useful: 32
  test - not_useful: 73
  test - useful: 11
```

#### **Step 3: Train and Evaluate**

```bash
python main.py
```

**Pipeline Stages:**
1. Load labels and PDFs
2. Extract text (with caching)
3. Preprocess and clean text
4. Extract TF-IDF features (2000 features)
5. Apply SMOTE balancing (32→217)
6. Train Random Forest classifier
7. Evaluate on test set
8. Save model and predictions

**Expected Runtime:**
- First run: ~3-4 minutes (text extraction)
- Subsequent runs: ~30-60 seconds (using cache)

---

## **Configuration**

Edit `src/project_config.py` to customize:

```python
# Directories
RAW_PDFS_DIR = Path("data/raw_pdfs")
USEFUL_PDFS_DIR = Path("data/useful_pdfs")
EXTRACTED_TEXTS_DIR = Path("data/extracted_texts")
RESULTS_DIR = Path("results")

# Model Hyperparameters
N_ESTIMATORS = 100          # Number of trees in Random Forest
MAX_DEPTH = 10              # Maximum tree depth
MIN_SAMPLES_SPLIT = 5       # Minimum samples to split node
RANDOM_STATE = 42           # Reproducibility seed

# Feature Extraction
MAX_FEATURES = 2000         # Maximum TF-IDF features
NGRAM_RANGE = (1, 2)        # Unigrams and bigrams

# Data Split
TEST_SIZE = 0.25            # 25% for testing
SMOTE_THRESHOLD = 3.0       # Apply SMOTE if imbalance ratio > 3
```

---

## **Current Issues & Next Steps**

### **Known Issues**

1. **Low Minority Class Recall (9%)**
   - Model only detected 1 out of 11 "useful" PDFs in test set
   - Root cause: Data quality and labeling criteria issues

2. **Feature Overlap**
   - Top features (electrical, electromagnetic, diameter) appear in both classes
   - No distinct features for "useful" class identified

3. **Insufficient Training Data**
   - Only 32 "useful" samples in training set
   - Minimum recommended: 100-150 samples

### **Action Plan**

#### **Phase 1: Data Quality Improvement (Priority: HIGH)**

**Week 1:**
- [ ] Define clear labeling criteria with supervisor
  - What makes a PDF "useful"?
  - Specific format/content requirements?
  - Domain-specific terminology?

- [ ] Manual review of existing labels
  - Verify 43 "useful" PDFs are correctly labeled
  - Check for mislabeled PDFs in "not useful" set
  - Document decision criteria

- [ ] Expand "useful" dataset
  - Target: 100-150 "useful" PDFs
  - Maintain consistent labeling criteria
  - Ensure diversity in useful examples

#### **Phase 2: Model Optimization (Priority: MEDIUM)**

**Week 2:**
- [ ] Hyperparameter tuning
  ```python
  N_ESTIMATORS = 300      # 100 → 300
  MAX_DEPTH = 20          # 10 → 20
  MAX_FEATURES = 5000     # 2000 → 5000
  ```

- [ ] Adjust class weights
  ```python
  class_weight={0: 1, 1: 10}  # Give 10x weight to "useful"
  ```

- [ ] Experiment with n-grams
  ```python
  NGRAM_RANGE = (1, 3)    # Add trigrams
  ```

#### **Phase 3: Alternative Approaches (Priority: LOW)**

- [ ] Try Gradient Boosting Classifier
- [ ] Implement cross-validation
- [ ] Test transfer learning approaches
- [ ] Consider multi-class classification

### **Expected Outcomes After Improvements**

| Metric | Current | Target | Minimum Acceptable |
|--------|---------|--------|--------------------|
| **Useful Recall** | 9% | 70%+ | 50%+ |
| **Useful Precision** | 100% | 60%+ | 40%+ |
| **Useful F1-Score** | 0.17 | 0.65+ | 0.45+ |
| **Overall Accuracy** | 88% | 92%+ | 88%+ |

---

## **Model Details**

### **Architecture**
- **Algorithm:** Random Forest Classifier
- **Estimators:** 100 trees
- **Max Depth:** 10
- **Class Weight:** Balanced (auto-adjusted for imbalance)
- **SMOTE:** Applied when imbalance ratio > 3.0

### **Training Process**
```
Original Training Data:
  - Not useful: 217 samples
  - Useful: 32 samples
  - Imbalance ratio: 1:6.8

After SMOTE:
  - Not useful: 217 samples
  - Useful: 217 samples (synthetically balanced)
  - Total: 434 samples
```

### **Feature Extraction**
- **Method:** TF-IDF (Term Frequency-Inverse Document Frequency)
- **Features:** 2000 most important terms
- **N-grams:** Unigrams + Bigrams
- **Stop Words:** English common words removed

### **Top 10 Features (Current Model)**
```
1. electrical      (0.0214)
2. diameter        (0.0155)
3. electromagnetic (0.0140)
4. cat             (0.0127)
5. human           (0.0117)
6. stimulation     (0.0104)
7. hz              (0.0104)
8. table1          (0.0103)
9. expression      (0.0101)
10. abstract       (0.0093)
```

**Note:** Feature overlap between classes indicates labeling criteria issues

---

## **Troubleshooting**

### **Issue: "imbalanced-learn not installed"**
```bash
pip install imbalanced-learn
```

### **Issue: "No PDF files found"**
- Ensure PDFs are in `data/raw_pdfs/` and `data/useful_pdfs/`
- Check file extensions (must be `.pdf`)

### **Issue: "Text extraction failed"**
- Some PDFs may be scanned images (OCR required)
- Try with different PDFs first
- Check logs in `data/extracted_texts/`

### **Issue: "Model performance is poor"**
- **This is expected with current dataset**
- Follow the Action Plan above to improve data quality
- Requires manual review and re-labeling

### **Issue: "Slow first run"**
- Normal: Text extraction takes 2-3 minutes for 333 PDFs
- Subsequent runs use cache (5-10 seconds)
- To clear cache: `rmdir /s /q data\extracted_texts`

---

## **Performance Optimization**

### **Caching System**
The intelligent caching system dramatically improves performance:

```
First run:  ~180 seconds (extract 333 PDFs)
Second run: ~5 seconds   (load from cache)
Speedup:    36x faster
```

**How it works:**
1. Extracted text saved to `data/extracted_texts/`
2. Each PDF → one `.txt` file
3. Future runs check cache first
4. Only extract if file missing or empty

**To force re-extraction:**
```bash
# Windows
rmdir /s /q data\extracted_texts
mkdir data\extracted_texts

# Then run:
python main.py
```

---

## **Contributing**

### **Current Priority Tasks**

1. **Data Collection** (Most Important)
   - Collect 70-120 additional "useful" PDFs
   - Document clear labeling criteria
   - Verify existing labels

2. **Experimentation**
   - Test different hyperparameters
   - Try alternative ML algorithms
   - Implement cross-validation

3. **Documentation**
   - Document labeling criteria
   - Add more usage examples
   - Create troubleshooting guide

---
## **License**



---

## 👥 **Contact**

canerrcc1@gmail.com

---

## **References**

- [scikit-learn Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [imbalanced-learn SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)
- [pdfplumber Documentation](https://github.com/jsvine/pdfplumber)

---

## **Changelog**

### [0.2.0] - 2025-11-26
- Complete supervised learning pipeline
- SMOTE for class imbalance
- Intelligent text extraction caching
- Comprehensive evaluation metrics
- Identified data quality issues
- Documented improvement roadmap

### [0.1.0] - 2025-11-XX
- Initial project setup
- Basic PDF loading functionality

---

## **Important Notes**

1. **Not Production Ready:** Current model has 9% recall on minority class
2. **Data Quality:** Requires manual review and labeling criteria refinement
3. **Expected Timeline:** 2-3 weeks for data improvement and retraining
4. **Minimum Dataset:** Need 100+ "useful" PDFs for acceptable performance

---

**Status:** **In Development** - Technical implementation complete, data quality improvement in progress