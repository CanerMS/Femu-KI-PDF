# PDF Classification System - Supervised Learning

A system that provides a detection of the pdf documents as "useful" or "not useful" profiting from supervised learning with Random Forest and intelligent text extraction caching.

---

## **Project Status**

| Component            | Status   |          Performance                 |
|----------------------|----------|--------------------------------------|
| **Pipeline**         | Complete | Fully operational                    |
| **Text Extraction**  | Complete | 333/333 PDFs processed               |
| **Preprocessing**    | Enhanced | Author info removal, noise filtering |
| **Model Training**   | Complete | Random Forest trained                |
| **Caching System**   | Complete | 36x speedup                          |
| **Overall Accuracy** | 82%      | Not useful: 89%, Useful: 36%         |
| **Production Ready** | Progress | Data quality improvement ongoing     |

---

## **Current Results**

### **Performance Metrics (Test Set: 84 PDFs)**

```
Classification Report:
              precision    recall  f1-score   support

  not_useful       0.90      0.89      0.90        73
      useful       0.33      0.36      0.35        11

    accuracy                           0.82        84
   macro avg       0.62      0.63      0.62        84
weighted avg       0.83      0.82      0.82        84
```


**Key Improvements (v0.3.0):**
- **4x better "useful" detection** (9% → 36% recall)
- **Model now learns both classes** (not just memorizing majority)
- **Better text preprocessing** (author info removal)
- **Trade-off:** Slight accuracy drop (88% → 82%) but more balanced learning

**Remaining Issues:**
- Still missing 64% of "useful" PDFs (7/11)
- Generic features dominating ("age", "score", "scan")
- Need more aggressive noise filtering
- New Idea: integrate the script also for the .txt files instead of pdf.files
- New Base Class implemented and must get integrated to the programm


---

## **Recent Improvements (v0.3.0)**

### **1. Enhanced Text Preprocessing**

**Added intelligent author section removal:**
```python
# Removes "Author contributions", education backgrounds, affiliations
if 'author' in text and 'contribution' in text:
    parts = re.split(r'\bauthor[s]?\s+contribution[s]?\b', text)
    text = parts[0]  # Keep only content before author section

# Removed noise keywords
noise_words = ['education', 'studying', 'diploma', 'degree', 
               'university', 'institute', 'college', 'school',
               'received', 'obtained', 'graduated', 'phd', 'bachelor', 'master']
```

**Impact:**
- Reduced noise in training data
- Model focuses on actual scientific content
- 4x improvement in useful PDF detection

### **2. Progress Tracking**

**Real-time preprocessing logs:**
```
INFO:preprocess:[█████░░░░░░░░░░] 150/249 (60.2%) | 40064745
INFO:preprocess:  Original: 45,914 chars, Cleaned: 28,500 chars, Reduction: 38.0%
```

### **3. Preprocessing Comparison Reports**

**Automatic before/after analysis:**
```
results/preprocessing_comparison.txt
- Shows original vs cleaned text
- Tracks removed keywords
- Displays reduction percentages
```

---

## **Features**

### **Implemented**
- **Supervised Learning Pipeline**
  - Random Forest classifier (100 estimators)
  - SMOTE for class imbalance handling (32→217 useful samples)
  - Class weights for minority class emphasis (1:15 ratio)
  
- **Intelligent Text Extraction**
  - Automatic caching system (reduces 3min to 5sec on re-runs)
  - Fallback mechanism: pdfplumber → PyPDF2
  - Progress tracking with statistics
  - **NEW:** Preprocessed text caching (`data/preprocessed_texts/`)
  
- **Advanced Text Preprocessing** NEW
  - Author section removal (contributions, affiliations)
  - Education background filtering
  - Noise keyword elimination
  - Number preservation for scientific notation
  - Real-time progress bars
  
- **Feature Engineering**
  - TF-IDF vectorization (2000 features)
  - Bigram support (1-2 word phrases)
  - English stop words removal
  - Feature importance analysis
  
- **Automated Workflows**
  - Stratified train/test split (75%/25%)
  - Automated PDF labeling based on directory structure
  - Model persistence (save/load)
  - Comprehensive logging with progress bars

- **Performance Analysis**
  - Confusion matrix
  - Classification report
  - Feature importance analysis
  - Per-class metrics
  - **NEW:** Preprocessing comparison reports

---

## **Project Structure**

```
Femu-KI-PDF/
├── data/
│   ├── raw_pdfs/              # "Not useful" PDFs (290 files)
│   ├── useful_pdfs/           # "Useful" PDFs (43 files)
│   ├── extracted_texts/       # Cached raw text extractions (333 files)
│   ├── preprocessed_texts/    # NEW: Cleaned texts (333 files)
│   └── labels.csv             # Training labels (333 entries)
├── src/
│   ├── project_config.py      # Centralized configuration
│   ├── loader.py              # PDF loading with label integration
│   ├── extractor.py           # Text extraction with caching
│   ├── preprocess.py          # ENHANCED: Advanced text cleaning
│   ├── features.py            # TF-IDF feature extraction
│   ├── model.py               # Random Forest classifier
│   ├── utils.py               # Helper functions
│   └── label_pdfs.py          # Automated labeling system
├── results/
│   ├── predictions.csv        # Test set predictions
│   ├── preprocessing_comparison.txt  # Before/after analysis
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

## **Usage**

### **Quick Start**

```bash
# Complete pipeline (labeling + training + evaluation)
python main.py
```

### **Force Re-preprocessing**

```bash
# Clear caches to re-extract and re-preprocess
rmdir /s /q data\extracted_texts
rmdir /s /q data\preprocessed_texts

python main.py
```

### **View Preprocessing Results**

```bash
# Check individual cleaned texts
type data\preprocessed_texts\37870716_clean.txt

# View comparison report
type results\preprocessing_comparison.txt
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
3. **Preprocess and clean text (with progress bars)**
4. **Save preprocessed texts**
5. Extract TF-IDF features (2000 features)
6. Apply SMOTE balancing (32→217)
7. Train Random Forest classifier
8. Evaluate on test set
9. Save model and predictions

**Expected Runtime:**
- First run: ~3-4 minutes (text extraction + preprocessing)
- Subsequent runs: ~30-60 seconds (using caches)

**Sample Output:**
```
INFO:preprocess:Preprocessing 249 documents
INFO:preprocess:[████████████████░░░░░░░░░] 150/249 (60.2%) | 40064745
INFO:preprocess:  Original: 45,914 chars, Cleaned: 28,500 chars, Reduction: 38.0%
...
INFO:preprocess:Preprocessing complete
INFO:__main__:Saved 249 preprocessed texts to: data\preprocessed_texts
```

---

## **Configuration**

Edit `src/project_config.py` to customize:

```python
# Directories
RAW_PDFS_DIR = Path("data/raw_pdfs")
USEFUL_PDFS_DIR = Path("data/useful_pdfs")
EXTRACTED_TEXTS_DIR = Path("data/extracted_texts")
PREPROCESSED_TEXTS_DIR = Path("data/preprocessed_texts") 
RESULTS_DIR = Path("results")

# Model Hyperparameters
N_ESTIMATORS = 100          # Number of trees in Random Forest
MAX_DEPTH = 10              # Maximum tree depth
MIN_SAMPLES_SPLIT = 5       # Minimum samples to split node
CLASS_WEIGHT = {0: 1, 1: 15}  # Emphasis on useful class
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

1. **Moderate Minority Class Recall (36%)** IMPROVED from 9%
   - Model detected 4 out of 11 "useful" PDFs in test set
   - Root cause: Generic features still dominating

2. **Feature Quality Issues**
   - Current top features: "age", "score", "scan", "detection"
   - These are **generic medical/research terms**, not specific to "useful" class
   - Need more aggressive noise filtering

3. **Insufficient Training Data**
   - Only 32 "useful" samples in training set
   - Minimum recommended: 100-150 samples

### **Immediate Next Steps (Priority Order)**

#### **HIGH PRIORITY: Aggressive Stop Words (1-2 hours)**

**Problem:** Generic words dominating features
```python
Top features: 'age', 'score', 'scan', 'detection', 'optical'
# These appear in BOTH useful and not_useful PDFs!
```

**Solution:** Add custom stop words list

In `src/project_config.py`:
```python
CUSTOM_STOP_WORDS = [
    # Generic research terms
    'study', 'studies', 'result', 'results', 'method', 'methods',
    'data', 'analysis', 'conclusion', 'background', 'objective',
    
    # Generic medical terms  
    'age', 'gender', 'male', 'female', 'patient', 'patients',
    'score', 'scores', 'scan', 'scans', 'detection', 'detecting',
    
    # Generic descriptors
    'significant', 'showed', 'used', 'using', 'based', 'compared',
    'observed', 'measured', 'performed', 'obtained'
]
```

In `src/features.py`:
```python
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
all_stop_words = list(ENGLISH_STOP_WORDS) + CUSTOM_STOP_WORDS

self.vectorizer = TfidfVectorizer(
    max_features=max_features,
    ngram_range=ngram_range,
    min_df=5,  # Increase from 3
    stop_words=all_stop_words  # Use combined list
)
```

**Expected Impact:** 
- Useful recall: 36% → 50-60%
- More specific features will emerge

---

#### **MEDIUM PRIORITY: Data Expansion (2-3 weeks)**

**Target:** Collect 70-120 additional "useful" PDFs
- Current: 32 samples
- Target: 100-150 samples
- Maintain consistent labeling criteria

---

#### **LOW PRIORITY: Model Tuning (after data expansion)**

- Increase class weight: `{0: 1, 1: 20}` (from 15)
- Add trigrams: `NGRAM_RANGE = (1, 3)`
- Increase features: `MAX_FEATURES = 3000`

---

### **Expected Outcomes After Immediate Improvements**

| Metric | Current (v0.3.0) | After Stop Words | Final Target |
|--------|------------------|------------------|--------------|
| **Useful Recall** | 36% | 50-60% | 70%+ |
| **Useful Precision** | 33% | 40-50% | 60%+ |
| **Useful F1-Score** | 0.35 | 0.45-0.55 | 0.65+ |
| **Overall Accuracy** | 82% | 85%+ | 92%+ |

---

## **Model Details**

### **Architecture**
- **Algorithm:** Random Forest Classifier
- **Estimators:** 100 trees
- **Max Depth:** 10
- **Class Weight:** {0: 1, 1: 15} (emphasize useful class)
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
- **Min DF:** 3 (term must appear in at least 3 documents)

### **Top 10 Features (v0.3.0)**
```
1. age           (0.0348)  # Generic - filtering added
2. detection     (0.0285)  # Generic
3. optical       (0.0266)
4. superior      (0.0243)
5. sampling      (0.0210)
6. score         (0.0202)  # Generic - filtering added
7. scan          (0.0170)  # Generic - filtering added
8. algorithm     (0.0167)
9. visible       (0.0159)
10. targeting    (0.0155)
```

**Note:** Generic terms indicate need for custom stop words

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

### **Issue: "Preprocessing too slow"**
- First run with 333 PDFs: ~60-90 seconds (normal)
- Check progress bars for status
- Subsequent runs use cache (~5 seconds)

### **Issue: "All text being removed during preprocessing"**
**FIXED in v0.3.0**
- Updated regex patterns to be less aggressive
- Numbers now preserved (important for scientific notation)
- Only author-specific sections removed

### **Issue: "Model performance is poor"**
**Partially addressed in v0.3.0**
- Useful recall improved from 9% → 36%
- Next step: Add custom stop words (see "Immediate Next Steps")
- Long term: Expand dataset to 100+ useful samples

---

## **Performance Optimization**

### **Dual Caching System** 

The intelligent caching system now operates at two levels:

**Level 1: Text Extraction Cache**
```
First run:  ~180 seconds (extract 333 PDFs)
Second run: ~5 seconds   (load from cache)
Location:   data/extracted_texts/
```

**Level 2: Preprocessing Cache** 
```
First run:  ~60 seconds  (preprocess 333 texts)
Second run: ~3 seconds   (load from cache)
Location:   data/preprocessed_texts/
```

**Total Speedup:** 
- Without cache: ~4 minutes
- With cache: ~8 seconds
- **50x faster!** 

**To force complete re-processing:**
```bash
# Windows
rmdir /s /q data\extracted_texts
rmdir /s /q data\preprocessed_texts

# Then run:
python main.py
```

---

## **Contributing**

### **Current Priority Tasks**

1. **Add Custom Stop Words** (Immediate - 1-2 hours)
   - Implement CUSTOM_STOP_WORDS in project_config.py
   - Update features.py to use combined stop word list
   - Test and evaluate impact on recall

2. **Data Collection** (High Priority - 2-3 weeks)
   - Collect 70-120 additional "useful" PDFs
   - Document clear labeling criteria
   - Verify existing labels

3. **Experimentation** (After data expansion)
   - Test different hyperparameters
   - Try alternative ML algorithms
   - Implement cross-validation

4. **Documentation**
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
- [TF-IDF Vectorization](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)

---

## **Changelog**

### [0.3.0] - 2025-12-02 MAJOR UPDATE
- **4x improvement in useful PDF detection** (9% → 36% recall)
- Enhanced text preprocessing with author section removal
- Added preprocessing cache system (data/preprocessed_texts/)
- Real-time progress bars for all stages
- Preprocessing comparison reports
- Number preservation for scientific notation
- Identified generic feature issue
- Documented custom stop words solution

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

1. **Not Production Ready:** Current model has 36% recall on minority class
   - **Immediate action needed:** Implement custom stop words
   - **Long term:** Expand dataset to 100+ useful samples

2. **Significant Progress:** 4x improvement in useful detection (v0.2 → v0.3)

3. **Next Milestone:** 50-60% useful recall (achievable with stop words)

4. **Final Target:** 70%+ useful recall (requires data expansion)

---

**Status:** **Active Development** - Core improvements implemented, optimization in progress

**Last Updated:** 2025-12-02
**Version:** 0.3.0