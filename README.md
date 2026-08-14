# PDF + TXT Classification Programm - Supervised Learning

A programm that classifies PDF and TXT documents as either "relevant" or "not relevant" supported by supervised learning with Random Forest, SVM, Logistic Regression mode and intelligent text extraction caching concluding with preprocess phase.

One can switch in 2 different modes: Supervised and Unsupervised learning. 
The file type can be arranged in main.py under "# 0. Choose File Type" by typing either "pdf" or "txt". 
In this specific case, supervised learning is more suitable. Therefore, I stopped improving unsupervised learning in previous version, but kept it for reference and comparison.  
The most difficult challenge in this case is, that some pdfs don't have any semantic similarities between them and including completely different types of words and structure, making it difficult for the model to generalize across documents.

I named not relevant data as "raw" and relevant file as "useful".

# Warning! 
The programm is not compatible with servers owning little capacity of RAM. You suppose to have good memory capacity in order to run especially SciBert. Using SciBert, only before it starts calculating the unseen data, there will be crash by uploading the model. Because SciBert has approximately 110 Millions parameters to upload into RAM. This causes shutdown, if you do not have sufficient RAM. I have 16GB RAM and it works fine.  
---

## **Project Status**

| Component            | Status   |              Performance                |
|----------------------|----------|-----------------------------------------|
| **Pipeline**         | Complete | Fully operational                       |
| **Model Training**   | Complete | Random Forest, SVM, Logical Regressiong |
| **Last Accuracy**    | 93.1%    | SciBert Understanding Integrated        |
| **Production Ready** | L. Phase | Data quality + Feature improvement ong. |

---

## **Current Results**

**Remaining Issues:**
- Optimization for more robuts programm


## **Features**

### **Implemented**

- **Semantic Understanding Bert**
  - Scibert integrated
  - 3 Categories available: Prediction (setable Threshold), Evaluation, Training

- **Semantic Understanding Bert**
  - SBERT integrated
  - Three modes possible: Semantic , TFD-ID , Combined
  - Combined mode result: 92% accuracy

- **Supervised Learning Pipeline**
  - Random Forest classifier
  - SMOTE for class imbalance handling for increasing the number of PDFs artifically, when needed
  - Workflow optimization
  
- **Intelligent Text Extraction**
  - Automatic caching system (reduces 3min to 5sec on re-runs)
  - Fallback mechanism: pdfplumber → PyPDF2
  - Progress tracking with statistics
  - Preprocessed text caching (`data/preprocessed_texts/`)
  
- **Advanced Text Preprocessing** 
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

- **Human-in-the-Loop (Feedback Loop)**
  - Automated threshold routing for uncertain predictions (`manual_check/`)
  - Unified `correct()` API for manual human-review corrections
  - `.jsonl` logging system to track all reviewer decisions
  
- **Automated Workflows**
  - Stratified train/test split (75%/25%)
  - Automated PDF labeling based on directory structure
  - Model persistence (save/load)
  - Comprehensive logging with progress bars
  - External JSON Pipeline Integration (`lisa.py`)

- **Performance Analysis**
  - Confusion matrix
  - Classification report
  - Feature importance analysis
  - Per-class metrics
  - Preprocessing comparison reports

---

## **Project Structure**

```text
Femu-KI-PDF/
├── data/
│   ├── raw_pdfs/              # "Not useful" PDFs (Training)
│   ├── raw_texts/             # "Not useful" Texts (Training)
│   ├── useful_pdfs/           # "Useful" PDFs (Training)
│   ├── useful_texts/          # "Useful" Texts (Training)
│   ├── extracted_texts/       # Cached raw text extractions 
│   ├── preprocessed_texts/    # Cleaned texts after preprocessing
│   ├── manual_check/          # Uncertain predictions requiring human review
│   ├── feedback/              # JSONL logs for human feedback actions
│   └── labels.csv             # Training labels 
├── src/
│   ├── project_config.py      # Centralized configuration
│   ├── loader.py              # PDF loading with label integration
│   ├── extractor.py           # Text extraction with caching
│   ├── preprocess.py          # ENHANCED: Advanced text cleaning
│   ├── features.py            # TF-IDF feature extraction
│   ├── model.py               # Random Forest classifier
│   ├── utils.py               # Helper functions
│   ├── label_files.py         # Automated labeling system
│   ├── semantic.py            # Semantic understanding (SciBERT/MiniLM)
│   ├── predict.py             # Predict unseen data (Production flow)
│   ├── evaluate_predictions.py# Evaluate unseen data
│   ├── feedback.py            # Human-in-the-loop correction logic
│   ├── feedback_apply.py      # Applies pending feedback to filesystem
│   └── lisa.py                # External JSON processing integration API
├── results/
│   ├── predictions.csv               # Test set predictions
│   ├── preprocessing_comparison.txt  # Before/after analysis
│   └── pdf_classifier.joblib         # Trained model
├── logs/
│   └── label_files.log        # Labeling process logs
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
# PDF Processing
pdfplumber>=0.9.0
PyPDF2>=3.0.0

# Machine Learning
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.0.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Utilities
joblib>=1.3.0

# Imbalanced Learning
imbalanced-learn>=0.11.0

# Semantic Understanding - AI
sentence-transformers>=2.2.0 # Transformer based AI model
torch>=2.0.0 # PyTorch - Deeplearning Framework
transformers>=4.35.0 # Hugging Face Transformers
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

### **Step-by-Step**

#### **Step 1: Prepare Your Data**

Place PDFs in appropriate directories:
```bash
data/
├── raw_pdfs/        # Put "not useful" PDFs here
└── useful_pdfs/     # Put "useful" PDFs here
├── raw_texts/       # Put "not useful" TXTs here
└── useful_texts/    # Put "useful" TXTs here
```

#### **Step 2: Choose File Type and Create Labels**

```python
# main.py 
FILE_TYPE = 'pdf' # switching into txt is also possible
```
```bash
python src/label_files.py
```

#### **Step 3: Train and Evaluate**

```bash
python main.py
```

**Pipeline Stages:**

**1: Training**
1. Load labels and PDFs or TXTs
2. Extract text (with caching if it includes already processed PDFs)
3. Preprocess and clean text (with progress bars)
4. Save preprocessed texts
5. Extract TF-IDF features (2000 features)
6. Apply SMOTE balancing (if needed)
7. If settings are combined, scibert, then it will be calculating vectors
8. Train Classifier Model with tfidef/scibert/sbert/or combined(tfidf+scibert) data outputs
9. Evaluate on test set 
10. Save model, scaler and tfidf dict as .joblib into `./results`, prediction and confusion matrix (png) in `\results`

**2: Prediction (Production)**
1. Chose the compatible file structure .txt/pdf in `predict.py`
2. Set up the path of scaler, tf-idf, and model `.joblib` files in `predict.py`
3. Upload the unseen PDF/TXT files into `data/to_test_files`
4. Run `src\predict.py`
5. The threshold can be arranged by changing `CONFIDENCE_THRESHOLD` in `predict.py`
6. **Smart Routing:**
   - If confidence **≥ Threshold**: Files are automatically moved to `useful_texts` or `raw_texts` (adds directly to your future training dataset).
   - If confidence **< Threshold**: Files are moved to `data/manual_check/` for a human reviewer.

**3: Human-in-the-loop (Reviewing Uncertainties)**
1. Go to `data/manual_check/` and review the texts the model was unsure about.
2. Open Python/Console and use the `feedback.py` API to log your decision:
   ```python
   from feedback import correct
   # If you decide the paper is useful:
   correct("paper_123.txt", decision="useful")
   ```
3. The file is immediately moved to `useful_texts` and the action is safely logged in `data/feedback/feedback.jsonl`.
4. (Optional) If you spot a wrong file already in training data, simply call `correct("paper_456.txt")` without a decision to automatically flip its label.

**4: Evaluation of the new files**
1. Run `src\evaluate_predictions.py`, it will create an advanced confusion matrix just like in Training stage.
2. See the results under `./results`
3. Note: These results can differ from the initial 93% accuracy because we force the model to decide 1/0 on unseen data. Schedule this stage once a month to check if the model is drifting and needs retraining.


## **Configuration**

Edit `src/project_config.py` to customize:

```python
# Directories
RAW_PDFS_DIR = Path("data/raw_pdfs")
USEFUL_PDFS_DIR = Path("data/useful_pdfs")
MANUAL_CHECK_DIR = Path("data/manual_check") # Human review
...

# Model Hyperparameters
N_ESTIMATORS = 100            # Number of trees in Random Forest
MAX_DEPTH = 10                # Maximum tree depth
CLASS_WEIGHT = {0: 1, 1: 15}  # Emphasis on useful class
...

# Important to edit CUSTOM_STOP_WORDS based on your CASE!!!
CUSTOM_STOP_WORDS = [......]
```

---

## **Current Issues & Next Steps**

### **HIGH PRIORITY: (30 days)**
- Optuna Integration for hyper parameters
- SHAP, machine tells why it is useful/unuseful

### **What kind of contribute does Smote provide?**
According to my experience, SMOTE approach doesn't work as fine as one needs in this specific scenario because the model relies heavily on term frequencies. Therefore, I'd recommend uploading as many PDFs or TXTs as possible to prevent the pipeline from needing SMOTE.

---

## **Some Errors That Can Occure**
### **Issue: "No PDF files found"**
- Ensure PDFs are in `data/raw_pdfs/` and `data/useful_pdfs/`
- Check file extensions (must be `.pdf`)

### **Issue: "Preprocessing too slow"**
- First run with x number of PDFs: ~60-90 seconds (normal)
- Check progress bars for status
- Subsequent runs use cache and takes usually ~5 seconds

---

## **To force complete re-processing:**

```bash
# For a fresh start using Windows
src\label_files.py # Labeling files to train the machine
rmdir /s /q data\extracted_raw_pdfs
rmdir /s /q data\extracted_raw_texts
rmdir /s /q data\extracted_useful_pdfs
rmdir /s /q data\extracted_useful_texts
rmdir /s /q data\preprocessed_raw_texts
rmdir /s /q data\preprocessed_useful_texts

# For a fresh start with labels.csv
del data\labels.csv

# For a fresh start for predicting new files
rmdir /s /q data\manual_check
rmdir /s /q data\to_test_files

# Then run:
python main.py or py main.py
```

## **References**
- [scikit-learn Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [imbalanced-learn SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)
- [pdfplumber Documentation](https://github.com/jsvine/pdfplumber)
- [TF-IDF Vectorization](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [semantic-transformer Documentation](https://sbert.net/)
- [scibert-resource](https://github.com/allenai/scibert)

---

## **Changelog**

### [0.7.0] - Upcoming
- **Architecture**: Replaced `sorted_pdfs` structure with direct automated routing to training directories (`useful_texts`, `raw_texts`) and `manual_check/`.
- **Human-in-the-Loop**: Integrated `feedback.py` with unified `correct()` API for manual label reviews and auto-flips.
- **Traceability**: Added append-only `.jsonl` logging system (`feedback.jsonl`) for tracking human reviewer decisions.
- **Integration**: Added `lisa.py` external pipeline integration for JSON stream processing.

### [0.6.3] - 2026-05-22
- Code fixes for better reproducibility 

### [0.6.2] - 2026-04-29
- Advanced Fallback system for src\predict.py ,Solved Problem: CPU was too much charged with fallback

### [0.6.1] - 2026-04-21
- Accuracy enhanced to 93.1% with summary texts
- SciBert (better with scientific texts) integrated: Combi from TF-IDF + SciBert = 93.1%
- Predict.py script added: Includes not only trained LR joblib, also TF-IDF and Scaler joblib
- Cache for trained files, don't wait 2 hours if you want to run again

### [0.6.0] - 2026-04-09
- Semantic Understanding (SBert) integrated
- 3 mode-switch possible: Combined, TF-IDF, Semantic Understanding
- Better results with 92% accuracy

**Status:** **Active Development** 
- Semantic understanding integration
- More and cleaner data

**Last Updated:** 2026-07-14
**Version:** 0.7.0