# PDF + TXT Document Classification Program - Supervised Learning

An end-to-end machine learning pipeline designed to automatically classify PDF and TXT documents as either **Relevant** (`useful`) or **Not Relevant** (`raw`). Built for accuracy, the system leverages a hybrid feature extraction approach to achieve robust classification, even when documents lack structural consistency or share little overlapping vocabulary.

### **Why This Project?**
It is a time consuming process to decide on scientific papers if they are relevant or not. It may take 10 minutes to decide on one paper, when you have to decide on 1000 papers, it will take 10000 minutes (166 hours). This is where this project comes in. The core challenge in automated document classification is that many scientific or technical PDFs lack semantic similarity. They often contain completely different structures, terminologies, and layouts, making it difficult for standard models to generalize. 
To overcome this, this project implements a **Hybrid NLP Architecture**:
1. **TF-IDF:** Captures statistical keyword importance and domain-specific terminology.
2. **SciBERT / MiniLM:** Understands deep contextual and semantic meaning.
3. **Logistic Regression:** Acts as a powerful meta-classifier to weigh these combined features.

### **Core Highlights**
- **Intelligent Pipeline:** Automated text extraction, advanced noise filtering, and incremental per-file caching.
- **Human-in-the-Loop (HITL):** Confident predictions are automatically routed, while uncertain documents are set aside for manual review and seamlessly fed back into the training loop.
- **Resource Optimized:** Implements memory-mapped arrays and garbage collection to process large vector spaces (like SciBERT's 768-dim embeddings) efficiently on standard hardware.

> [!WARNING]
> **Memory Requirements**
> - **SciBERT** (`allenai/scibert_scivocab_uncased`): ~110M parameters -> ~450 MB RAM on load
> - **Minimum:** 8 GB RAM | **Recommended:** 16 GB RAM
> - On low-memory machines, the process will be killed during model initialization, before any or some files are processed.
> - **GPU:** Automatically used if CUDA is available; falls back to CPU otherwise (significantly slower).
> - **Lighter alternative:** Switch to MiniLM in `project_config.py` (`SEMANTIC_MODEL_TYPE = 'minilm'`) uses ~90 MB RAM and achieves ~92% accuracy. 
---

## **Project Status**

| Core Aspect          | Current State  | Key Characteristics                       |
|----------------------|----------------|-------------------------------------------|
| **End-to-End Flow**  | Fully Active   | Incremental caching, garbage collection   |
| **Classification**   | 93.7% Accuracy | Logistic Regression via Hybrid Features   |
| **NLP Engine**       | Integrated     | TF-IDF combined with SciBERT / MiniLM     |
| **Production**       | Ready          | Smart routing and feedback loop enabled   |

---

## **Current Results**

### **Implemented**

- **Garbage Collector**
  - Automatic Memory Cleanup after model operations
  - Reduces memory usage, reduces RAM usage significantly

- **Smart Cache Management**
  - Automatically detects changed source files
  - Updates cache only when needed
  - Preserves previous cache files for reference

- **Semantic Understanding SciBERT**
  - SciBERT integrated
  - 3 Categories available: Prediction (setable Threshold), Evaluation, Training

- **Semantic Understanding MiniLM/BERT**
  - SBERT integrated
  - Three modes possible: Semantic , TF-IDF , Combined
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
│   ├── extracted_useful_texts/ # Cached useful text extractions
│   ├── extracted_raw_pdfs/    # Cached useful pdf extractions
│   ├── preprocessed_raw_texts/ # Cleaned texts after preprocessing
│   ├── preprocessed_useful_texts/ # Cleaned texts after preprocessing
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
│   ├── evaluate_predictions.py  # Evaluate unseen data
│   ├── feedback.py            # Human-in-the-loop correction logic
│   ├── give_feedback.py       # 
│   └── lisa.py                # External JSON processing integration API
├── results/
│   ├── predictions.csv               # Test set predictions
│   ├── preprocessing_comparison.txt  # Before/after analysis
│   ├── classification_report.txt     # Classification report
│   ├── feature_importance.txt        # Feature importance analysis (only in TF-IDF mode, not in combined)
│   ├── confusion_matrix.png          # Confusion matrix
│   ├── *_classifier.joblib           # Trained model
│   ├── *_scaler.joblib               # Trained scaler
│   └── *_tfidf.joblib                # Trained vectorizer
├── logs/
│   └── label_files.log               # Labeling process logs
├── main.py                           # Main pipeline orchestrator
├── requirements.txt                  # Dependencies
├── .gitignore                        # Gitignore
└── README.md                         # Project README
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

> **Prerequisites:** Python 3.8+, 8 GB RAM minimum (16 GB recommended for SciBERT)

**Step 1 - Clone & Install**
```bash
git clone <repository-url>
cd Femu-KI-PDF
pip install -r requirements.txt
```

**Step 2 - Add your documents**
```
data/
├── useful_texts/  <- Put your RELEVANT .txt files here
└── raw_texts/     <- Put your NOT RELEVANT .txt files here
```
> Switch to PDF mode: set `FILE_TYPE = 'pdf'` in `main.py` and use `useful_pdfs/` / `raw_pdfs/` instead.

**Step 3 - Configure**

Open `src/project_config.py` and adjust at minimum:
```python
FEATURE_MODE = 'combined'        # 'tfidf' | 'semantic' | 'combined'
SEMANTIC_MODEL_TYPE = 'scibert'  # 'scibert' | 'minilm' (lighter, less RAM)
CUSTOM_STOP_WORDS = [...]        # Add domain-specific noise words for your case
```

**Step 4 - Label & Train**
```bash
py src/label_files.py   # Scans directories → generates data/labels.csv
py main.py              # Full training pipeline → saves model to results/
```

**Step 5 - Predict new files**
```bash
# Drop new files into:
data/to_test_files/

# Run prediction:
py src/predict.py

```

**Step 6 - Review uncertain predictions**
- Results automatically routed to:
```bash
- data/useful_texts/      confident USEFUL predictions    (added to training data)
- data/raw_texts/         confident NOT USEFUL predictions (added to training data)
- data/manual_check/      uncertain predictions (review manually)
```

# After reviewing files in data/manual_check/:
py src/give_feedback.py <filename_without_extension> <true/false>

# Examples:
py src/give_feedback.py artikel_042 true    # → marks as USEFUL
py src/give_feedback.py artikel_042 false   # → marks as NOT USEFUL
```

**Step 7 - Retrain with new data**
```bash
# No deletions needed - embeddings are cached per file (.npy)
# Only new files will be recomputed; everything else loads from cache
py src/label_files.py
py main.py
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
FILE_TYPE = 'txt' # switching into pdf is also possible
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

1. Setup environment variables: Edit `src/project_config.py`, `src/label_files.py` and `src/main.py` to customize the settings of the pipeline.
2. Load labels from `labels.csv` and corresponding PDF/TXT files
3. Extract text from files *(disk-cached per file skipped on re-runs)*
4. Preprocess and clean text *(noise removal, author filtering, normalization)*
5. Save preprocessed texts to `data/preprocessed_texts/` for inspection
6. Extract TF-IDF features *(up to 2000 features; Chi-Squared selection applied in `tfidf` mode only)*
7. Extract semantic embeddings via **SciBERT** or **MiniLM** *(per-file .npy cache, only new files computed)*
8. Combine TF-IDF + semantic vectors if `FEATURE_MODE = 'combined'` *(MaxAbsScaler applied)*
9. Apply **SMOTE** oversampling if class imbalance exceeds threshold
10. Train **Logistic Regression** classifier *(5-fold cross-validation included)*
11. Evaluate on held-out test set *(accuracy, classification report, confusion matrix)*
12. Save model, TF-IDF vocabulary, and scaler as `.joblib` into `results/`
13. Confusion matrix in png format and prediction in csv format are saved in `results/`

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
3. The file is immediately moved to `useful_texts` or `raw_texts` based on your feedback.
4. (Optional) If you spot a wrong file already in training data, simply call `correct("paper_456.txt")` without a decision to automatically flip its label.

**4: Evaluation of the new files**
1. Run `src\evaluate_predictions.py`, it will create an advanced confusion matrix just like in Training stage.
2. See the results under `./results`
3. Note: These results can differ from the initial 93% accuracy because we force the model to decide 1/0 on unseen data. Schedule this stage once a month to check if the model is drifting and needs retraining.

## **Left As References (Not Actively Maintained)**
These components exist in the codebase and still functional, but were 
not further improved as better alternatives were found:
1. **Unsupervised / Anomaly Detection** (`model.py` -> `AnomalyDetector`)
   - Uses IsolationForest; no labels required
   - Abandoned: supervised learning gave significantly better results
2. **Random Forest** (`model.py` -> `FILEClassifier`)
   - Still selectable via `project_config.py`
   - Lower accuracy than Logistic Regression in this use case
3. **SVM** (`model.py` -> `SVMClassifier`)
   - Still selectable via `project_config.py`
   - ~3-5% less accurate than Logistic Regression
4. **MiniLM / SBERT** (`semantic.py` -> `SemanticFeatureExtractor`)
   - Faster and lighter than SciBERT (~384 dim vs 768 dim)
   - ~92% accuracy vs SciBERT's 93.1% on scientific texts
   - Use this if RAM/speed is a concern
5. **`combine_with_tfidf()` method** (in both semantic classes)
   - Defined but never called; main.py uses its own combining logic
6. **SMOTE** (`train.py` in `balancing_type='smote'`)
   - Balancing method that oversamples minority class
   - Not used in final pipeline; main.py uses `class_weight` instead
7. **chisquare** (`train.py` in `scoring_method='chi2'`)
   - Feature selection method
   - Not used in final pipeline; main.py uses `feature_selection='best_k'`


## **Configuration**

Edit `src/project_config.py` to customize the settings of the pipeline:

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

### **Issue: "MemoryError: model.fit()"**
- Model is too big for your RAM
- Use MiniLM instead of SciBERT (set `SEMANTIC_MODEL_TYPE = 'minilm'` in `project_config.py`)

### **Issue: "No files found" after preprocessing**
- Ensure PDFs are in `data/raw_pdfs/` and `data/useful_pdfs/`
- Check file extensions (must be `.pdf`)

### **Issue: "Evaluation Failed"**
- Ensure you have a `labels.csv` file in your `data` directory
- Or run `src/label_files.py` to create one

### **Issue: "Data Splitting Error: X and y must have same number of samples"**
- This means there are no new "unlabeled" texts in `data/useful_texts/` or `data/raw_texts/` to split for evaluation.
- Check `data/useful_texts/` and `data/raw_texts/` - if both are empty, the model has nothing new to evaluate.
- Fix: Add new PDF files to `data/raw_pdfs/` and `data/useful_pdfs/` and run the pipeline again to generate new training data.

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
del data\features\cache_scibert\*.npy

# For a fresh start for predicting new files
rmdir /s /q data\manual_check
rmdir /s /q data\to_test_files

# Then if you want to retrain run:
python label_files.py or py label_files.py
python main.py or py main.py
```

## **References & Acknowledgments**
- [scikit-learn Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [imbalanced-learn SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)
- [pdfplumber Documentation](https://github.com/jsvine/pdfplumber)
- [TF-IDF Vectorization](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [sentence-transformers Documentation](https://sbert.net/)
- **SciBERT**: Beltagy, I., Lo, K., & Cohan, A. (2019). *SciBERT: A Pretrained Language Model for Scientific Text*. EMNLP. Model: [`allenai/scibert_scivocab_uncased`](https://huggingface.co/allenai/scibert_scivocab_uncased), provided by [Allen Institute for AI (AllenAI)](https://github.com/allenai/scibert) under the **Apache License 2.0**.

---

## **Changelog**

### [1.0.0] - 2026-09-25
- **Architecture**: Production Ready For Special Needs
- **Memory Management**: Garbage Collector Added for better performance avoiding memory leaks
- **Smart Cache Management**: Updated

### [0.7.0] - 
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

### What can be done in the future
- Add web interface, for example FASTAPI
- Add more ML models and compare their performance
- Add more NLP models and compare their performance
- Change the architecture into streaming if needed
- Improve the referenced part of the code to make it more modular

**Last Updated:** 2026-09-25
**Version:** 1.0.0