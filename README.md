# PDF + TXT Classification Programm - Supervised Learning

A programm that classifies PDF and TXT documents as either "useful" or "not useful" supported by supervised learning with Random Forest, SVM, Logistic Regression mode and intelligent text extraction caching.
One can switch in 2 different modes: Supervised and Unsupervised learning. Also, the file type can be arranged in main.py under "# 0. Choose File Type" by typing either "pdf" or "txt". Nevertheless, in this specific case, supervised learning is more suitable. Therefore, I stopped improving unsupervised learning in previous version, but kept it for reference and comparison.  The most difficult challenge in this case is, that some pdfs don't have any semantic similarities between them and including completely different types of words and structure, making it difficult for the model to generalize across documents.
I named unuseful data as "raw" and useful data as "useful".

# Warning! 
The programm is not compatible with servers having little RAM. You suppose to have good memory capacity in order to run especially SciBert. Using SciBert, only before it starts calculating the unseen data, there will be crash by uploading the model. Because SciBert has approximately 110 Millions parameters to upload into RAM. This cause shutdown, if you do not have sufficient RAM. I have 16GB RAM and it works fine.  
---

## **Project Status**

| Component            | Status   |              Performance                |
|----------------------|----------|-----------------------------------------|
| **Pipeline**         | Complete | Fully operational                       |
| **Model Training**   | Complete | Random Forest, SVM, Logical Regressiong |
| **Overall Accuracy** | 93.1%    | SciBert Understanding Integrated        |
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
  - Preprocessing comparison reports

---

## **Project Structure**

```
Femu-KI-PDF/
├── data/
│   ├── raw_pdfs/              # "Not useful" PDFs 
│   ├── raw_texts/             # "Not useful" Texts
│   ├── useful_pdfs/           # "Useful" PDFs
│   ├── useful_texts/          # "Useful" Texts
│   ├── extracted_texts/       # Cached raw text extractions 
│   ├── preprocessed_texts/    # Cleaned texts after preprocessing
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
│   ├── semantic.py            # Semantic understanding (ongoing, needs integration) 
│   ├── predict.py             # Predict unseen data, works alone
│   └── evaluate_predictions.py       # Evaluate unseen data
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
├── raw_pdfs/        # Put "not useful" PDFs here
└── useful_pdfs/     # Put "useful" PDFs here
├── raw_texts/       # Put "not useful" TXTs here
└── useful_texts/    # Put "useful" TXTs here
```

#### **Step 2: Choose File Type and Create Labels**

```bash
main.py FILE_TYPE = 'pdf' # switching into txt is also possible

python src/label_files.py
```

**Output:**
```

```

#### **Step 3: Train and Evaluate**

```bash
python main.py
```

**Pipeline Stages:**

**1: Training**
1. Load labels and PDFs or TXTs
2. Extract text (with caching if it includes already processed PDFs)
3. **Preprocess and clean text (with progress bars)**
4. **Save preprocessed texts**
5. Extract TF-IDF features (2000 features)
6. Apply SMOTE balancing (if needed)
7. If settings are combined, scibert, then it will be calculating vectors
8. Train Classifier Model with tfidef/scibert/sbert/or combined(tfidf+scibert) data outputs
8. Evaluate on test set 
9. Save model, scaler and tfidf dict as .joblib into ./results, prediction and confusion matrix (png) in \results

**2: Prediction**
1. Chose the compatible file structure .txt/pdf in predict.py
2. Have the scaler.joblib, tf-idf.joblib, scibert.joblib
3. Set up the path of these joblib files in predict.py 
4. Upload the unseen PDF/TXT files into src/to_test_files
5. run src\predict.py
6. Programm is going to create 3 Files to localize them (Useful, Not_Useful, Manual_Check) in data/sorted_pdfs
7. The threshold for being useful,unuseful can be arranged by changing CONFIDENCE_THRESHOLD in predict.py
8. Based on the Confidence Threshold deciding if human control needed or useful, unuseful.
9. Manuel_Check directory is createad for human control

**3: Evaluation of the new files**
1. Important: Before you run, make sure you have labels.cv in .\data 
2. Manually: You can type in the list in accurate format, which files useful/unuseful
3. Automatically: You can run src\label_files.py after you uploaded your useful/unuseful files into raw_pdfs/useful_pdfs , raw_texts/useful_texts 
4. Programm can only work with one file type at once, means set the file type (txt/pdf) in evaluate_predictions.py and also label_files.py if automatically label process needed
5. Run the src\evaluate_predictions.py, it will create advanced confusion matrix just like in first stage: Training above
6. You can see the results under ./results 
7. Do not forget, that this results can be different/lower/higher from 93% accuracy. Because we push model to decide either for 1 or 0 on unseendata. In industry the stage 2 is
used. You can check with 3. stage once in a month, to be sure, whether model is drifting, or it needs to be trained again.

**Expected Runtime:**
- Depends strongly on the model you use.
- SciBert can last above 1-2 hours at the first time. Then, due to cache system, that I implemented, it would last so much shorter for the upcoming tries. That cache will be created in ./data/features/cache_scibert
- SBert is so much faster (maybe not even 10 mins), but the results can be 1% to 5% worse.
- About classification models, Random Forest lasts longer than SVM and Logistic Regression.
- I personally stopped training the machine with Random Forest since I have gotten better results with Logistic Regression in previous versions. But I keep these models for reference.


## **Configuration**

Edit `src/project_config.py` to customize:

```python

...

# Directories
RAW_PDFS_DIR = Path("data/raw_pdfs")
USEFUL_PDFS_DIR = Path("data/useful_pdfs")
EXTRACTED_USEFUL_TEXTS_DIR = Path("data/extracted_useful_texts")
EXTRACTED_NOT_USEFUL_TEXTS_DIR = Path("data/extracted_raw_texts")
EXTRACTED_USEFUL_PDF_DIR = Path("data/extracted_useful_pdfs")
EXTRACTED_NOT_USEFUL_PDF_DIR = Path("data/extracted_raw_pdfs")
PREPROCESSED_TEXTS_DIR = Path("data/preprocessed_texts") 
RESULTS_DIR = Path("results")

# Model Hyperparameters
N_ESTIMATORS = 100            # Number of trees in Random Forest
MAX_DEPTH = 10                # Maximum tree depth
MIN_SAMPLES_SPLIT = 5         # Minimum samples to split node
CLASS_WEIGHT = {0: 1, 1: 15}  # Emphasis on useful class
RANDOM_STATE = 42             # Reproducibility seed

# Feature Extraction
MAX_FEATURES = 2000           # Maximum TF-IDF features
NGRAM_RANGE = (1, 2, 3)       # Unigrams and bigrams

# Data Split
TEST_SIZE = 0.25              # 25% for testing
SMOTE_THRESHOLD = 3.0         # Apply SMOTE if imbalance ratio > 3

# Important to edit CUSTOM_STOP_WORDS based on your CASE!!!
CUSTOM_STOP_WORDS = [......]

# For the stage 2: Prediction
CONFIDENCE_THRESHOLD = int 0 bis 100
...
```

---

## **Current Issues & Next Steps**

### **Known Issues**

### **HIGH PRIORITY: (30 days)**
- Optuna Integration for hyper parameters
- SHAP, machine tells why it is useful/unuseful

### **What kind of contribute does Smote provide?**
```
Suppose...
Original Training Data:
  - Not useful: 217 samples
  - Useful: 32 samples
  - Imbalance ratio: 1:6.8

After SMOTE:
  - Not useful: 217 samples
  - Useful: 217 samples (synthetically balanced)
  - Total: 434 samples

Important: 
According to my experience, SPOT approach doesn't work as fine as one needs in this specific scenario. The reason behind is, My model based on the words and the number of their appearence. Therefore, I'd recommend the approach of uploading as many as possible PDFs or TXTs (whatever you are working with, inorder to prevent pipeline to use SMOTE approach)
```

### **Feature Extraction**
- These are observable in project_config.py
- **Method:** TF-IDF (Term Frequency-Inverse Document Frequency)
- **Features:** 2000 most important terms
- **N-grams:** Unigrams + Bigrams
- **Stop Words:** English common words and words that I think extra noise removed
- **Min DF:** 3 (term must appear in at least 3 documents)

### **Top 4 Features (v0.3.0)**
```
- This is an example that I got as a result 
- age or algorith etc. I 've forbidden such words using custom_stop_words in project_config.py
1. age           (0.0348)  # Generic - filtering added
2. detection     (0.0285)  # Generic
3. optical       (0.0266)
4. superior      (0.0243)

```
### **Top 4 Useful Features (v0.4.0)**
```
feature_index,feature_name,useful_mean_tfidf,not_useful_mean_tfidf,difference

890,stimulation,0.05333353823875638,0.01894441336819477,0.034389124870561616
356,field,0.054022637516202565,0.021323286963875792,0.03269935055232677
42,antenna,0.0391295588308538,0.006827449314358755,0.032302109516495044
337,exposure,0.04140112148788596,0.010149379306450464,0.0312517421814355

```

### **Top 4 Unuseful Features (v0.4.0)**
```
feature_index,feature_name,useful_mean_tfidf,not_useful_mean_tfidf,difference

616,mri,0.010843031107129517,0.03393329435540108,-0.023090263248271563
82,beam,0.0009140193554566058,0.0164845234152568,-0.015570504059800193
820,screen,0.00043567619084909707,0.015347509185121138,-0.014911832994272041
657,opt,0.0032156956088274474,0.017461207400490775,-0.014245511791663328
```


### **Some Errors That Can Occure**
### **Issue: "No PDF files found"**
- Ensure PDFs are in `data/raw_pdfs/` and `data/useful_pdfs/`
- Check file extensions (must be `.pdf`)

### **Issue: "Text extraction failed"**
- Some PDFs may be scanned images (OCR required)
- Try with different PDFs first
- Check logs in `data/extracted_texts/`

### **Issue: "Preprocessing too slow"**
- First run with x number of PDFs: ~60-90 seconds (normal)
- Check progress bars for status
- Subsequent runs use cache and takes usually ~5 seconds

### **Issue: "All text being removed during preprocessing"**
- Updated regex patterns to be less aggressive
- Numbers now preserved (important for scientific notation)
- Only author-specific sections removed

---

## **ML Terms in confusion matrix**
**Recall**
- Performance metric in machine learning especially for classification tasks
- It measures the model's ability to identify all actual positive samples correctly in a dataset
- Recall = True Positives (TP) / (True Positives (TP) + False Negatives (FN))
- Focuses on minimizing false negatives

**F1 Score**
- An important metric for evaluation classification models in ML
- Combines precision and recall together
- It is especially useful, when a balance between recall and precision is needed
- F1 = 2 * (Precision * Recall) / (Precision + Recall)
- Useful, when one of the class's data number is overweighting 

**Accuracy**
- The formel used is (TP + TN) / Total Sample Number

**Precision**
- Measures the correctness of positive predictions
- The formel used is  TP / (TP + FP) if (TP + FP) > 0 else 0


**To force complete re-processing:**

- Espeacially important for the following runs with different data
```bash

# For a fresh start using Windows
src\label_files.py
rmdir /s /q data\extracted_raw_pdfs
rmdir /s /q data\extracted_raw_texts
rmdir /s /q data\extracted_useful_pdfs
rmdir /s /q data\extracted_useful_texts
rmdir /s /q data\preprocessed_raw_texts
rmdir /s /q data\preprocessed_useful_texts

# For a fresh start with labels.csv
rmdir /s /q data\labels.csv

# For a fresh start for predicting new files
rmdir /s /q data\sorted_pdfs\Manual_Check
rmdir /s /q data\sorted_pdfs\Not_Useful
rmdir /s /q data\sorted_pdfs\Useful

# Then run:
python main.py or py main.py
```


## **License**

---

## **Contact**

canerrcc1@gmail.com

---

## **References**

- [scikit-learn Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [imbalanced-learn SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)
- [pdfplumber Documentation](https://github.com/jsvine/pdfplumber)
- [TF-IDF Vectorization](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [semantic-transformer Documentation](https://sbert.net/)
---

## **Changelog**

### [0.6.2] - 2026-04-29
- Advanced Fallback system for src\predict.py ,Solved Problem: CPU was too much charged

### [0.6.1] - 2026-04-21
- Accuracy enhanced to 93.1% with summary texts
- SciBert (better with scientific texts) integrated: Combi from TF-IDF + SciBert = 93.1%
- Predict.py script added: Includes not only trained LR joblib, also TF-IDF and Scaler joblib
- Cache for trained files, don't wait 2 hours if you want to run again

### [0.6.0] - 2026-04-09
- Semantic Understanding (SBert) integrated
- 3 mode-switch possible: Combined, TF-IDF, Semantic Understanding
- Better results with 92% accuracy

### [0.5.1] - 2026-03-27
- Detailed extraction script
- Ability to see the extracted texts based on their class
- Ability to see the preprocessed texts based on their class

### [0.5.0] - 2026-03-11
- %90 Fallback for the first time ever for the both classes,
- Semantic understanding initialized, needs to get integrated,
- New Model-Types such SVM and Logistic Regression integrated,
- Logistic Regression gives more balanced results,
- Random Forest Classifier works better in unuseful class,
- Tested with 60330 Text files (Logistic Regression),
- Bugs fixed

### [0.4.1] - 2026-01-21
- Logging based on "FILE_TYPE" instead of PDF (tunable at the beginning of main.py)
- Removed a problem due to the mismatching numbers of train/test files
- Tested workflow for .txt files, it works properly 
- 66% accuracy for the text files based on article summaries 
- Needed collecting more and high quality data for better test

### [0.4.0] - 2026-01-20
- **From scratch 8x improvement in PDF detection** (9% -> 72% recall)
- Reached 72% recall for the last test appr. (100 usefull and unusefull PDFs)
- New parent classes added ("Unifiedloader" and "PDFExtractor") for ".txt" detection
- Thus, PDF and TXT classes inherit mutual features from the base class (loader, extractor)
- Workflow improved: In previous version workflow caused inconsistent Problems 
- Removed unnecessary Methods 
- Better SMOTE integration
- More stop words for denoising the data  

### [0.3.0] - 2025-12-02
- **4x improvement in useful PDF detection** (9% -> 36% recall)
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
- Unsupervised learning
- Basic PDF loading functionality

**Status:** **Active Development** 
- Semantic understanding integration
- More and cleaner data

**Last Updated:** 2026-04-29
**Version:** 0.6.2