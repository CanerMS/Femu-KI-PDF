# PDF Classifier - Supervised Machine Learning

Automatically classify PDFs as "useful" or "not useful" using Random Forest classifier with SMOTE balancing.

## 🎯 Features

- **Supervised Learning**: Random Forest with class balancing
- **SMOTE**: Handles severe class imbalance (43:290 ratio)
- **Dual-Folder Support**: Separate folders for useful/not_useful PDFs
- **Automated Pipeline**: From PDF → Text → Features → Predictions
- **Comprehensive Evaluation**: Accuracy, precision, recall, confusion matrix

## 📊 Your Data

- **Useful PDFs**: 43 (13%)
- **Not Useful PDFs**: 290 (87%)
- **Train/Test Split**: 75%/25% stratified
- **SMOTE Applied**: Balances training data automatically

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Organize Your PDFs
```
data/
  raw_pdfs/          # Place "not useful" PDFs here (290 files)
  useful_pdfs/       # Place "useful" PDFs here (43 files)
```

### 3. Create Labels
```bash
python src/label_pdfs.py
```

Creates `data/labels.csv` with balanced train/test splits.

### 4. Run Pipeline
```bash
python main.py
```

## 📈 Expected Results

With your 43:290 imbalance:

```
Training: 217 PDFs (32 useful, 185 not_useful)
After SMOTE: 370 PDFs (185 useful, 185 not_useful) - balanced!

Testing: 73 PDFs (11 useful, 62 not_useful)

Expected Accuracy: 80-90%
Useful PDF Recall: 70-85%
```

## 🔧 Configuration

### Adjust SMOTE Threshold
```python
# In main.py line 90
if not_useful_count / useful_count > 5:  # Change threshold (default: 5)
```

### Adjust Features
```python
# In main.py line 84
feature_extractor = FeatureExtractor(max_features=2000)  # Increase for better accuracy
```

### Adjust Model
```python
# In model.py lines 89-94
RandomForestClassifier(
    n_estimators=200,        # More trees = better (but slower)
    max_depth=15,            # Deeper = more complex patterns
    min_samples_split=3,     # Lower = more flexible
    class_weight='balanced'  # Keep this for imbalance
)
```

## 📂 Output Files

- `results/predictions.csv` - Classification results
- `results/pdf_classifier.joblib` - Trained model
- `data/extracted_texts/` - Extracted PDF text
- `logs/label_pdfs.log` - Labeling process logs

## 🐛 Troubleshooting

**"Loaded 0 useful PDFs"**
- Ensure useful PDFs are in `data/useful_pdfs/`
- Run `python src/label_pdfs.py` to regenerate labels

**Low Accuracy (<70%)**
- Increase `max_features` to 3000-5000
- Try `n_estimators=200` in Random Forest
- Check if PDFs have extractable text

**ImportError: imblearn**
```bash
pip install imbalanced-learn
```