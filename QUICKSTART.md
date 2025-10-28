# Quick Start Guide

## Your PDF Classifier is Ready! 🎉

### What You Have:
✅ Complete anomaly detection pipeline
✅ 50/50 train-test split (150 PDFs each)
✅ TF-IDF feature extraction
✅ Isolation Forest model
✅ Jupyter notebook for exploration
✅ All source code files

### Next Steps:

#### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

Required packages:
- pdfplumber (PDF text extraction)
- scikit-learn (machine learning)
- pandas, numpy (data processing)
- matplotlib, seaborn (visualization)

#### 2. Add Your PDFs
Place your 300 "not useful" PDFs in:
```
data/raw_pdfs/
```

#### 3. Run the Pipeline
```bash
python main.py
```

This will:
- Load and split your 300 PDFs (150 train, 150 test)
- Extract text from all PDFs
- Clean and preprocess the text
- Create TF-IDF features
- Train Isolation Forest on training set
- Predict anomalies on test set
- Save results to `results/predictions.csv`

#### 4. View Results
Open `results/predictions.csv`:
- **prediction = 0**: "not useful" (normal, similar to training)
- **prediction = 1**: "useful" (anomaly, different from training)
- **anomaly_score**: Lower = more anomalous (more likely useful)

#### 5. Explore (Optional)
```bash
jupyter notebook notebooks/exploration.ipynb
```

### Testing Individual Modules

Before running the full pipeline, test modules:
```bash
python test_modules.py
```

### Module Overview:

**loader.py** - Loads PDFs from directory, splits train/test
**extractor.py** - Extracts text from PDFs (uses pdfplumber or PyPDF2)
**preprocess.py** - Cleans text (removes URLs, special chars, normalizes)
**features.py** - Creates TF-IDF feature vectors
**model.py** - Trains/predicts with Isolation Forest
**utils.py** - Helper functions (save results, etc.)
**main.py** - Orchestrates the complete pipeline

### How Anomaly Detection Works:

Since you only have "not useful" PDFs:
1. **Train** on 150 "not useful" PDFs → model learns what "not useful" looks like
2. **Test** on 150 "not useful" PDFs → documents that are different get flagged
3. **Contamination = 10%** → expects ~15 PDFs to be flagged as anomalies

### Adjusting Parameters:

In `src/model.py`, adjust contamination:
```python
detector = AnomalyDetector(contamination=0.05)  # Expect 5% anomalies
detector = AnomalyDetector(contamination=0.15)  # Expect 15% anomalies
```

In `src/features.py`, adjust features:
```python
extractor = FeatureExtractor(
    max_features=1000,    # More features
    ngram_range=(1, 3)    # Include trigrams
)
```

### If Results Aren't Good:

1. **Check PDFs**: Ensure they contain extractable text
2. **Adjust contamination**: Try 0.05, 0.10, 0.15, 0.20
3. **More features**: Increase max_features to 1500 or 2000
4. **Collect useful PDFs**: If possible, get examples of "useful" PDFs
5. **Switch approach**: Use binary classification with both classes

### Need Help?

Check the README.md for full documentation!

---

**Ready to go!** Just add your PDFs and run `python main.py` 🚀
