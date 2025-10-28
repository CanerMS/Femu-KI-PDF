# PDF Classifier - Anomaly Detection

A machine learning project to identify "not useful" PDFs using anomaly detection with Isolation Forest.

##  Project Overview

- **Goal**: Automatically classify PDFs as "useful" or "not useful"
- **Approach**: Anomaly detection using Isolation Forest
- **Data Split**: 50/50 train-test split (150 PDFs each)
- **Features**: TF-IDF text representations

##  Project Structure

```
pdf_classifier_project/

 data/
    raw_pdfs/              # Place your 300 PDF files here
    extracted_texts/       # Extracted text (auto-generated)
    labels.csv             # Labels template

 src/
    loader.py              # PDF file loading
    extractor.py           # Text extraction from PDFs
    preprocess.py          # Text cleaning and preprocessing
    features.py            # TF-IDF feature extraction
    model.py               # Isolation Forest anomaly detection
    utils.py               # Helper functions

 notebooks/
    exploration.ipynb      # Data exploration and visualization

 results/
    predictions.csv        # Classification results (auto-generated)

 main.py                    # Main pipeline
 requirements.txt           # Python dependencies
```

##  Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Add Your PDFs
Place your 300 "not useful" PDFs in the `data/raw_pdfs/` folder.

### 3. Run the Pipeline
```bash
python main.py
```

The pipeline will:
1. Load all PDFs from `data/raw_pdfs/`
2. Split into 150 training, 150 testing
3. Extract text from PDFs
4. Clean and preprocess text
5. Extract TF-IDF features
6. Train Isolation Forest on training set
7. Predict on test set
8. Save results to `results/predictions.csv`

### 4. View Results
Check `results/predictions.csv`:
- `filename`: PDF filename
- `prediction`: 0 (not useful) or 1 (useful/anomaly)
- `anomaly_score`: Lower scores = more anomalous (more likely useful)
- `label`: Human-readable label

### 5. Explore in Jupyter (Optional)
```bash
jupyter notebook notebooks/exploration.ipynb
```

##  How Anomaly Detection Works

Since you only have "not useful" PDFs:

1. **Training Phase**: The model learns the patterns of "not useful" PDFs
2. **Testing Phase**: Documents that deviate from these patterns are flagged as anomalies (potentially "useful")
3. **Contamination**: Set to 10% by default (expects ~10% of test data to be anomalies)

##  Configuration

Adjust parameters in `src/model.py`:

```python
AnomalyDetector(
    contamination=0.1,    # Expected % of useful PDFs (0.1 = 10%)
    random_state=42       # For reproducibility
)
```

Adjust features in `src/features.py`:

```python
FeatureExtractor(
    max_features=1000,    # Max number of TF-IDF features
    ngram_range=(1, 2)    # Unigrams and bigrams
)
```

##  Understanding the Output

**Anomaly Scores**:
- **Lower scores** (more negative): Strong anomalies, very different from training data  likely "useful"
- **Higher scores** (closer to 0): Similar to training data  likely "not useful"

**Example**:
```csv
filename,prediction,anomaly_score,label
doc1.pdf,0,0.15,not_useful      # Normal, similar to training
doc2.pdf,1,-0.42,useful         # Anomaly, very different
```

##  Next Steps

If anomaly detection doesn't work well:

1. **Adjust contamination**: Try different values (0.05, 0.15, 0.2)
2. **Collect useful PDFs**: Gather examples of "useful" PDFs
3. **Switch to binary classification**: Use both classes for supervised learning
4. **Feature engineering**: Add custom features (document length, keyword counts, etc.)

##  Requirements

- Python 3.8+
- pdfplumber or PyPDF2 for PDF text extraction
- scikit-learn for machine learning
- pandas, numpy for data handling
- matplotlib, seaborn for visualization

##  Troubleshooting

**No PDFs found?**
- Ensure PDFs are in `data/raw_pdfs/` directory
- Check file extensions are `.pdf`

**Import errors?**
- Run: `pip install -r requirements.txt`

**Poor performance?**
- Try adjusting the `contamination` parameter
- Check if PDFs contain extractable text
- Review text in `data/extracted_texts/`

##  License

This project is open source and available for educational purposes.
