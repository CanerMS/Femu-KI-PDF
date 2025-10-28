"""
Test Script
Quick tests for individual modules
"""
import sys
from pathlib import Path

sys.path.insert(0, 'src')

def test_loader():
    print("\n=== Testing PDF Loader ===")
    from loader import PDFLoader
    
    try:
        loader = PDFLoader()
        pdf_files = loader.get_pdf_files()
        print(f" Found {len(pdf_files)} PDF files")
        
        if len(pdf_files) > 0:
            train, test = loader.split_train_test()
            print(f" Split: {len(train)} train, {len(test)} test")
        return True
    except Exception as e:
        print(f" Error: {e}")
        return False

def test_extractor():
    print("\n=== Testing Text Extractor ===")
    from loader import PDFLoader
    from extractor import PDFExtractor
    
    try:
        loader = PDFLoader()
        pdf_files = loader.get_pdf_files()
        
        if len(pdf_files) == 0:
            print(" No PDFs to test. Skipping.")
            return True
        
        extractor = PDFExtractor()
        text = extractor.extract_text_from_pdf(pdf_files[0])
        print(f" Extracted {len(text)} characters from {pdf_files[0].name}")
        return True
    except Exception as e:
        print(f" Error: {e}")
        return False

def test_preprocessor():
    print("\n=== Testing Text Preprocessor ===")
    from preprocess import TextPreprocessor
    
    try:
        preprocessor = TextPreprocessor()
        test_text = "This is a TEST with URLs http://test.com and special!!!chars"
        cleaned = preprocessor.clean_text(test_text)
        print(f" Original: {test_text}")
        print(f" Cleaned: {cleaned}")
        return True
    except Exception as e:
        print(f" Error: {e}")
        return False

def test_features():
    print("\n=== Testing Feature Extractor ===")
    from features import FeatureExtractor
    
    try:
        extractor = FeatureExtractor(max_features=10)
        texts = ["this is document one", "this is document two", "another document here"]
        features = extractor.fit_transform(texts)
        print(f" Created feature matrix: {features.shape}")
        return True
    except Exception as e:
        print(f" Error: {e}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("Running Module Tests")
    print("="*60)
    
    results = []
    results.append(("Loader", test_loader()))
    results.append(("Extractor", test_extractor()))
    results.append(("Preprocessor", test_preprocessor()))
    results.append(("Features", test_features()))
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    for name, passed in results:
        status = " PASSED" if passed else " FAILED"
        print(f"{name}: {status}")
    
    all_passed = all(r[1] for r in results)
    print("\n" + ("="*60))
    if all_passed:
        print("All tests passed! Ready to run main.py")
    else:
        print("Some tests failed. Check errors above.")
    print("="*60)
