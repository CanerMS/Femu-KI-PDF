"""
Feature Analysis and Visualization
"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from project_config import RESULTS_DIR
results_dir = RESULTS_DIR

def analyze_features(): # Analyzes and visualizes top features from feature extraction
    results_dir = Path(results_dir) # Directory where results are stored
    
    # Read selected features
    features_df = pd.read_csv(results_dir / 'selected_features.csv') # Read selected features CSV
    print(f"Total features: {len(features_df)}") # Print total number of features
    print(f"\nFirst 50 features:") # Print first 50 features
    print(features_df.head(50)) # Display first 50 features
    
    # Read top features for each class
    useful_features = pd.read_csv(results_dir / 'top_useful_features.csv') # Read top useful features CSV
    not_useful_features = pd.read_csv(results_dir / 'top_not_useful_features.csv') # Read top not useful features CSV
    
    print(f"\n{'='*60}")
    print("TOP 20 USEFUL FEATURES:")
    print('='*60)
    for idx, row in useful_features.head(20).iterrows(): # Iterate over top 20 useful features
        print(f"{idx+1:2d}. {row['feature_name']:25s} | diff: {row['difference']:+.4f}") # Print feature details
    
    print(f"\n{'='*60}") 
    print("TOP 20 NOT_USEFUL FEATURES:")
    print('='*60)
    for idx, row in not_useful_features.head(20).iterrows(): # Iterate over top 20 not useful features
        print(f"{idx+1:2d}. {row['feature_name']:25s} | diff: {row['difference']:+.4f}") # Print feature details
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6)) # Create subplots for visualization
    
    # Useful features
    top20_useful = useful_features.head(20) # Get top 20 useful features
    ax1.barh(range(len(top20_useful)), top20_useful['difference'], color='green', alpha=0.7) # Horizontal bar plot
    ax1.set_yticks(range(len(top20_useful))) # Set y-ticks
    ax1.set_yticklabels(top20_useful['feature_name']) # Set y-tick labels
    ax1.set_xlabel('TF-IDF Difference') # Set x-label
    ax1.set_title('Top 20 USEFUL Features', fontsize=14, fontweight='bold') # Set title
    ax1.invert_yaxis() # Invert y-axis for better readability
    ax1.grid(axis='x', alpha=0.3) # Add grid lines
    
    # Not_useful features
    top20_not_useful = not_useful_features.head(20) # Get top 20 not useful features
    ax2.barh(range(len(top20_not_useful)), top20_not_useful['difference'], color='red', alpha=0.7) # Horizontal bar plot
    ax2.set_yticks(range(len(top20_not_useful))) # Set y-ticks
    ax2.set_yticklabels(top20_not_useful['feature_name']) # Set y-tick labels
    ax2.set_xlabel('TF-IDF Difference') # Set x-label
    ax2.set_title('Top 20 NOT_USEFUL Features', fontsize=14, fontweight='bold') # Set title
    ax2.invert_yaxis() # Invert y-axis for better readability
    ax2.grid(axis='x', alpha=0.3) # Add grid lines
    
    plt.tight_layout() # Adjust layout to prevent overlap
    plt.savefig(results_dir / 'feature_importance.png', dpi=300, bbox_inches='tight') # Save figure
    print(f"\nVisualization saved to: {results_dir / 'feature_importance.png'}") # Print save location
    plt.show() # Show plot
    
    # Gürültü tespiti
    print(f"\n{'='*60}")
    print("NOISE DETECTION:") 
    print('='*60)
    
    useful_names = set(useful_features['feature_name'].head(50)) # Get top 50 useful feature names
    not_useful_names = set(not_useful_features['feature_name'].head(50)) # Get top 50 not useful feature names
    noise_features = useful_names & not_useful_names # Find intersection (noise features)
    
    if noise_features: 
        print(f"Found {len(noise_features)} noise features:") # Print count of noise features
        for feature in sorted(noise_features): # Print each noise feature
            print(f"   - {feature}") 
    else:
        print("No noise features! Classes are well separated.")

if __name__ == '__main__':
    analyze_features() # Run analysis