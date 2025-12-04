"""
Advanced Confusion Matrix Visualization
Shows both counts and percentages in each cell
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def plot_confusion_matrix_advanced(cm, class_names, output_path='results/confusion_matrix_advanced.png'):
    """
    Plot confusion matrix with counts and percentages
    
    Args:
        cm: Confusion matrix (2x2 numpy array)
        class_names: List of class names ['not_useful', 'useful']
        output_path: Output file path (default: results/confusion_matrix_advanced.png)
    """
    # Calculate percentages (row-wise normalization)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create figure with larger size for better readability
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot heatmap (without default annotations)
    sns.heatmap(
        cm,
        annot=False,  # We'll add custom annotations
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        cbar_kws={'label': 'Count'},
        square=True,
        linewidths=3,
        linecolor='white',
        ax=ax
    )
    
    # Add custom annotations (count + percentage in each cell)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            percent = cm_percent[i, j]
            
            # Choose text color based on cell darkness
            # White text for dark cells, black for light cells
            text_color = 'white' if count > cm.max() / 2 else 'black'
            
            # Add text with count and percentage
            ax.text(
                j + 0.5, i + 0.5,  # Center of cell
                f'{count:,}\n({percent:.1f}%)',  # Format: "54\n(74.0%)"
                ha='center',
                va='center',
                fontsize=24,
                fontweight='bold',
                color=text_color
            )
    
    # Styling: Add titles
    plt.title(
        'Confusion Matrix\n(Count and Row Percentage)', 
        fontsize=22, 
        fontweight='bold', 
        pad=20
    )
    plt.ylabel('Actual Label', fontsize=18, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=18, fontweight='bold')
    
    # Increase tick label size
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    # Calculate metrics
    tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp
    
    accuracy = (tn + tp) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Add metrics box below the matrix
    metrics_text = (
        f'Overall Metrics:\n'
        f'Accuracy: {accuracy:.1%} | '
        f'Precision: {precision:.1%} | '
        f'Recall: {recall:.1%} | '
        f'F1-Score: {f1:.1%}\n'
        f'Total Samples: {total:,}'
    )
    
    plt.text(
        0.5, -0.15,  # Position: center, below matrix
        metrics_text,
        ha='center',
        transform=ax.transAxes,
        fontsize=14,
        bbox=dict(
            boxstyle='round,pad=1', 
            facecolor='lightblue', 
            edgecolor='navy',
            alpha=0.8,
            linewidth=2
        )
    )
    
    # Add cell explanations
    plt.text(
        0.02, 0.98,  # Top-left corner
        'TN: True Negative\nFP: False Positive\nFN: False Negative\nTP: True Positive',
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    # Save figure with high DPI
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Advanced confusion matrix saved to: {output_path}")
    print(f"\nAdvanced confusion matrix visualization saved!")
    print(f"   Location: {output_path}")
    print(f"   Metrics: Acc={accuracy:.1%}, Prec={precision:.1%}, Rec={recall:.1%}, F1={f1:.1%}")

# Optional: Standalone test function
if __name__ == '__main__':
    # Example confusion matrix (for testing)
    cm_example = np.array([[54, 19], [6, 5]])
    class_names = ['not_useful', 'useful']
    
    plot_confusion_matrix_advanced(cm_example, class_names)
    print("\nTest visualization complete! Check results/confusion_matrix_advanced.png")