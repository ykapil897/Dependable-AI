import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def visualize_explanations(explanation_file, output_dir):
    """Generate visualizations from explanation data"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Load explanation data
    df = pd.read_csv(explanation_file)
    
    # Group by instance and label
    for (instance_id, label_id), group in df.groupby(['instance_id', 'label_id']):
        # Create feature importance visualization
        plt.figure(figsize=(10, 6))
        
        # Sort by absolute contribution
        group = group.reindex(group['contribution'].abs().sort_values(ascending=False).index)
        
        # Plot horizontal bar chart
        colors = ['green' if x > 0 else 'red' for x in group['contribution']]
        plt.barh(group['feature_id'].astype(str), group['contribution'], color=colors)
        
        plt.xlabel('Feature Contribution')
        plt.ylabel('Feature ID')
        plt.title(f'Top Features for Instance {instance_id}, Label {label_id}')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(f'{output_dir}/explanation_instance_{instance_id}_label_{label_id}.png')
        plt.close()
    
    # Generate summary visualizations
    if len(df) > 0:
        # Top features across all predictions
        top_features = df.groupby('feature_id')['contribution'].apply(lambda x: np.sum(np.abs(x)))
        top_features = top_features.sort_values(ascending=False).head(20)
        
        plt.figure(figsize=(12, 8))
        top_features.plot(kind='bar')
        plt.xlabel('Feature ID')
        plt.ylabel('Total Absolute Contribution')
        plt.title('Most Influential Features Overall')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/top_features_overall.png')
        plt.close()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python visualize_explanations.py <explanation_file> <output_dir>")
        sys.exit(1)
    
    explanation_file = sys.argv[1]
    output_dir = sys.argv[2]
    visualize_explanations(explanation_file, output_dir)