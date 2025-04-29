import numpy as np
import sys
import os

def generate_adversarial_examples(feature_file, label_file, output_file, 
                                 perturbation_strength=0.1, num_features=1024):
    """
    Generate adversarial examples for testing model robustness.
    
    Args:
        feature_file: Path to test features
        label_file: Path to test labels  
        output_file: Where to save adversarial examples
        perturbation_strength: Magnitude of adversarial perturbation
    """
    # Read feature dimensions
    with open(feature_file, 'r') as f:
        header = f.readline().strip().split()
        num_cols, num_rows = int(header[0]), int(header[1])
    
    # Load features
    features = np.zeros((num_cols, num_rows))
    with open(feature_file, 'r') as f:
        f.readline()  # Skip header
        for i, line in enumerate(f):
            if i >= num_cols:
                break
            values = [float(x) for x in line.strip().split()]
            features[i, :] = values
    
    # Create adversarial examples by adding random noise
    # More sophisticated approaches would target specific features
    # but this is a simple baseline
    np.random.seed(42)  # For reproducibility
    noise = np.random.normal(0, perturbation_strength, features.shape)
    
    # Add noise to create adversarial examples
    adversarial_features = features + noise
    
    # Ensure values remain in a reasonable range
    adversarial_features = np.clip(adversarial_features, 0, None)
    
    # Save adversarial examples
    with open(output_file, 'w') as f:
        f.write(f"{num_cols} {num_rows}\n")
        for i in range(num_cols):
            f.write(" ".join([str(x) for x in adversarial_features[i, :]]) + "\n")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python generate_adversarial.py <feature_file> <label_file> <output_file>")
        sys.exit(1)
    
    feature_file = sys.argv[1]
    label_file = sys.argv[2]
    output_file = sys.argv[3]
    
    generate_adversarial_examples(feature_file, label_file, output_file)