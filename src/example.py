
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from rmt_analyzer import RMTAnalyzer
from rmt_distillation import RMTDistillation

def create_example_model():
    return nn.Sequential(
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

def demonstrate_rmt_analysis():
    print("Creating example model...")
    model = create_example_model()

    with torch.no_grad():
        for param in model.parameters():
            if param.dim() >= 2:
                nn.init.normal_(param, mean=0, std=0.1)

    print("\nInitializing RMT Analyzer...")
    analyzer = RMTAnalyzer(model)

    print("\nPerforming full model analysis...")
    results = analyzer.full_model_analysis()

    print("\nSpectral Metrics Summary:")
    print("-" * 50)
    for layer_name, metrics in results.items():
        print(f"\nLayer: {layer_name}")
        print(f"  Spectral Entropy: {metrics['spectral_entropy']:.4f}")
        print(f"  Stable Rank: {metrics['stable_rank']:.4f}")
        print(f"  Condition Number: {metrics['condition_number']:.2e}")
        print(f"  Spectral Radius: {metrics['spectral_radius']:.4f}")
        print(f"  Matrix Shape: {metrics['matrix_shape']}")

    layer_names = list(results.keys())
    if layer_names:
        print(f"\nGenerating visualization for layer: {layer_names[0]}")
        fig1 = analyzer.visualize_spectrum(layer_names[0])
        plt.show()

        print("\nGenerating layer comparison...")
        fig2 = analyzer.compare_layers()
        plt.show()

    print("\nDemonstrating RMT-based distillation...")
    distiller = RMTDistillation(model)
    student_model = distiller.create_student_architecture(compression_ratio=0.3)

    print(f"\nTeacher model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Student model parameters: {sum(p.numel() for p in student_model.parameters())}")
    compression_ratio = sum(p.numel() for p in student_model.parameters()) / sum(p.numel() for p in model.parameters())
    print(f"Compression ratio: {compression_ratio:.3f}")

    return analyzer, results

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    analyzer, results = demonstrate_rmt_analysis()
    print("\nRMT Analysis complete!")
