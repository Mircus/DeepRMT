
import matplotlib.pyplot as plt
import numpy as np

def visualize_spectrum(layer_name: str, metrics: dict, eigenvals: np.ndarray, singular_vals: np.ndarray, figsize=(15, 10)):
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle(f'Spectral Analysis: {layer_name}', fontsize=16)

    axes[0, 0].hist(eigenvals, bins=50, alpha=0.7, density=True, label='Empirical')
    mp_x, mp_density = metrics['mp_theoretical']
    if len(mp_x) > 0:
        axes[0, 0].plot(mp_x, mp_density, 'r-', label='Marchenko-Pastur', linewidth=2)
    axes[0, 0].set_title('Eigenvalue Distribution vs Marchenko-Pastur')
    axes[0, 0].set_xlabel('Eigenvalue')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].hist(eigenvals, bins=50, alpha=0.7, density=True, label='Empirical')
    wigner_x, wigner_density = metrics['wigner_theoretical']
    axes[0, 1].plot(wigner_x, wigner_density, 'g-', label='Wigner Semicircle', linewidth=2)
    axes[0, 1].set_title('Eigenvalue Distribution vs Wigner Semicircle')
    axes[0, 1].set_xlabel('Eigenvalue')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(range(len(singular_vals)), singular_vals, 'bo-', markersize=3)
    axes[0, 2].set_title('Singular Value Spectrum')
    axes[0, 2].set_xlabel('Index')
    axes[0, 2].set_ylabel('Singular Value')
    axes[0, 2].set_yscale('log')
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].scatter(np.real(eigenvals), np.imag(eigenvals), alpha=0.6, s=10)
    axes[1, 0].set_title('Eigenvalues in Complex Plane')
    axes[1, 0].set_xlabel('Real Part')
    axes[1, 0].set_ylabel('Imaginary Part')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[1, 0].axvline(x=0, color='k', linestyle='-', alpha=0.3)

    metric_names = ['Spectral Entropy', 'Stable Rank', 'Condition Number', 'Spectral Radius']
    metric_values = [
        metrics['spectral_entropy'],
        metrics['stable_rank'],
        np.log10(metrics['condition_number']) if metrics['condition_number'] != np.inf else 10,
        metrics['spectral_radius']
    ]
    axes[1, 1].bar(metric_names, metric_values)
    axes[1, 1].set_title('Spectral Metrics')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)

    info_text = f"""
    Matrix Shape: {metrics['matrix_shape']}
    Aspect Ratio: {metrics['aspect_ratio']:.3f}
    Nuclear Norm: {metrics['nuclear_norm']:.3f}
    # Eigenvalues: {metrics['num_eigenvals']}
    # Singular Values: {metrics['num_singular_vals']}
    """
    axes[1, 2].text(0.1, 0.5, info_text, transform=axes[1, 2].transAxes, fontsize=10, verticalalignment='center')
    axes[1, 2].set_title('Layer Information')
    axes[1, 2].axis('off')

    plt.tight_layout()
    return fig
