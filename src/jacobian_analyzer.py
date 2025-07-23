
import torch
import torch.nn as nn
import numpy as np

class JacobianAnalyzer:
    """Analyze Jacobian matrices during training"""

    def __init__(self, model: nn.Module):
        self.model = model
        self.jacobian_data = {}

    def compute_jacobian_spectrum(self, x: torch.Tensor, y: torch.Tensor):
        self.model.eval()
        x.requires_grad_(True)
        output = self.model(x)
        jacobian_eigenvals = {}
        for i in range(output.shape[1]):
            if x.grad is not None:
                x.grad.zero_()
            grad_outputs = torch.zeros_like(output)
            grad_outputs[:, i] = 1.0
            grads = torch.autograd.grad(
                outputs=output,
                inputs=x,
                grad_outputs=grad_outputs,
                retain_graph=True,
                create_graph=False
            )[0]
            jacobian_row = grads.view(grads.shape[0], -1)
            jtj = torch.mm(jacobian_row.t(), jacobian_row) / jacobian_row.shape[0]
            eigenvals = torch.linalg.eigvals(jtj).real.cpu().numpy()
            jacobian_eigenvals[f'output_{i}'] = eigenvals
        return jacobian_eigenvals
