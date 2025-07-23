
import torch
import torch.nn as nn
import torch.nn.functional as F
from .rmt_analyzer import RMTAnalyzer

class RMTDistillation:
    """Model distillation based on RMT insights"""

    def __init__(self, teacher_model: nn.Module):
        self.teacher_model = teacher_model
        self.teacher_analyzer = RMTAnalyzer(teacher_model)

    def analyze_teacher(self):
        return self.teacher_analyzer.full_model_analysis()

    def create_student_architecture(self, compression_ratio: float = 0.5) -> nn.Module:
        teacher_analysis = self.analyze_teacher()
        student_layers = []
        for layer_name, param in self.teacher_model.named_parameters():
            if 'weight' in layer_name and param.dim() >= 2:
                metrics = teacher_analysis.get(layer_name, {})
                stable_rank = metrics.get('stable_rank', param.shape[1])
                if param.dim() == 2:
                    in_features, out_features = param.shape[1], param.shape[0]
                    effective_rank = min(stable_rank, min(in_features, out_features))
                    compressed_features = max(1, int(effective_rank * compression_ratio))
                    student_layers.append(nn.Linear(in_features, compressed_features))
        if student_layers:
            return nn.Sequential(*student_layers)
        else:
            return nn.Sequential(
                nn.Linear(784, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 10)
            )

    def distillation_loss(self, student_output: torch.Tensor, teacher_output: torch.Tensor, 
                         true_labels: torch.Tensor, temperature: float = 3.0, 
                         alpha: float = 0.7) -> torch.Tensor:
        distillation_loss = F.kl_div(
            F.log_softmax(student_output / temperature, dim=1),
            F.softmax(teacher_output / temperature, dim=1),
            reduction='batchmean'
        ) * (temperature ** 2)
        classification_loss = F.cross_entropy(student_output, true_labels)
        total_loss = alpha * distillation_loss + (1 - alpha) * classification_loss
        return total_loss

    def spectral_regularization(self, model: nn.Module, lambda_reg: float = 1e-4) -> torch.Tensor:
        reg_loss = 0.0
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                singular_vals = torch.linalg.svdvals(param)
                reg_loss += lambda_reg * torch.sum(singular_vals)
                reg_loss += lambda_reg * singular_vals[0]
        return reg_loss
