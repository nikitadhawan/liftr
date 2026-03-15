import torch
from src.utils import OUTPUT_METRICS
from .base import BaseFsdEstimator

class GroundTruth(BaseFsdEstimator):
    """Exact FSD by evaluating the model at both the old and new parameters."""
    def __init__(self, metric_type, reduction, stochastic):
        super().__init__()
        self.metric_type = metric_type
        self.reduction = reduction
        self.stochastic = stochastic
        self.task_loaders = {}
        self.task_params = {}
        self.task_infos = {}

    def on_task_end(self, task_id, model, task_info, **kwargs):
        train_loader = kwargs.get('train_loader', None)
        self.task_loaders[task_id] = train_loader
        self.task_infos[task_id] = task_info
        self.task_params[task_id] = {name: param.data.clone() for name, param in model.named_parameters()}

    def get_fsd(self, prev_task_id, model, device):
        params0 = self.task_params[prev_task_id]
        params1 = {name: param for name, param in model.named_parameters()}
        loader = self.task_loaders[prev_task_id]
        output_metric = OUTPUT_METRICS[self.metric_type]
        if self.stochastic:
            inputs, _ = next(iter(loader))
            inputs = inputs.to(device)
            with torch.no_grad():
                out0 = torch.func.functional_call(model, params0, (inputs,))
            out1 = torch.func.functional_call(model, params1, (inputs,))
            task_fsd = output_metric(out0, out1, reduce=self.reduction) / inputs.shape[0]
        else:
            count = 0
            task_fsd = torch.tensor(0.0, device=device)
            for i, (inputs, _) in enumerate(loader):
                inputs = inputs.to(device)
                with torch.no_grad():
                    out0 = torch.func.functional_call(model, params0, (inputs,))
                out1 = torch.func.functional_call(model, params1, (inputs,))
                fsd = output_metric(out0, out1, reduce=self.reduction)
                task_fsd = task_fsd + fsd
                count += inputs.shape[0]
            task_fsd = task_fsd / count
        return task_fsd