import torch

from .base import BaseFsdEstimator
from src.utils import OUTPUT_METRICS

class NTK(BaseFsdEstimator):
    """Neural Tangent Kernel approximation: FSD via linearized model on a stored coreset."""
    def __init__(self, metric_type, reduction, sample_size):
        super().__init__()
        self.metric_type = metric_type
        self.reduction = reduction
        self.sample_size = sample_size
        self.task_params = {}
        self.coresets = {}
        self.task_infos = {}
        self.out0s = {}
        self.model_lins = {}

    
    def on_task_end(self, task_id, model, task_info, **kwargs):
        train_loader = kwargs.get('train_loader', None)
        device = kwargs.get('device', None)
        subset_inputs = []
        total_collected = 0
        for inputs, _ in train_loader:
            if total_collected >= self.sample_size:
                break
            remaining = self.sample_size - total_collected
            if inputs.shape[0] > remaining:
                subset_inputs.append(inputs[:remaining])
                total_collected += remaining
                break
            else:
                subset_inputs.append(inputs)
                total_collected += inputs.shape[0]
        subset_inputs = torch.cat(subset_inputs, dim=0)
        assert subset_inputs.shape[0] == self.sample_size
        self.coresets[task_id] = subset_inputs.to(device)
        self.task_infos[task_id] = task_info
        self.task_params[task_id] = {name: param.detach().clone() for name, param in model.named_parameters()}

        params0 = self.task_params[task_id]
        def model_fn(params, inputs):
            return torch.func.functional_call(model, params, (inputs,))
        out0, model_lin = torch.func.linearize(
            lambda p: model_fn(p, self.coresets[task_id]),
            params0
        )
        self.out0s[task_id] = out0
        self.model_lins[task_id] = model_lin

    def get_fsd(self, prev_task_id, model, device):
        params1 = {name: param for name, param in model.named_parameters()}
        output_metric = OUTPUT_METRICS[self.metric_type]
        
        params0 = self.task_params[prev_task_id]
        out0 = self.out0s[prev_task_id].to(device)
        model_lin = self.model_lins[prev_task_id]       
        delta_params = {k: params1[k] - params0[k] for k in params0}
        out1 = out0 + model_lin(delta_params)
        out0 = out0.reshape((self.sample_size, -1))
        out1 = out1.reshape((self.sample_size, -1))
        task_fsd = output_metric(out0, out1, reduce=self.reduction)
        return task_fsd
