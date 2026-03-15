import torch
from src.utils import OUTPUT_METRICS
from .base import BaseFsdEstimator

class RandomSubset(BaseFsdEstimator):
    """FSD estimator using output differences on a random subset of training data."""
    def __init__(self, metric_type, reduction, sample_size):
        super().__init__()
        self.metric_type = metric_type
        self.reduction = reduction
        self.sample_size = sample_size
        self.task_subset = {}
        self.task_params = {}
        self.task_infos = {}

    def on_task_end(self, task_id, model, task_info, **kwargs):
        train_loader = kwargs.get('train_loader', None)
        subset_inputs = []
        subset_labels = []
        total_collected = 0
        for inputs, labels in train_loader:
            if total_collected >= self.sample_size:
                break
            remaining = self.sample_size - total_collected
            if inputs.shape[0] > remaining:
                subset_inputs.append(inputs[:remaining])
                subset_labels.append(labels[:remaining])
                total_collected += remaining
                break
            else:
                subset_inputs.append(inputs)
                subset_labels.append(labels)
                total_collected += inputs.shape[0]

        subset_inputs = torch.cat(subset_inputs, dim=0)
        subset_labels = torch.cat(subset_labels, dim=0)
        assert subset_inputs.shape[0] == self.sample_size
        assert subset_labels.shape[0] == self.sample_size
        self.task_subset[task_id] = (subset_inputs, subset_labels)
        self.task_infos[task_id] = task_info
        self.task_params[task_id] = {name: param.data.clone() for name, param in model.named_parameters()}

    def get_fsd(self, prev_task_id, model, device):
        params0 = self.task_params[prev_task_id]
        params1 = {name: param for name, param in model.named_parameters()}
        inputs, _ = self.task_subset[prev_task_id]
        output_metric = OUTPUT_METRICS[self.metric_type]
        task_fsd = torch.tensor(0.0, device=device)
        inputs = inputs.to(device)
        with torch.no_grad():
            out0 = torch.func.functional_call(model, params0, (inputs,))
        out1 = torch.func.functional_call(model, params1, (inputs,))
        fsd = output_metric(out0, out1, reduce=self.reduction)
        task_fsd = task_fsd + fsd
        task_fsd = task_fsd / inputs.shape[0]
        return task_fsd