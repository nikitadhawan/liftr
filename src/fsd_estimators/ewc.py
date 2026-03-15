from copy import deepcopy
import torch
from .base import BaseFsdEstimator

class EWC(BaseFsdEstimator):
    """Elastic Weight Consolidation: Fisher-weighted quadratic parameter penalty."""
    def __init__(self):
        super().__init__()
        self.optpar_dicts = {}
        self.fisher_dicts = {}

    def on_task_end(self, task_id, model, task_info, **kwargs):
        optpar_dict = {}
        fisher_dict = {}
        params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        for name, param in deepcopy(params).items():
            optpar_dict[name] = param.detach().clone()
            param.data.zero_()
            fisher_dict[name] = param.detach().clone()
        model.eval()
        train_loader = kwargs.get('train_loader', None)
        device = kwargs.get('device', None)
        loss_fn = kwargs.get('loss_fn', None)
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            model.zero_grad()
            logits = model.forward(inputs)
            loss = loss_fn(logits, labels, task_info=task_info)
            loss.backward()
            
            for name, param in params.items():
                fisher_dict[name].data += param.grad.data.clone().pow(2) / len(train_loader)

        self.optpar_dicts[task_id] = optpar_dict
        self.fisher_dicts[task_id] = fisher_dict

    def get_fsd(self, prev_task_id, model, device):
        model.train()
        task_fsd = torch.tensor(0.0, device=device)
        fisher_dict = self.fisher_dicts[prev_task_id]
        optpar_dict = self.optpar_dicts[prev_task_id]
        params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        for name, param in params.items():
            fisher = fisher_dict[name]
            optpar = optpar_dict[name]
            task_fsd = task_fsd + (fisher * (optpar - param).pow(2)).sum()
        return task_fsd

    