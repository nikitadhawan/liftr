import torch
from torch import fx, Tensor

from torch.fx.interpreter import Interpreter
from typing import Any, Dict, Tuple, Union

from torch.nn.functional import linear
from torch.distributions.multivariate_normal import MultivariateNormal

from src.models.transformer import QKDotProduct, Softmax, WeightedValues, Add, CausalMask
from src.liftr_modes import LAYER_STATS_DICTS, LAYER_MOMENTS, LAYER_STOCH_OUT
from src.utils import OUTPUT_METRICS
from .base import BaseFsdEstimator


class CustomTracer(fx.Tracer):
    """FX tracer that treats custom attention modules as leaf nodes."""
    def is_leaf_module(self, m, module_qualified_name):
        if isinstance(m, (QKDotProduct, Softmax, WeightedValues, Add, CausalMask)):
            return True
        return super().is_leaf_module(m, module_qualified_name)


class ModeInterpreter(Interpreter):
    """A computation graph interpreter that can run in different modes."""

    def __init__(self, gm: fx.GraphModule, mode: str = "default", **extra_kwargs):
        """Set the computation graph and mode.

        Args:
            gm: A graph module containing the computation graph to interpret.
            mode: The mode of interpretation. Supported modes are:
                - 'default': No changes to the computation graph.
                - 'store_stats': Store precomputed stats for FSD computation.
                - 'fsd': Propagate first two moments for FSD computation.
        """
        super().__init__(gm)
        assert mode in ["default", "store_stats", "determ_fsd", "stoch_fsd"], f"Unknown {mode=}"
        self.mode = mode
        for node in self.module.graph.nodes:
            if node.op not in {"placeholder", "output"}:
                assert node.op == "call_module", f"Node {node} is not a module call, instead a {node.op}."

        self.extra_kwargs = extra_kwargs

    def run(self, *args, **kwargs):
        self.extra_kwargs.update(kwargs)
        return super().run(*args)

    def call_module(
        self, target: str, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Interpret a module call node.

        Args:
            target: The name of the module to call.
            args: Positional arguments to pass to the module.
            kwargs: Keyword arguments to pass to the module.

        Returns:
            The result of the module call, potentially modified by the current mode.
        """
        if self.mode == "default":
            return super().call_module(target, args, kwargs)

        elif self.mode == "store_stats":

            layer = self.module.get_submodule(target)
            (x,) = args
            if isinstance(x, tuple):
                x = tuple(elem.clone().detach() for elem in x)
            else:
                x = x.clone().detach()
            task_id = self.extra_kwargs.get('task_id', None)
            assert task_id is not None
            diag_cov = self.extra_kwargs.get('diag_cov', False)

            stats_dict_fn = LAYER_STATS_DICTS.get(layer.__class__, None)
            if stats_dict_fn is None:
                raise NotImplementedError(f"No stats_dict function for {layer.__class__.__name__}")
            stats_dict = stats_dict_fn(layer, x, task_id, diag_cov)
            layer._stats_dict = stats_dict
            return super().call_module(target, args, kwargs) 

        elif self.mode == "determ_fsd":
            layer = self.module.get_submodule(target)
            ((E_dx, Cov_dx),) = args
            prev_task_id = self.extra_kwargs.get('prev_task_id', None)
            assert prev_task_id is not None
            diag_cov = self.extra_kwargs.get('diag_cov', False)

            moments_fn = LAYER_MOMENTS.get(layer.__class__, None)
            if moments_fn is None:
                raise NotImplementedError(f"No moments function for {layer.__class__.__name__}")
            return moments_fn(layer, E_dx, Cov_dx, prev_task_id, diag_cov)

        elif self.mode == "stoch_fsd":
            layer = self.module.get_submodule(target)
            (dx,) = args
            prev_task_id = self.extra_kwargs.get('prev_task_id', None)
            sample_size = self.extra_kwargs.get('sample_size', None)
            assert prev_task_id is not None
            assert sample_size is not None
            diag_cov = self.extra_kwargs.get('diag_cov', False)

            out_fn = LAYER_STOCH_OUT.get(layer.__class__, None)
            if out_fn is None:
                raise NotImplementedError(f"No stochastic output function for {layer.__class__.__name__}")
            return out_fn(layer, dx, prev_task_id, sample_size, diag_cov)

        else:
            raise NotImplementedError


class LIFTR(BaseFsdEstimator):
    """FSD estimator using linearized moment propagation through the computation graph."""
    def __init__(self, metric_type, reduction, sample_size, stochastic=False, diag_cov=False):
        super().__init__()
        self.metric_type = metric_type
        self.reduction = reduction
        self.sample_size = sample_size
        self.stochastic = stochastic
        self.diag_cov = diag_cov
        self.graph_module = None
    
    def on_task_start(self, model, task_id, **kwargs):
        if not self.graph_module:
            tracer = CustomTracer()
            graph = tracer.trace(model)
            self.graph_module = fx.GraphModule(model, graph)
        
    def on_task_end(self, task_id, model, task_info, **kwargs):
        self.train_loader = kwargs.get('train_loader', None)
        device = kwargs.get('device', None)

        model.eval()
        for i, (inputs, _) in enumerate(self.train_loader):
            if i == 0:
                self.inputs_shape = inputs.shape[1:]
            ModeInterpreter(self.graph_module, mode="store_stats").run(
                inputs.to(device),
                task_id=task_id,
                diag_cov=self.diag_cov,
            )

    
    def get_fsd(self, prev_task_id, model, device): 
        if self.stochastic:
            dx = torch.zeros(
                (self.sample_size,) + self.inputs_shape, 
                device=device
            )
            dout = ModeInterpreter(self.graph_module, mode="stoch_fsd").run(
                dx,
                prev_task_id=prev_task_id,
                sample_size=self.sample_size,
                diag_cov=self.diag_cov,
            )
            dout = dout.reshape((self.sample_size, -1))
            fsd = 0.5 * (dout**2).sum(dim=1).mean()

        else:
            E_dx = torch.zeros(self.inputs_shape, device=device)
            Cov_dx = torch.zeros(
                *self.inputs_shape, *self.inputs_shape,
                device=device
            )
            E_dout, Cov_dout = ModeInterpreter(self.graph_module, mode="determ_fsd").run(
                (E_dx, Cov_dx),
                prev_task_id=prev_task_id,
                diag_cov=self.diag_cov,
            )
            if E_dout.ndim == 2:
                x, y = E_dout.shape
                E_dout = E_dout.flatten()
                Cov_dout = Cov_dout.reshape(x*y, x*y)
            fsd = 0.5 * ((E_dout**2).sum() + Cov_dout.trace())

        return fsd
        