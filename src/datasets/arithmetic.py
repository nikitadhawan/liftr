import random
import torch
from torch.utils.data import TensorDataset
import torch.nn.functional as F


class Arithmetic:

    def __init__(self,
                 seq_len: int = 3,
                 num_tasks: int = 2,
                 p: int = 113,
                 num_samples: int = 2000,
                 frac_train: int = 0.5,
                 seed: int = 0):
        """Dataset of arithmetic tasks modulo p.

        Each task t samples pairs (a, b) and the label is f_t(a, b) mod p,
        where f_t is chosen from a set of functions (add, subtract, x2xyy2, rand).
        Inputs are integer tokens; only the last time step is supervised.
        """
        self.num_tasks = num_tasks
        self.seq_len = seq_len
        self.p = int(p)
        self.cur_iter = 0

        self.random_answers = torch.randint(low=0, high=self.p, size=(self.p, self.p), dtype=torch.long)
        self.fns = self.fns_dict()
        task_order = ['add', 'subtract', 'x2xyy2', 'rand']
        self.task_names = task_order[:num_tasks]
        self.fn_to_token = {fn: self.p + i for i, fn in enumerate(task_order)}

        # Useful for configuring the model: vocabulary/output size required
        self.vocab_size = self.p + num_tasks

        self.num_samples = num_samples
        self.frac_train = frac_train

        random.seed(seed)
        torch.manual_seed(seed)

    def fns_dict(self):
        """Return the dictionary of arithmetic functions keyed by task name."""
        return {
            'add': lambda x, y: (x + y) % self.p,
            'subtract': lambda x, y: (x - y) % self.p,
            'x2xyy2': lambda x, y: (x**2 + x * y + y**2) % self.p,
            'rand': lambda x, y: self.random_answers[x, y]
        }

    def _gen_split(self, fn_name: str, num_samples: int, frac_train: float):
        """Generate train/test TensorDatasets for a single arithmetic task."""
        all_pairs = torch.cartesian_prod(torch.arange(self.p), torch.arange(self.p))
        total_pairs = all_pairs.shape[0]
        indices = torch.randperm(total_pairs)[:num_samples]
        sampled_pairs = all_pairs[indices]
        a, b = sampled_pairs[:, 0], sampled_pairs[:, 1]

        X = torch.zeros((num_samples, self.seq_len), dtype=torch.long)
        fn_token = self.fn_to_token[fn_name]
        if self.seq_len < 3:
            raise ValueError("seq_len must be >= 3 for arithmetic dataset")
        X[:, -3] = a
        X[:, -2] = fn_token
        X[:, -1] = b
        Y = self.fns[fn_name](a, b)
        num_train = int(frac_train*num_samples)
        X_train, X_test = X[:num_train], X[num_train:]
        Y_train, Y_test = Y[:num_train], Y[num_train:]
        return TensorDataset(X_train, Y_train), TensorDataset(X_test, Y_test)

    def next_task(self):
        """Return (train_dataset, test_dataset, task_info) for the next task."""
        if self.cur_iter >= self.num_tasks:
            raise Exception('Number of tasks exceeded!')
        fn_name = self.task_names[self.cur_iter]
        train_ds, test_ds = self._gen_split(fn_name, self.num_samples, self.frac_train)
        task_info = {"fn": fn_name, "p": self.p}
        self.cur_iter += 1
        return train_ds, test_ds, task_info

    def reset(self):
        """Reset the task iterator to the first task."""
        self.cur_iter = 0

    def evaluate(self, outputs, targets, task_info=None):
        """Compute accuracy on the final token prediction."""
        if outputs.dim() == 3:
            logits = outputs[:, -1, :]
        else:
            logits = outputs
        preds = torch.argmax(logits, dim=-1)
        correct = (preds == targets).sum().item()
        total = targets.size(0)
        return correct / total

    def loss_fn(self, outputs, targets, task_info=None):
        """Cross-entropy loss on the final token prediction."""
        if outputs.dim() == 3:
            logits = outputs[:, -1, :]
        else:
            logits = outputs
        return F.cross_entropy(logits, targets)