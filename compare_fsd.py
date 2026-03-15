import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
import wandb
import csv
import os

import numpy as np
from scipy.stats import spearmanr, pearsonr
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.fsd_estimators.ground_truth import GroundTruth
from src.utils import OPTIMIZERS, set_random_seed


def grad_cosine_sim(true_fsd, fsd_estimate, model):
    """Compute cosine similarity between gradients of true and estimated FSD."""
    params = [p for p in model.parameters() if p.requires_grad]
    grad_true = torch.autograd.grad(
        true_fsd,
        params,
        retain_graph=True,     
        create_graph=False
    )
    grad_est = torch.autograd.grad(
        fsd_estimate,
        params,
        retain_graph=False,
        create_graph=False
    )


    def flatten(grads):
        return torch.cat([g.reshape(-1) for g in grads])
    g_true = flatten(grad_true)
    g_est = flatten(grad_est)

    cos_sim = F.cosine_similarity(g_true, g_est, dim=0)
    return cos_sim.item()


@hydra.main(config_path="conf/", config_name="config.yaml", version_base="1.2")
def main(cfg: DictConfig) -> None:
    set_random_seed(cfg.seed)

    if cfg.use_wandb:
        wandb.init(
            project="liftr",
            name=cfg.experiment_name,
            config=dict(cfg),
        )

    use_cuda = True if torch.cuda.is_available() else False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = instantiate(cfg.dataset)
    model = instantiate(cfg.model)
    fsd_estimator = instantiate(cfg.fsd_estimator)
    true_fsd_estimator = GroundTruth("euc", "sum", stochastic=False)

    if use_cuda:
        model.cuda()
    
    if getattr(cfg.dataset, "_target_", None) == "src.datasets.arithmetic.Arithmetic":
        optimizer = OPTIMIZERS[cfg.optimizer](
            model.parameters(), 
            cfg.learning_rate, 
            weight_decay=0.1, betas=(0.9, 0.98)
        )
        for param in model.embedding.parameters():
            param.requires_grad = False
    else:
        optimizer = OPTIMIZERS[cfg.optimizer](
            model.parameters(),
            cfg.learning_rate
        )

    num_tasks = dataset.num_tasks
    trainloaders = []
    testloaders, test_metric_list, task_infos = [], [], []

    fsd_estimates, true_fsds = [], []
    grad_cos_sims = []

    for tid in range(num_tasks):
        itrain, itest, task_info = dataset.next_task()
        
        itrainloader = DataLoader(
            dataset=itrain,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=2)
        itestloader = DataLoader(
            dataset=itest,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=2)

        trainloaders.append(itrainloader)
        testloaders.append(itestloader)
        task_infos.append(task_info)

        fsd_estimator.on_task_start(model, tid)
        true_fsd_estimator.on_task_start(model, tid)

        task_info_cur = task_infos[tid]
        for epoch in range(cfg.train_epochs):

            model.train()
            total_loss, count = 0, 0
            total_fsd = 0
            for inputs, labels in itrainloader:
                if use_cuda:
                    inputs, labels = inputs.cuda(), labels.cuda()
                
                optimizer.zero_grad()
                logits = model.forward(inputs)
                loss = dataset.loss_fn(logits, labels, task_info=task_info_cur)
                total_loss += loss.item()
                count += 1

                loss.backward()
                optimizer.step()

            if tid > 0:
                fsd = torch.tensor(0.0, device=inputs.device)
                for prev_tid in range(tid):
                    fsd = fsd + fsd_estimator.get_fsd(
                        prev_tid, 
                        model, 
                        device, 
                    )
                fsd = fsd / tid

                true_fsd = torch.tensor(0.0, device=inputs.device)
                for prev_tid in range(tid):
                    true_fsd = true_fsd + true_fsd_estimator.get_fsd(
                        prev_tid, 
                        model, 
                        device, 
                    )
                true_fsd = true_fsd / tid

                grad_cos_sims.append(grad_cosine_sim(true_fsd, fsd, model))
                fsd_estimates.append(fsd.item())
                true_fsds.append(true_fsd.item())
                    
                if cfg.use_wandb:
                    avg_train_loss = total_loss / count if count > 0 else 0
                    wandb.log({
                        "train_loss": avg_train_loss,
                        "fsd_estimate": fsd.item(),
                        "true_fsd": true_fsd.item(),
                    }, step=(tid-1) * cfg.train_epochs + epoch)   

        fsd_estimator.on_task_end(tid, model, task_info, train_loader=itrainloader, device=device, loss_fn=dataset.loss_fn)
        true_fsd_estimator.on_task_end(tid, model, task_info, train_loader=itrainloader, device=device, loss_fn=dataset.loss_fn)

    fsd_estimates_np = np.array(fsd_estimates)
    true_fsds_np = np.array(true_fsds)
    spearman_corr, spearman_p = spearmanr(fsd_estimates_np, true_fsds_np)
    pearson_corr, pearson_p = pearsonr(fsd_estimates_np, true_fsds_np)
    avg_grad_cos_sim = sum(grad_cos_sims) / len(grad_cos_sims)
    print(f"\n{'─' * 46}")
    print(f"  Pearson correlation:    {pearson_corr:.4f}  (p={pearson_p:.2e})")
    print(f"  Spearman correlation:   {spearman_corr:.4f}  (p={spearman_p:.2e})")
    print(f"  Avg grad cosine sim:    {avg_grad_cos_sim:.4f}")
    print(f"{'─' * 46}\n")

    if getattr(cfg, 'output_csv', None):
        output_dir = os.path.dirname(cfg.output_csv) if os.path.dirname(cfg.output_csv) else '.'
        os.makedirs(output_dir, exist_ok=True)

        file_exists = os.path.exists(cfg.output_csv) and os.path.getsize(cfg.output_csv) > 0
        with open(cfg.output_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['experiment_name', 'epoch', 'fsd_estimate', 'true_fsd', 'grad_cos_sim',
                                 'pearson_corr', 'pearson_p', 'spearman_corr', 'spearman_p', 'avg_grad_cos_sim'])
            for epoch_idx, (fsd_est, true_fsd_val, grad_sim) in enumerate(zip(fsd_estimates, true_fsds, grad_cos_sims), start=1):
                writer.writerow([cfg.experiment_name, epoch_idx, fsd_est, true_fsd_val, grad_sim,
                                 pearson_corr, pearson_p, spearman_corr, spearman_p, avg_grad_cos_sim])

    if cfg.use_wandb:
        wandb.finish()

    return fsd_estimates, true_fsds


if __name__ == "__main__":
    main()