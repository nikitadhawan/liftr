import hydra 
from omegaconf import DictConfig
from hydra.utils import instantiate
import wandb
from tqdm import trange

import numpy as np
import torch
from torch.utils.data import DataLoader
import csv
import os

from src.utils import OPTIMIZERS, set_random_seed


def tasks_fsd(acc_list_of_lists):
    """Restructure per-task accuracy records and compute per-task means."""
    num_tasks = len(acc_list_of_lists)
    all_tasks = []
    for task_id in range(num_tasks):
        all_tasks.append(
            [after[task_id] for after in acc_list_of_lists[task_id:]])
    means = [np.mean(after) for after in acc_list_of_lists]
    return all_tasks, means

def tasks_bwt(task_list):
    """Compute per-task and average backward transfer."""
    bwt_list = []
    for i in range(len(task_list)):
        bwt = 100 * (task_list[i][-1] - task_list[i][0])
        bwt_list.append(bwt)
    avg_bwt = sum(bwt_list) / (len(task_list) - 1)
    return bwt_list, avg_bwt


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

    if use_cuda:
        model.cuda()
    
    if getattr(cfg.dataset, "_target_", None) == "src.datasets.arithmetic.Arithmetic":
        optimizer = OPTIMIZERS[cfg.optimizer](
            model.parameters(), 
            cfg.learning_rate, 
            weight_decay=cfg.weight_decay, betas=(0.9, 0.98)
        )
    else:
        optimizer = OPTIMIZERS[cfg.optimizer](
            model.parameters(),
            cfg.learning_rate
        )

    for module in model.modules():
        if isinstance(module, torch.nn.Embedding):
            for param in module.parameters():
                param.requires_grad = False

    num_tasks = dataset.num_tasks
    trainloaders = []
    testloaders, test_metric_list, task_infos = [], [], []

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

        task_info_cur = task_infos[tid]
        for epoch in trange(cfg.train_epochs, desc=f"Task {tid + 1}/{num_tasks}"):

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

                if tid > 0 and cfg.fsd_weight > 0:
                    fsd = torch.tensor(0.0, device=inputs.device)
                    for prev_tid in range(tid):
                        fsd = fsd + fsd_estimator.get_fsd(
                            prev_tid, 
                            model, 
                            device, 
                        )
                    fsd = fsd / tid
                    total_fsd = total_fsd + fsd.item()
                    loss = loss + cfg.fsd_weight * fsd
                loss.backward()
                optimizer.step()

            
            if epoch % 10 == 0:
                val_loss, val_count = 0, 0
                val_output, val_target = [], []
                model.eval()
                with torch.no_grad():
                    for inputs, labels in testloaders[0]:
                        if use_cuda:
                            inputs, labels = inputs.cuda(), labels.cuda()
                        logits = model.forward(inputs)
                        val_output.append(logits.detach().cpu())
                        val_target.append(labels.detach().cpu())
                        loss = dataset.loss_fn(logits, labels, task_info=task_info_cur)
                        val_loss += loss.item()
                        val_count += 1
                    val_output = torch.cat(val_output, dim=0)
                    val_target = torch.cat(val_target, dim=0)
                    val_acc = dataset.evaluate(val_output, val_target, task_info=task_info_cur)
                    
            if cfg.use_wandb:
                avg_train_loss = total_loss / count if count > 0 else 0
                avg_fsd = total_fsd / count if count > 0 else 0
                if epoch % 10 == 0:
                    avg_val_loss = val_loss / val_count if val_count > 0 else 0
                    wandb.log({
                        "train_loss": avg_train_loss,
                        "fsd": avg_fsd,
                        f"task0_val_loss": avg_val_loss,
                        f"task0_val_acc": val_acc,
                    }, step=tid * cfg.train_epochs + epoch)
                else:
                    wandb.log({
                        "train_loss": avg_train_loss,
                        "fsd": avg_fsd,
                    }, step=tid * cfg.train_epochs + epoch)    

        fsd_estimator.on_task_end(tid, model, task_info, train_loader=itrainloader, device=device, loss_fn=dataset.loss_fn)

        model.eval()
        test_metric = []
        for test_tid, testdata in enumerate(testloaders):
            all_outputs = []
            all_targets = []
            for inputs, labels in testdata:
                if use_cuda:
                    inputs, labels = inputs.cuda(), labels.cuda()
                task_info_t = task_infos[test_tid]
                outputs = model.forward(inputs)
                all_outputs.append(outputs.detach().cpu())
                all_targets.append(labels.detach().cpu())
            all_outputs = torch.cat(all_outputs, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            metric = dataset.evaluate(all_outputs, all_targets, task_info=task_info_t)
            test_metric.append(metric)

        test_metric_list.append(test_metric)
        mean_acc = sum(test_metric) / len(test_metric)
        print(f"\n── After task {tid + 1}/{num_tasks} " + "─" * 30)
        for t, acc in enumerate(test_metric):
            print(f"  Task {t}: {acc:.4f}")
        print(f"  Mean:   {mean_acc:.4f}")

        if cfg.use_wandb:
            wandb.log({
                f"avg_performance": np.mean(test_metric)
            }, step=(1 + tid) * cfg.train_epochs)
            if tid > 0:
                current_tasks = []
                for task_id in range(tid + 1):
                    current_tasks.append(
                        [after[task_id] for after in test_metric_list[task_id:]]
                    )
                _, current_avg_bwt = tasks_bwt(current_tasks)
                wandb.log({
                    f"avg_bwt": current_avg_bwt
                }, step=(1 + tid) * cfg.train_epochs)


    tasks, means = tasks_fsd(test_metric_list)
    bwt_list, avg_bwt = tasks_bwt(tasks)
    print(f"\n{'─' * 46}")
    print(f"  Final mean accuracy: {means[-1]:.4f}")
    print(f"  Average BWT:         {avg_bwt:.4f}")
    print(f"{'─' * 46}\n")

    if cfg.use_wandb:
        wandb.log({
            f"final_avg_performance": means[-1],
            f"final_avg_bwt": avg_bwt,
        })
      
        wandb.finish()

    if cfg.output_csv:
        output_dir = os.path.dirname(cfg.output_csv)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        file_exists = os.path.exists(cfg.output_csv) and os.path.getsize(cfg.output_csv) > 0
        with open(cfg.output_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['experiment_name', 'fsd_estimator', 'fsd_weight', 'sample_size', 'depth', 'avg_acc', 'avg_bwt', 'task0', 'task1', 'seed'])
            exp_name = getattr(cfg, 'experiment_name', 'unknown')
            estimator_name = cfg.fsd_estimator._target_.split('.')[-1].lower()
            sample_size = getattr(cfg.fsd_estimator, 'sample_size', 0)
            depth = getattr(cfg.model, 'num_blocks', 0)
            final_task_accs = test_metric_list[-1]
            task0_acc = final_task_accs[0] 
            task1_acc = final_task_accs[1] 
            writer.writerow([exp_name, estimator_name, cfg.fsd_weight, sample_size, depth, means[-1], avg_bwt, task0_acc, task1_acc, cfg.seed])

    return test_metric_list, means[-1], avg_bwt


if __name__ == "__main__":
    main()