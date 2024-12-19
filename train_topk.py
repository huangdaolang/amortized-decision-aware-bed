import torch
import torch.nn.functional as F
import wandb
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf
import logging
import os

from preprocess.HPOB import HPOB

from architectures.tndp_topk import TNDP_Topk
from utils import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="./configs", config_name="train_topk")
def train(cfg):
    if cfg.device == "cuda":
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    if cfg.wandb.use_wandb:
        run = wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.run_name,
            group=cfg.wandb.group,
            tags=cfg.wandb.tags,
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            dir=get_original_cwd(),  # don't nest wandb inside hydra dir
        )
        # Save hydra configs with wandb (handles hydra's multirun dir)
        try:
            hydra_log_dir = os.path.join(HydraConfig.get().runtime.output_dir, ".hydra")
            wandb.save(str(hydra_log_dir), policy="now")
        except FileExistsError:
            pass

    model = TNDP_Topk(
        dim_x=cfg.nn.dim_x,
        dim_y=cfg.nn.dim_y,
        dim_embedding=cfg.nn.dim_embedding,
        num_layers=cfg.nn.num_layers,
        embedding_depth=cfg.nn.embedding_depth,
        num_heads=cfg.nn.num_heads,
        dim_feedforward=cfg.nn.dim_feedforward,
        dropout=cfg.nn.dropout,
        horizon=cfg.nn.T + 1,
        embedding_way=cfg.nn.embedding_way
    )

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total number of parameters: {total_params}")
    logger.info(model)

    sampler = HPOB(meta_dataset=cfg.dataset.meta_dataset)

    T = cfg.dataset.T
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.steps)

    for step in range(cfg.steps):
        model.train()
        optimizer.zero_grad()
        batch = sampler.sample(
            batch_size=cfg.batch_size,
            n_context=cfg.dataset.n_context,
            n_query=cfg.dataset.n_query,
            n_target=cfg.dataset.n_target,
            min_n_context=cfg.dataset.min_n_initial_context,
            max_n_context=cfg.dataset.max_n_initial_context,
            )

        log_dict = {}
        query_log_probs = []
        rewards = []
        target_losses = 0
        topk_list = []

        for t in range(T):
            if t == 0:
                y_obs = batch.context_y.squeeze(-1)

                topk_values, topk_indices = torch.topk(y_obs, cfg.dataset.k, dim=1)
                values_sum = topk_values.sum(dim=1)

                topk_list.append(values_sum.detach())

            batch.t = torch.tensor(t, dtype=torch.long)
            outs = model(batch)

            query_out = outs.query_out
            decision_out = outs.decision_out

            # query part
            query_probs = F.softmax(query_out.squeeze(-1), dim=1)
            query_distribution = torch.distributions.Categorical(query_probs)
            next_query_idx = query_distribution.sample()  # [batch_size]

            # add new query to context data
            next_query_x = torch.gather(batch.query_x, 1, next_query_idx.reshape(-1, 1, 1).expand(-1, -1, batch.query_x.shape[-1]))  # [batch_size, 1, dim_x]
            next_query_y = torch.gather(batch.query_y, 1, next_query_idx.reshape(-1, 1, 1))  # [batch_size, 1, 1]
            batch.context_x = torch.cat([batch.context_x, next_query_x], dim=1)
            batch.context_y = torch.cat([batch.context_y, next_query_y], dim=1)

            # delete selected query data
            all_indices = torch.arange(batch.query_x.size(1)).unsqueeze(0).expand(batch.query_x.size(0), -1)  # [batch_size, n_query]
            mask = all_indices != next_query_idx.unsqueeze(1)  # [batch_size, n_query]
            batch.query_x = batch.query_x[mask].reshape(batch.query_x.size(0), -1, batch.query_x.size(-1))
            batch.query_y = batch.query_y[mask].reshape(batch.query_y.size(0), -1, batch.query_y.size(-1))

            # add query_idx log prob to list
            query_log_probs.append(query_distribution.log_prob(next_query_idx))

            y_obs = batch.context_y.squeeze(-1)

            topk_values, topk_indices = torch.topk(y_obs, cfg.dataset.k, dim=1)

            values_sum = topk_values.sum(dim=1)

            topk_list.append(values_sum.detach())

            reward = topk_list[-1] - topk_list[-2]
            reward = torch.clamp(reward, min=0)
            rewards.append(reward)

            # regression part
            mean, std = torch.chunk(decision_out, 2, dim=-1)
            std = torch.exp(std)
            pred_tar = torch.distributions.Normal(mean, std)

            target_log_prob = pred_tar.log_prob(batch.target_y).flatten()  # [batch_size * n_target]
            target_loss = - target_log_prob.mean()  # [1]
            target_losses += target_loss

        if step % cfg.print_freq == 0:
            print(f"target prediction: {mean.flatten()[:5].cpu().detach().numpy()}")
            print(f"target ground truth: {batch.target_y.flatten()[:5].cpu().detach().numpy()}")

        query_log_probs = torch.stack(query_log_probs, dim=0)
        rewards = torch.stack(rewards, dim=0)
        true_utility = torch.stack(topk_list, dim=0)

        R = torch.zeros(rewards.size(1))
        returns = []

        for t in range(T - 1, -1, -1):
            R = rewards[t] + cfg.gamma * R
            returns.insert(0, R.clone())

        returns = torch.stack(returns)
        policy_loss_unnorm = -torch.sum(query_log_probs * returns) / T  # used to monitor the policy loss

        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        # calculate REINFORCE loss
        policy_loss = -torch.sum(query_log_probs * returns) / T

        loss = policy_loss + target_losses / T

        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % cfg.print_freq == 0:
            logger.info(f"Step {step} loss: {loss} one reward trajectory: {rewards[:, 0].cpu().detach().numpy()} "
                        f"true utility: {true_utility[:, 0].cpu().detach().numpy()}")

        log_dict["target_loss"] = target_losses.item()
        log_dict["policy_loss"] = policy_loss_unnorm.item()
        log_dict["train_loss"] = loss.item()
        log_dict["train_step"] = step

        if cfg.wandb.use_wandb:
            wandb.log(log_dict)

        if cfg.checkpoint and step % cfg.checkpoint_interval == 0:
            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": step + 1,
            }

            output_directory = HydraConfig.get().runtime.output_dir
            print(f"{output_directory=}")
            checkpoint_path = os.path.join(output_directory, "ckpt.tar")

            torch.save(ckpt, checkpoint_path)


if __name__ == "__main__":
    train()

