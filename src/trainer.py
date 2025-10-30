import os
import logging
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from src.generator import QAPGenerator
from src.utils.utils import AverageMeter, num_param, TimeEstimator
from src.utils.ops import qap_cost
from src.backend.post_process import local_search
from src.backend.sampling import sequential_sampling
from src.models.model import QAPNet


def setup_logger(log_path):
    """Setup logger with console and file handlers."""
    logger = logging.getLogger('PLMA')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_format)
    
    # File handler
    file_handler = logging.FileHandler(os.path.join(log_path, 'training_process.log'))
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter('%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_format)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

class Trainer:
    def __init__(self, args, generator_params, model_params, trainer_params, optimizer_params, plma_configs):
        self.args = args

        self.generator = QAPGenerator(**generator_params)

        self.model = QAPNet(**model_params)
        num_param(self.model)

        self.optimizer = Adam(self.model.parameters(), **optimizer_params["optimizer"])
        self.scheduler = MultiStepLR(self.optimizer, **optimizer_params["scheduler"])
        self.trainer_params = trainer_params

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.problem_size = generator_params["problem_size"]
        self.metric_keys = ["loss", "initial_cost", "improved_cost", "entropy"]
        self.metrics_history = {k: [] for k in self.metric_keys}
        self.time_estimator = TimeEstimator()
        self.log_path = args.log_path
        self.logger = setup_logger(self.log_path)
        self.plma_configs = plma_configs
        self.setup_mcpg_backend()

    def setup_mcpg_backend(self):
        assert self.device.type == "cuda", "MCPG backend requires CUDA."
        from src.backend.qap_backend import QAPBackendGPU
        self.backend = QAPBackendGPU(device=self.device)
        self.backend.setup_rand_states(self.trainer_params["batch_size"] * self.plma_configs["num_samples"] * self.plma_configs["num_actions"])

    def run(self):
        for epoch in range(1, self.trainer_params["epochs"] + 1):
            metrics = self.train_one_epoch()
            self.scheduler.step()

            # Store metrics
            for k, v in metrics.items():
                self.metrics_history[k].append(v)

            # Log status and save
            self._log_epoch_stats(epoch, metrics)
            if epoch % self.trainer_params["model_save_interval"] == 0:
                torch.save(self.model.state_dict(), os.path.join(self.log_path, f"epoch_{epoch}.pth"))

    def train_one_epoch(self):
        meters = {k: AverageMeter() for k in self.metric_keys}
        episode = 0
        train_num_episode = self.trainer_params["train_episodes"]

        while episode < train_num_episode:
            batch_size = self.trainer_params["batch_size"]

            batch = self.generator.generate(batch_size)
            batch_metrics = self.train_one_batch(batch)

            episode += batch_size
            for k, v in batch_metrics.items():
                meters[k].update(v, batch_size)

        return {k: meter.avg for k, meter in meters.items()}
    
    def train_one_batch(self, batch):
        D_batch, F_batch = batch[0], batch[1]
        heatmap = self.model(D_batch, F_batch)
        
        # Get start states, run MCMC and local search
        start_states, _, _ = sequential_sampling(heatmap.detach(), self.plma_configs["num_samples"], D_batch, F_batch) # [B, S, n]
        terminal_states = self.backend.mcmc_step(start_states, torch.exp(heatmap.detach()), self.plma_configs["chain_length"])
        initial_costs = qap_cost(D_batch, F_batch, start_states)
        improved_states, improved_costs = local_search(terminal_states, D_batch, F_batch, self.backend, self.plma_configs["local_search_iter"], self.plma_configs.get("num_actions"))
        
        # Compute Loss
        component_scores = torch.take_along_dim(heatmap.unsqueeze(1), terminal_states.to(torch.int64).unsqueeze(-1), dim=3).squeeze(-1) # [B, K*M, n]
        total_score = component_scores.sum(dim=2)  # [B, K*M]
        advantage = (improved_costs.detach() - improved_costs.mean(-1, keepdim=True).detach()) / (improved_costs.std(-1, keepdim=True).detach() + 1e-8) # [B, K*M]
        rl_loss = (advantage * total_score).mean()
        entropy = - torch.sum(heatmap * torch.exp(heatmap), dim=(1,2)).mean()     # Here heatmap is in the log space
        loss = rl_loss - entropy * self.plma_configs["entropy_weight"]

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"initial_cost": initial_costs.amin(-1).mean().item(),  "improved_cost": improved_costs.amin(-1).mean().item(), "loss": loss.item(), "entropy": entropy.item()}

    def _log_epoch_stats(self, epoch, metrics):
        """Log epoch statistics with labeled metrics."""
        elapsed, remain = self.time_estimator.get_est_string(epoch, self.trainer_params["epochs"])
        improvement = metrics['initial_cost'] - metrics['improved_cost']
        total_epochs = self.trainer_params["epochs"]

        log_message = (
            "-" * 50
            + "\n"
            + f"Epoch {epoch:3d}/{total_epochs} | Time Est.: Elapsed[{elapsed}], Remain[{remain}]\n"
            + f"Loss: {metrics['loss']:.3f} | Entropy: {metrics['entropy']:.3f}\n"
            + f"Initial cost: {metrics['initial_cost']:.3f} | Improved cost: {metrics['improved_cost']:.3f} | Improvement: {improvement:.3f}"
        )
        self.logger.info(log_message)
