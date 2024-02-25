#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import sys
sys.path.insert(0, '/home/sandor/robots_ws/legged_gym/legged_gym/scripts')
from train import train as training_function
import os, subprocess
from dataclasses import asdict
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
except ModuleNotFoundError:
    raise ModuleNotFoundError("Wandb is required to log to Weights and Biases.")


class WandbSummaryWriter(SummaryWriter):
    """Summary writer for Weights and Biases."""

    def __init__(self, log_dir: str, flush_secs: int, cfg):
        super().__init__(log_dir, flush_secs)

        try:
            project = "legged_project"
        except KeyError:
            raise KeyError("Please specify wandb_project in the runner config, e.g. legged_gym.")

        try:
            entity = os.environ["WANDB_USERNAME"]
        except KeyError:
            raise KeyError(
                "Wandb username not found. Please run or add to ~/.bashrc: export WANDB_USERNAME=YOUR_USERNAME"
            )

        wandb.init(project=project, entity=entity)

        # Fetch Git Branch Name
        branch_name = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode().strip()
        # Fetch Last Commit Message
        last_commit = subprocess.check_output(["git", "log", "-1", "--pretty=%B"]).decode().strip()

        # Format the time
        time_of_launch = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Mapping of Docker pod names (container IDs) to human-readable names
        pod_name_mappings = {
            "9a6d760de3db": "isaac",
            "90e979582683": "solod",
            "5698c0583017": "quattro",
            "4277d4343f1f": "debug"
        }
        
        # Attempt to get Docker pod name
        container_id = os.environ.get("HOSTNAME", "")
        pod_name = pod_name_mappings.get(container_id, "")  # Default to empty string if not found

        # Conditionally prepend Docker pod name
        formatted_pod_name = f"{pod_name}" if pod_name else ""
        
        # Use backslashes to escape quotes and format the name
        wandb.run.name = f'{formatted_pod_name}_{branch_name}_"{last_commit}"_{time_of_launch}'


        self.name_map = {
            "Train/mean_reward/time": "Train/mean_reward_time",
            "Train/mean_episode_length/time": "Train/mean_episode_length_time",
        }

        run_name = os.path.split(log_dir)[-1]

        wandb.log({"log_dir": run_name})

    # sweep_config = {
    #     'method': 'random', #grid, random
    #     "name": "sweep",
    #     'metric': {
    #     'name': 'Loss/value_function',
    #     'goal': 'minimize'   
    #     },
    #     'parameters': {
    #         'num_learning_epochs': {
    #             'values': [2, 5, 10]
    #         },
    #         'learning_rate': {
    #             'values': [1e-2, 1e-3, 1e-4, 3e-4, 3e-5, 1e-5]
    #         },
    #         'lam':{
    #             'values':[0.9, 0.925, 0.95, 0.975, 0.99]
    #         },
    #     'gamma':{
    #             'values':[0.975, 0.98, 0.985, 0.99, 0.995]
    #         },
    #         'num_mini_batches': {
    #             'values': [1, 2, 4, 8, 16]
    #         },
    #         'desired_kl': {
    #             'values': [0.01, 0.02, 0.03, 0.04, 0.05]
    #         },
    #         'optimizer': {
    #             'values': ['adam', 'sgd']
    #         },
    #     }
    # }

    # # # 3: Start the sweep
    # sweep_id = wandb.sweep(sweep=sweep_config, project="legged_project")
    
    # wandb.agent(sweep_id, function=training_function, count=100)

    def store_config(self, env_cfg, runner_cfg, alg_cfg, policy_cfg):
        wandb.config.update({"runner_cfg": runner_cfg})
        wandb.config.update({"policy_cfg": policy_cfg})
        wandb.config.update({"alg_cfg": alg_cfg})
        wandb.config.update({"env_cfg": env_cfg})

    def _map_path(self, path):
        if path in self.name_map:
            return self.name_map[path]
        else:
            return path

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None, new_style=False):
        super().add_scalar(
            tag,
            scalar_value,
            global_step=global_step,
            walltime=walltime,
            new_style=new_style,
        )
        wandb.log({self._map_path(tag): scalar_value}, step=global_step)

    def stop(self):
        wandb.finish()

    def log_config(self, env_cfg, runner_cfg, alg_cfg, policy_cfg):
        self.store_config(env_cfg, runner_cfg, alg_cfg, policy_cfg)

    def save_model(self, model_path, iter):
        wandb.save(model_path, base_path=os.path.dirname(model_path))

    def save_file(self, path, iter=None):
        wandb.save(path, base_path=os.path.dirname(path))
