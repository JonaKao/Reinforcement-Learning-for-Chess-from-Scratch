import csv
from stable_baselines3.common.callbacks import BaseCallback

class CSVLoggerCallback(BaseCallback):
    def __init__(self, filename="models/metrics.csv", verbose=0):
        super().__init__(verbose)
        self.filename = filename
        # write header once
        with open(self.filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timesteps", "ep_len_mean", "ep_rew_mean"])

    def _on_step(self) -> bool:
        # log once per rollout (every n_steps)
        # use model.n_steps to avoid zero or missing values
        n_steps = getattr(self.model, "n_steps", 0)
        if n_steps > 0 and self.num_timesteps % n_steps == 0:
            data = self.logger.name_to_value  # SB3’s internal stats
            with open(self.filename, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.num_timesteps,
                    data.get("rollout/ep_len_mean", ""),
                    data.get("rollout/ep_rew_mean", ""),
                ])
        return True
