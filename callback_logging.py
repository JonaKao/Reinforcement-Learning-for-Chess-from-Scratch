# callback_logging.py

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import collections
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


class TensorboardScalarsCallback(BaseCallback):
    def __init__(self, env_ref, eval_games=50, verbose=0):
        super().__init__(verbose)
        self.env_ref = env_ref
        self.results = collections.deque(maxlen=eval_games)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [{}])
        if infos and isinstance(infos, (list, tuple)):
            info = infos[0]
            if "final_observation" in info or "episode" in info:
                outcome = self.env_ref.board.outcome()
                if outcome is not None:
                    if outcome.winner is True:
                        self.results.append(1.0)
                    elif outcome.winner is False:
                        self.results.append(0.0)
                    else:
                        self.results.append(0.5)

                if len(self.results) > 0:
                    win_rate = float(np.mean(self.results))
                    self.logger.record("custom/win_rate", win_rate)

                # log current difficulty knobs
                self.logger.record("custom/blunder_chance", float(self.env_ref.blunder_chance))
                self.logger.record("custom/engine_nodes", float(self.env_ref.engine_nodes or 0))
                self.logger.record("custom/think_time", float(self.env_ref.think_time or 0.0))
        return True
