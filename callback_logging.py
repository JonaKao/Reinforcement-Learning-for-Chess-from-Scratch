import csv
import collections
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class CSVLoggerCallback(BaseCallback):
    def __init__(self, filename="models/metrics.csv", verbose=0):
        super().__init__(verbose)
        self.filename = filename
        with open(self.filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timesteps", "ep_len_mean", "ep_rew_mean"])

    def _on_step(self) -> bool:
        n_steps = getattr(self.model, "n_steps", 0)
        if n_steps > 0 and self.num_timesteps % n_steps == 0:
            data = self.logger.name_to_value
            with open(self.filename, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.num_timesteps,
                    data.get("rollout/ep_len_mean", ""),
                    data.get("rollout/ep_rew_mean", ""),
                ])
        return True


class TensorboardScalarsCallback(BaseCallback):
    """
    Logs rolling win_rate and difficulty knobs to TensorBoard.
    Assumes single-env DummyVecEnv and env exposes:
      .board, .blunder_chance, .engine_nodes, .think_time
    """
    def __init__(self, env_ref, eval_games: int = 50, verbose: int = 0):
        super().__init__(verbose)
        self.env_ref = env_ref
        self.window = collections.deque(maxlen=eval_games)  # rolling results

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [{}])

    # Always record current difficulty knobs each step so they show up immediately
        self.logger.record("custom/blunder_chance", float(self.env_ref.blunder_chance))
        self.logger.record("custom/engine_nodes", float(self.env_ref.engine_nodes or 0))
        self.logger.record("custom/think_time", float(self.env_ref.think_time or 0.0))

        if not infos or not isinstance(infos, (list, tuple)):
            return True
        info0 = infos[0] or {}

    # Episode ended?
        if "final_observation" in info0 or "episode" in info0:
        # Prefer a real chess outcome; if None (illegal move/trunc), count as loss
            outcome = self.env_ref.board.outcome(claim_draw=True)
            if outcome is None:
                result = 0.0
            else:
                result = 1.0 if outcome.winner is True else (0.5 if outcome.winner is None else 0.0)

            self.window.append(result)
            self.logger.record("custom/last_result", float(result))
            self.logger.record("custom/win_rate", float(np.mean(self.window)))

        return True