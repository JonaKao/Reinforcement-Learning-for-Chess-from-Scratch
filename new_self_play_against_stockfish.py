import numpy as np
import torch
import collections
import chess
# Monkey-patch to disable Categorical validation for maskable distribution
import torch.distributions.categorical as _cat
_original_cat_init = _cat.Categorical.__init__
def _patched_cat_init(self, probs=None, logits=None, validate_args=None):
    return _original_cat_init(self, probs=probs, logits=logits, validate_args=False)
_cat.Categorical.__init__ = _patched_cat_init

from sb3_contrib.common.wrappers.action_masker import ActionMasker
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from callback_logging import CSVLoggerCallback
import os
from Chess_Environment_Stockfish_Opponent import ChessEnv
from action_mapping import move_to_index

# ======= Curriculum Callback =======
class CurriculumCallback(BaseCallback):
    def __init__(self, env_ref, eval_games=200, win_upper=0.65, win_lower=0.35, increment=0.02):
        super().__init__()
        self.env_ref = env_ref  # Reference to base_env (not the wrapper)
        self.eval_games = eval_games
        self.win_upper = win_upper
        self.win_lower = win_lower
        self.increment = increment
        self.results = collections.deque(maxlen=eval_games)
        self.last_adjustment = 0

    def _on_step(self) -> bool:
        info = self.locals.get("infos", [{}])[0]
        # Detect end of game and record result
        if "final_observation" in info:  # done=True
            outcome = self.env_ref.board.outcome()
            if outcome is not None:
                if outcome.winner is True:   # Agent played White and won
                    self.results.append(1.0)
                elif outcome.winner is False: # Agent lost
                    self.results.append(0.0)
                else:
                    self.results.append(0.5)  # Draw

            # Check if we should adjust difficulty
            if len(self.results) == self.eval_games:
                win_rate = np.mean(self.results)
                if win_rate > self.win_upper:
                    self.env_ref.think_time += self.increment
                    print(f"[Curriculum] Win rate {win_rate:.2f} > {self.win_upper:.2f}, increasing think_time to {self.env_ref.think_time:.3f}s")
                    self.results.clear()
                elif win_rate < self.win_lower and self.env_ref.think_time > self.increment:
                    self.env_ref.think_time = max(self.increment, self.env_ref.think_time - self.increment)
                    print(f"[Curriculum] Win rate {win_rate:.2f} < {self.win_lower:.2f}, decreasing think_time to {self.env_ref.think_time:.3f}s")
                    self.results.clear()
        return True

# Build mask_fn for legal moves
def mask_fn(env) -> np.ndarray:
    mask = np.zeros(env.action_space.n, dtype=bool)
    for mv in env.board.legal_moves:
        mask[move_to_index[mv.uci()]] = True
    return mask

def main():
    os.makedirs("models", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # === Start very weak Stockfish ===
    base_env = ChessEnv(
        engine_path=r"C:\Tools\Stockfish\stockfish.exe",
        target_elo=1320,           # minimum supported by your engine
        think_time=0.02,           # very weak to start
        nodes=None,                # not limiting by nodes yet
        randomize_start_color=True,
    )
    env = ActionMasker(base_env, mask_fn)

    print(f"[train] Using engine path: {base_env.engine_path}")
    print(f"[train] Engine active? {base_env.engine is not None}")

    checkpoint_path = "models/chess_ppo_33200000_steps.zip"
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        model = MaskablePPO.load(
            checkpoint_path,
            env=env,
            device=device,
        )
        print(f"Checkpoint loaded! Current timesteps: {model.num_timesteps}")
    else:
        print(f"Checkpoint not found at {checkpoint_path}! Starting fresh...")
        model = MaskablePPO(
            MaskableActorCriticPolicy,
            env,
            ent_coef=0.02,
            verbose=1,
            tensorboard_log="runs/chess_gpu",
            device=device,
        )
        print("New model created! Current timesteps:", model.num_timesteps)

    # Callbacks for logging & checkpoints
    csv_cb  = CSVLoggerCallback()
    ckpt_cb = CheckpointCallback(
        save_freq=1_000_000,
        save_path="models/",
        name_prefix="chess_ppo"
    )
    curriculum_cb = CurriculumCallback(
        env_ref=base_env,
        eval_games=200,
        win_upper=0.65,
        win_lower=0.35,
        increment=0.02
    )

    print("Beginning/continuing training from", model.num_timesteps, "steps")
    model.learn(
        total_timesteps=model.num_timesteps + 3_000_000,
        reset_num_timesteps=False,
        callback=[csv_cb, ckpt_cb, curriculum_cb]
    )

    model.save(f"models/chess_ppo_final_{model.num_timesteps}_steps")
    print(f"Training complete! Final model saved as models/chess_ppo_final_{model.num_timesteps}_steps.zip")

if __name__ == "__main__":
    main()
