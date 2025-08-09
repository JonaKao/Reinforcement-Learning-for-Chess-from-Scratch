import numpy as np
import torch
import collections
import chess
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

# ======= Curriculum Callback (blunder-first) =======
class CurriculumCallback(BaseCallback):
    def __init__(self, env_ref, eval_games=150, win_upper=0.65, win_lower=0.35):
        super().__init__()
        self.env_ref = env_ref
        self.eval_games = eval_games
        self.win_upper = win_upper
        self.win_lower = win_lower
        self.results = collections.deque(maxlen=eval_games)

    def _on_step(self) -> bool:
        info = self.locals.get("infos", [{}])[0]
        # end-of-episode marker
        if "final_observation" in info:
            outcome = self.env_ref.board.outcome()
            if outcome is not None:
                if outcome.winner is True:
                    self.results.append(1.0)
                elif outcome.winner is False:
                    self.results.append(0.0)
                else:
                    self.results.append(0.5)

            if len(self.results) == self.eval_games:
                win_rate = float(np.mean(self.results))
                bc = self.env_ref.blunder_chance
                nd = self.env_ref.engine_nodes if self.env_ref.engine_nodes is not None else 8
                tt = self.env_ref.think_time if self.env_ref.think_time is not None else 0.0

                if win_rate > self.win_upper:
                    # Make engine stronger: fewer blunders, more nodes; once no blunders, add time
                    if bc > 0.0:
                        new_bc = max(0.0, bc - 0.10)
                        self.env_ref.set_difficulty(blunder_chance=new_bc)
                        print(f"[Curriculum] Win {win_rate:.2f} > {self.win_upper:.2f}: blunder_chance {bc:.2f} -> {new_bc:.2f}")
                    else:
                        new_tt = min(0.10, tt + 0.02)
                        self.env_ref.set_difficulty(think_time=new_tt)
                        print(f"[Curriculum] Win {win_rate:.2f} > {self.win_upper:.2f}: think_time {tt:.2f}s -> {new_tt:.2f}s")

                    new_nd = min(250, nd + 10)
                    self.env_ref.set_difficulty(engine_nodes=new_nd)
                    print(f"[Curriculum]             engine_nodes {nd} -> {new_nd}")

                    self.results.clear()

                elif win_rate < self.win_lower:
                    # Make engine easier: more blunders, fewer nodes, less time
                    new_bc = min(0.90, bc + 0.10)
                    new_nd = max(1, nd - 5)
                    new_tt = max(0.0, tt - 0.01)
                    self.env_ref.set_difficulty(blunder_chance=new_bc, engine_nodes=new_nd, think_time=new_tt)
                    print(f"[Curriculum] Win {win_rate:.2f} < {self.win_lower:.2f}: "
                          f"blunder_chance {bc:.2f}->{new_bc:.2f}, engine_nodes {nd}->{new_nd}, think_time {tt:.2f}s->{new_tt:.2f}s")
                    self.results.clear()
        return True

def mask_fn(env) -> np.ndarray:
    mask = np.zeros(env.action_space.n, dtype=bool)
    for mv in env.board.legal_moves:
        mask[move_to_index[mv.uci()]] = True
    return mask

def main():
    os.makedirs("models", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Start comically weak: many blunders, barely searches
    base_env = ChessEnv(
        engine_path=r"C:\Tools\Stockfish\stockfish.exe",
        target_elo=1320,
        think_time=0.0,
        nodes=None,
        randomize_start_color=True,
        blunder_chance=0.90,
        engine_nodes=3,
    )
    env = ActionMasker(base_env, mask_fn)

    print(f"[train] Using engine path: {base_env.engine_path}")
    print(f"[train] Engine active? {base_env.engine is not None}")
    print(f"[train] Initial difficulty: blunder_chance={base_env.blunder_chance:.2f}, engine_nodes={base_env.engine_nodes}, think_time={base_env.think_time:.2f}s")

    checkpoint_path = "models/chess_ppo_33200000_steps.zip"
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        model = MaskablePPO.load(checkpoint_path, env=env, device=device)
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

    csv_cb  = CSVLoggerCallback()
    ckpt_cb = CheckpointCallback(save_freq=1_000_000, save_path="models/", name_prefix="chess_ppo")
    curriculum_cb = CurriculumCallback(env_ref=base_env, eval_games=150, win_upper=0.65, win_lower=0.35)

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
