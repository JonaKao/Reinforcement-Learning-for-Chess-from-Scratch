# Chess_Environment_Stockfish_Opponent.py
import chess
import chess.engine
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from action_mapping import index_to_move, move_to_index
import shutil

# capture shaping to reduce draws / encourage tactics
PIECE_VALUES = {
    chess.PAWN:   0.01,
    chess.KNIGHT: 0.03,
    chess.BISHOP: 0.03,
    chess.ROOK:   0.05,
    chess.QUEEN:  0.09,
}

class ChessEnv(gym.Env):
    def __init__(
        self,
        # Default to your installed Stockfish path; can be overridden by caller
        engine_path: str | None = r"C:\Tools\Stockfish\stockfish.exe",
        target_elo: int = 1000,
        think_time: float | None = 0.05,   # seconds per engine move
        nodes: int | None = None,          # alternative throttle; leave None to use time
        randomize_start_color: bool = True,
    ):
        super().__init__()
        self.board = chess.Board()
        self.action_space = spaces.Discrete(len(index_to_move))
        self.observation_space = spaces.Box(low=0, high=1, shape=(8, 8, 12), dtype=np.int8)
        self.done = False

        # Engine config
        self.engine = None
        self.engine_path = engine_path
        self.think_time = think_time
        self.nodes = nodes
        self.randomize_start_color = randomize_start_color
        self.target_elo = target_elo

        if self.engine_path is not None:
            # Allow 'stockfish' if it’s on PATH; otherwise use the full path provided
            resolved = shutil.which(self.engine_path) or self.engine_path
            try:
                self.engine = chess.engine.SimpleEngine.popen_uci(resolved)
                print(f"[ChessEnv] Stockfish started successfully at '{resolved}'")
                # Optional: show engine ID info
                try:
                    info = self.engine.protocol.uci()
                    print(f"[ChessEnv] Engine ID: {info.id}")
                except Exception:
                    pass

                # Weaken engine
                self.engine.configure({"UCI_LimitStrength": True})
                # Clamp to a safe range (engine will clamp internally too)
                min_elo, max_elo = 1200, 3600
                elo = int(max(min(self.target_elo, max_elo), min_elo))
                self.engine.configure({"UCI_Elo": elo})

                # Legacy "Skill Level" (0..20) — rough map near ~1000 Elo
                skill = max(0, min(20, int((self.target_elo - 900) / 50)))
                self.engine.configure({"Skill Level": skill})
            except Exception as e:
                print(f"[ChessEnv] ERROR: Failed to start engine at '{resolved}': {e}. Running without opponent.")
                self.engine = None

    @property
    def action_mask(self):
