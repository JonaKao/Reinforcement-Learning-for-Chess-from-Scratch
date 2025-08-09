# Chess_Environment_Stockfish_Opponent.py
import chess
import chess.engine
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from action_mapping import index_to_move, move_to_index
import shutil
import random

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
        engine_path: str | None = r"C:\Tools\Stockfish\stockfish.exe",
        target_elo: int = 1000,
        think_time: float | None = 0.05,
        nodes: int | None = None,             # legacy throttle (kept for compatibility)
        randomize_start_color: bool = True,
        # NEW: handicaps to make Stockfish very weak
        blunder_chance: float = 0.35,         # 35% of replies are random legal moves
        engine_nodes: int | None = 12,        # tiny node cap for engine replies
    ):
        super().__init__()
        self.board = chess.Board()
        self.action_space = spaces.Discrete(len(index_to_move))
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(8, 8, 12), dtype=np.int8
        )
        self.done = False

        # Engine config
        self.engine = None
        self.engine_path = engine_path
        self.think_time = think_time
        self.nodes = nodes
        self.randomize_start_color = randomize_start_color
        self.target_elo = target_elo

        # NEW: handicap controls
        self.blunder_chance = float(np.clip(blunder_chance, 0.0, 1.0))
        self.engine_nodes = engine_nodes

        if self.engine_path is not None:
            resolved = shutil.which(self.engine_path) or self.engine_path
            try:
                self.engine = chess.engine.SimpleEngine.popen_uci(resolved)
                print(f"[ChessEnv] Stockfish started successfully at '{resolved}'")
                try:
                    info = self.engine.protocol.uci()
                    print(f"[ChessEnv] Engine ID: {info.id}")
                except Exception:
                    pass

                # Engine weakening baseline (Elo floor enforced by engine)
                self.engine.configure({"UCI_LimitStrength": True})
                min_elo, max_elo = 1320, 3600
                elo = int(max(min(self.target_elo, max_elo), min_elo))
                self.engine.configure({"UCI_Elo": elo})

                # Force lowest skill level
                self.engine.configure({"Skill Level": 0})
            except Exception as e:
                print(
                    f"[ChessEnv] ERROR: Failed to start engine at '{resolved}': {e}. Running without opponent."
                )
                self.engine = None

    @property
    def action_mask(self):
        mask = np.zeros(self.action_space.n, dtype=bool)
        for mv in self.board.legal_moves:
            mask[move_to_index.get(mv.uci(), 0)] = True
        return mask

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.board.reset()

        # No random openings; optionally randomize which side the agent plays
        if self.randomize_start_color:
            if np.random.rand() < 0.5:
                self.board.turn = chess.BLACK
                self.starting_color = "black"
            else:
                self.board.turn = chess.WHITE
                self.starting_color = "white"
        else:
            self.starting_color = "white"

        self.done = False

        # If engine exists and it's their turn, let it move once so agent starts
        if self.engine is not None and not self.board.is_game_over():
            agent_is_white = self.starting_color == "white"
            if (agent_is_white and self.board.turn == chess.BLACK) or (
                not agent_is_white and self.board.turn == chess.WHITE
            ):
                self._engine_reply(apply_reward=False)

        print(f"[ChessEnv.reset()] Starting color: {self.starting_color}")
        return self._get_obs(), {}

    def _get_obs(self):
        planes = np.zeros((8, 8, 12), dtype=np.int8)
        piece_plane = {
            chess.PAWN: 0,
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK: 3,
            chess.QUEEN: 4,
            chess.KING: 5,
        }
        for sq in chess.SQUARES:
            p = self.board.piece_at(sq)
            if p:
                plane = piece_plane[p.piece_type] + (6 if p.color == chess.BLACK else 0)
                r, c = divmod(sq, 8)
                planes[r, c, plane] = 1
        return planes

    def _terminal_reward(self):
        if self.board.is_checkmate():
            winner_is_white = not self.board.turn
            return 2.5 if winner_is_white == self._agent_was_white_last_move else -2.5
        if (
            self.board.is_stalemate()
            or self.board.is_insufficient_material()
            or self.board.can_claim_fifty_moves()
            or self.board.can_claim_threefold_repetition()
        ):
            return -1.0
        return 0.0

    def _engine_reply(self, apply_reward: bool = True):
        """Engine moves once. Returns incremental reward from opponent’s capture or terminal."""
        if self.engine is None or self.done or self.board.is_game_over():
            return 0.0

        pre_board = self.board.copy()

        # --- Handicap 1: random blunders ---
        if random.random() < self.blunder_chance:
            legal = list(self.board.legal_moves)
            if not legal:
                return 0.0
            eng_move = random.choice(legal)
        else:
            # --- Handicap 2: starve search with tiny node cap ---
            if self.engine_nodes is not None:
                limit = chess.engine.Limit(nodes=max(1, int(self.engine_nodes)))
            elif self.nodes is not None:
                limit = chess.engine.Limit(nodes=self.nodes)
            elif self.think_time is not None:
                limit = chess.engine.Limit(time=self.think_time)
            else:
                limit = chess.engine.Limit(nodes=8)

            try:
                result = self.engine.play(self.board, limit)
                eng_move = result.move
            except Exception as e:
                print(f"[ChessEnv] Engine failed to move: {e}")
                self.done = True
                return -1.0

        opp_captured = pre_board.piece_at(eng_move.to_square)
        self.board.push(eng_move)

        reward = 0.0
        if opp_captured:
            reward -= PIECE_VALUES.get(opp_captured.piece_type, 0.0)

        if self.board.is_game_over():
            self.done = True
            reward += self._terminal_reward()
        return reward

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = int(action)

        if action not in index_to_move:
            raise ValueError(
                f"Action {action} invalid; must be in 0..{len(index_to_move)-1}"
            )

        uci = index_to_move[action]
        try:
            move = chess.Move.from_uci(uci)
        except chess.InvalidMoveError:
            self.done = True
            return self._get_obs(), -1.0, True, False, {}

        if move not in self.board.legal_moves:
            self.done = True
            return self._get_obs(), -1.0, True, False, {}

        self._agent_was_white_last_move = self.board.turn == chess.WHITE

        captured_piece = self.board.piece_at(move.to_square)
        self.board.push(move)

        reward = -0.002
        if captured_piece:
            reward += PIECE_VALUES.get(captured_piece.piece_type, 0.0)

        if self.board.is_game_over():
            self.done = True
            reward += self._terminal_reward()
            return self._get_obs(), reward, True, False, {}

        reward += self._engine_reply(apply_reward=True)

        if self.done:
            return self._get_obs(), reward, True, False, {}

        return self._get_obs(), reward, False, False, {}

    def render(self, mode="human"):
        print(self.board)

    def close(self):
        if self.engine is not None:
            try:
                self.engine.quit()
            except Exception:
                pass

    def legal_actions(self):
        return list(self.board.legal_moves)
