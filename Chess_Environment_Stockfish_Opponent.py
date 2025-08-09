# Chess_Environment.py
import chess
import chess.engine
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from action_mapping import index_to_move, move_to_index  # <- added move_to_index
import random
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
        engine_path: str | None = None,
        target_elo: int = 1000,
        think_time: float | None = 0.05,   # seconds per engine move (tiny!)
        nodes: int | None = None,          # alternative throttle; leave None to use time
        random_opening_moves: tuple[int, int] = (8, 10),
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
        self.random_opening_moves = random_opening_moves
        self.randomize_start_color = randomize_start_color
        self.target_elo = target_elo

        if self.engine_path is not None:
            # Allow 'stockfish' if it’s on PATH
            resolved = shutil.which(self.engine_path) or self.engine_path
            try:
                self.engine = chess.engine.SimpleEngine.popen_uci(resolved)
                # Use LimitStrength + Elo (Stockfish often clamps min Elo ~1320)
                self.engine.configure({"UCI_LimitStrength": True})
                # Clamp to plausible range to avoid engine error
                min_elo, max_elo = 1200, 3600  # conservative bounds; engine will clamp further if needed
                elo = int(max(min(self.target_elo, max_elo), min_elo))
                self.engine.configure({"UCI_Elo": elo})

                # Also scale legacy "Skill Level" (0..20) to approximate ~1000 strength
                # Rough map: 0~900, 1~1000, 2~1100, ...
                skill = max(0, min(20, int((self.target_elo - 900) / 50)))
                self.engine.configure({"Skill Level": skill})
            except Exception as e:
                print(f"[ChessEnv] Failed to start engine at '{resolved}': {e}. Running without opponent.")
                self.engine = None

    @property
    def action_mask(self):
        mask = np.zeros(self.action_space.n, dtype=bool)
        for mv in self.board.legal_moves:
            mask[ move_to_index.get(mv.uci(), 0) ] = True
        return mask

    def _apply_random_opening(self):
        lo, hi = self.random_opening_moves
        num_opening_moves = random.randint(lo, hi)
        for _ in range(num_opening_moves):
            legal_moves = list(self.board.legal_moves)
            if not legal_moves or self.board.is_game_over():
                break
            self.board.push(random.choice(legal_moves))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.board.reset()

        if self.randomize_start_color:
            self.board.turn = chess.BLACK if np.random.rand() < 0.5 else chess.WHITE
            self.starting_color = "black" if self.board.turn == chess.BLACK else "white"
        else:
            self.starting_color = "white"

        self.done = False
        self._apply_random_opening()

        # If engine exists and it’s their turn, let it move once so agent is to move
        if self.engine is not None and not self.done and self.board.is_game_over() is False:
            if not self._agent_to_move():
                self._engine_reply(apply_reward=False)

        print(f"[ChessEnv.reset()] Starting color: {self.starting_color}")
        return self._get_obs(), {}

    def _agent_to_move(self):
        # In this setup, "agent" always moves when env.step is called.
        # That means: after reset/engine reply we ensure board.turn == agent.
        return True  # conceptual; we always sync after reset/engine reply

    def _get_obs(self):
        planes = np.zeros((8, 8, 12), dtype=np.int8)
        piece_plane = {
            chess.PAWN:   0,
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK:   3,
            chess.QUEEN:  4,
            chess.KING:   5,
        }
        for sq in chess.SQUARES:
            p = self.board.piece_at(sq)
            if p:
                plane = piece_plane[p.piece_type] + (6 if p.color == chess.BLACK else 0)
                r, c = divmod(sq, 8)
                planes[r, c, plane] = 1
        return planes

    def _terminal_reward(self):
        # Positive if agent just delivered mate, negative if agent just got mated,
        # -1 on draws to discourage them.
        if self.board.is_checkmate():
            # Winner is side that just moved; if that wasn’t the agent, this is negative
            winner_is_white = not self.board.turn  # after checkmate, turn is side that CANNOT move
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
        limit = None
        if self.nodes is not None:
            limit = chess.engine.Limit(nodes=self.nodes)
        elif self.think_time is not None:
            limit = chess.engine.Limit(time=self.think_time)
        else:
            limit = chess.engine.Limit(time=0.05)

        try:
            result = self.engine.play(self.board, limit)
            eng_move = result.move
        except Exception as e:
            print(f"[ChessEnv] Engine failed to move: {e}")
            self.done = True
            return -1.0

        # Opponent capture -> negative shaping
        opp_captured = pre_board.piece_at(eng_move.to_square)
        self.board.push(eng_move)

        reward = 0.0
        if opp_captured:
            reward -= PIECE_VALUES.get(opp_captured.piece_type, 0.0)

        # Terminal after engine move?
        if self.board.is_game_over():
            self.done = True
            # after engine move, if mate: agent lost -> negative
            reward += self._terminal_reward()
        return reward

    def step(self, action):
        # unwrap numpy arrays
        if isinstance(action, np.ndarray):
            action = int(action)

        if action not in index_to_move:
            raise ValueError(f"Action {action} invalid; must be in 0..{len(index_to_move)-1}")

        uci = index_to_move[action]

        try:
            move = chess.Move.from_uci(uci)
        except chess.InvalidMoveError:
            self.done = True
            return self._get_obs(), -1.0, True, False, {}

        # illegal-move penalty
        if move not in self.board.legal_moves:
            self.done = True
            return self._get_obs(), -1.0, True, False, {}

        # Track agent color for terminal reward sign
        self._agent_was_white_last_move = self.board.turn == chess.WHITE

        # Pre-move capture check
        captured_piece = self.board.piece_at(move.to_square)

        # Agent move
        self.board.push(move)

        # Base move penalty to encourage faster play
        reward = -0.002

        # Positive capture shaping
        if captured_piece:
            reward += PIECE_VALUES.get(captured_piece.piece_type, 0.0)

        # Terminal after agent move?
        if self.board.is_game_over():
            self.done = True
            reward += self._terminal_reward()
            return self._get_obs(), reward, True, False, {}

        # Engine reply (if configured)
        reward += self._engine_reply(apply_reward=True)

        # If game ended after engine reply, we’re done
        if self.done:
            return self._get_obs(), reward, True, False, {}

        # Otherwise, it’s agent’s turn again
        return self._get_obs(), reward, False, False, {}

    def render(self, mode='human'):
        print(self.board)

    def close(self):
        if self.engine is not None:
            try:
                self.engine.quit()
            except Exception:
                pass

    def legal_actions(self):
        return list(self.board.legal_moves)
