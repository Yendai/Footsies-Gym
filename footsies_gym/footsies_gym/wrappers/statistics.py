import gymnasium as gym
from ..moves import FootsiesMove


MOVE_ID_TO_MOVE = {move.value.id: move for move in FootsiesMove}


class FootsiesStatistics(gym.Wrapper):
    """Collect statistics on the FOOTSIES environment. The environment that this wrapper receives should not be wrapped by observation wrappers"""

    def __init__(self, env):
        super().__init__(env)
        self._special_moves_per_episode = []
        self._special_moves_from_neutral_per_episode = []
        self._combo_stance_steps_per_episode = []
        self._combo_armed_steps_per_episode = []
        self._combo_toggle_count_per_episode = []
        self._combo_followups_queued_per_episode = []
        self._wins_per_episode = []

        self._special_moves_per_episode_counter = 0
        self._special_moves_from_neutral_per_episode_counter = 0
        self._combo_stance_steps_per_episode_counter = 0
        self._combo_armed_steps_per_episode_counter = 0
        self._combo_toggle_count_per_episode_counter = 0
        self._combo_followups_queued_per_episode_counter = 0

        self._prev_p1_move = None  # used to make sure special moves are only counted when they are performed, and not every time step they are active
        self._prev_combo_stance_active = False

    def _get_raw_env(self):
        return self.env.unwrapped

    def _get_p1_move(self) -> FootsiesMove | None:
        state = getattr(self._get_raw_env(), "_current_state", None)
        if state is None:
            return None

        return MOVE_ID_TO_MOVE.get(state.p1Move)

    def _get_combo_flags(self) -> tuple[bool, bool]:
        raw_env = self._get_raw_env()
        return bool(raw_env.combo_stance_active), bool(raw_env.combo_armed)

    def _reset_episode_counters(self):
        self._special_moves_per_episode_counter = 0
        self._special_moves_from_neutral_per_episode_counter = 0
        self._combo_stance_steps_per_episode_counter = 0
        self._combo_armed_steps_per_episode_counter = 0
        self._combo_toggle_count_per_episode_counter = 0
        self._combo_followups_queued_per_episode_counter = 0

    def _finalize_episode(self, reward: float):
        self._special_moves_per_episode.append(self._special_moves_per_episode_counter)
        self._special_moves_from_neutral_per_episode.append(
            self._special_moves_from_neutral_per_episode_counter
        )
        self._combo_stance_steps_per_episode.append(
            self._combo_stance_steps_per_episode_counter
        )
        self._combo_armed_steps_per_episode.append(
            self._combo_armed_steps_per_episode_counter
        )
        self._combo_toggle_count_per_episode.append(
            self._combo_toggle_count_per_episode_counter
        )
        self._combo_followups_queued_per_episode.append(
            self._combo_followups_queued_per_episode_counter
        )
        self._wins_per_episode.append(1 if reward > 0 else 0)
        self._reset_episode_counters()

    def reset(self, *, seed: int = None, options: dict = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._prev_p1_move = self._get_p1_move()
        combo_stance_active, _ = self._get_combo_flags()
        self._prev_combo_stance_active = combo_stance_active
        self._reset_episode_counters()

        return obs, info

    def step(self, action):
        next_obs, reward, terminated, truncated, info = self.env.step(action)

        p1_move = self._get_p1_move()
        if self._prev_p1_move != p1_move and p1_move in {
            FootsiesMove.B_SPECIAL,
            FootsiesMove.N_SPECIAL,
        }:
            self._special_moves_per_episode_counter += 1

            if self._prev_p1_move not in {
                FootsiesMove.B_ATTACK,
                FootsiesMove.N_ATTACK,
            }:
                self._special_moves_from_neutral_per_episode_counter += 1

        combo_stance_active, combo_armed = self._get_combo_flags()
        if combo_stance_active:
            self._combo_stance_steps_per_episode_counter += 1
        if combo_armed:
            self._combo_armed_steps_per_episode_counter += 1
        if combo_stance_active != self._prev_combo_stance_active:
            self._combo_toggle_count_per_episode_counter += 1

        if getattr(self._get_raw_env(), "combo_queued", False):
            self._combo_followups_queued_per_episode_counter += 1
        
        self._prev_p1_move = p1_move
        self._prev_combo_stance_active = combo_stance_active

        if terminated or truncated:
            self._finalize_episode(reward)

        return next_obs, reward, terminated, truncated, info

    @property
    def metric_special_moves_per_episode(self):
        return self._special_moves_per_episode

    @property
    def metric_special_moves_from_neutral_per_episode(self):
        return self._special_moves_from_neutral_per_episode

    @property
    def metric_combo_stance_steps_per_episode(self):
        return self._combo_stance_steps_per_episode

    @property
    def metric_combo_armed_steps_per_episode(self):
        return self._combo_armed_steps_per_episode

    @property
    def metric_combo_toggle_count_per_episode(self):
        return self._combo_toggle_count_per_episode

    @property
    def metric_combo_followups_queued_per_episode(self):
        return self._combo_followups_queued_per_episode

    @property
    def metric_wins_per_episode(self):
        return self._wins_per_episode

    def summary(self) -> dict:
        total_episodes = len(self.metric_wins_per_episode)
        if total_episodes == 0:
            return {
                "episodes": 0,
                "special_moves_average": 0.0,
                "special_moves_total": 0,
                "special_moves_from_neutral_average": 0.0,
                "special_moves_from_neutral_total": 0,
                "combo_stance_steps_average": 0.0,
                "combo_stance_steps_total": 0,
                "combo_armed_steps_average": 0.0,
                "combo_armed_steps_total": 0,
                "combo_toggles_average": 0.0,
                "combo_toggles_total": 0,
                "combo_followups_queued_average": 0.0,
                "combo_followups_queued_total": 0,
                "wins": 0,
                "winrate": 0.0,
            }

        total_special_moves = sum(self.metric_special_moves_per_episode)
        total_special_moves_from_neutral = sum(
            self.metric_special_moves_from_neutral_per_episode
        )
        total_combo_stance_steps = sum(self.metric_combo_stance_steps_per_episode)
        total_combo_armed_steps = sum(self.metric_combo_armed_steps_per_episode)
        total_combo_toggles = sum(self.metric_combo_toggle_count_per_episode)
        total_combo_followups_queued = sum(
            self.metric_combo_followups_queued_per_episode
        )
        total_wins = sum(self.metric_wins_per_episode)

        return {
            "episodes": total_episodes,
            "special_moves_average": total_special_moves / total_episodes,
            "special_moves_total": total_special_moves,
            "special_moves_from_neutral_average": total_special_moves_from_neutral / total_episodes,
            "special_moves_from_neutral_total": total_special_moves_from_neutral,
            "combo_stance_steps_average": total_combo_stance_steps / total_episodes,
            "combo_stance_steps_total": total_combo_stance_steps,
            "combo_armed_steps_average": total_combo_armed_steps / total_episodes,
            "combo_armed_steps_total": total_combo_armed_steps,
            "combo_toggles_average": total_combo_toggles / total_episodes,
            "combo_toggles_total": total_combo_toggles,
            "combo_followups_queued_average": total_combo_followups_queued / total_episodes,
            "combo_followups_queued_total": total_combo_followups_queued,
            "wins": total_wins,
            "winrate": total_wins / total_episodes,
        }
    
    def report(self):
        summary = self.summary()

        print("Report")
        print(f" Episodes: {summary['episodes']}")
        print(" Results")
        print(f"  Wins: {summary['wins']}")
        print(f"  Winrate: {summary['winrate']:.3f}")
        print(" Special moves")
        print(f"  Average: {summary['special_moves_average']}")
        print(f"  Total: {summary['special_moves_total']}")
        print(" Special moves from neutral")
        print(f"  Average: {summary['special_moves_from_neutral_average']}")
        print(f"  Total: {summary['special_moves_from_neutral_total']}")
        print(" Combo stance")
        print(f"  Steps average: {summary['combo_stance_steps_average']}")
        print(f"  Steps total: {summary['combo_stance_steps_total']}")
        print(f"  Toggles average: {summary['combo_toggles_average']}")
        print(f"  Toggles total: {summary['combo_toggles_total']}")
        print(" Combo armed")
        print(f"  Steps average: {summary['combo_armed_steps_average']}")
        print(f"  Steps total: {summary['combo_armed_steps_total']}")
        print(" Combo follow-ups queued")
        print(f"  Average: {summary['combo_followups_queued_average']}")
        print(f"  Total: {summary['combo_followups_queued_total']}")